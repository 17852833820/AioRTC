import asyncio
import datetime
import logging
import random
import time

from .codecs import get_decoder
from .exceptions import InvalidStateError
from .jitterbuffer import JitterBuffer
from .mediastreams import MediaStreamTrack
from .rtp import (RTP_SEQ_MODULO, RtcpReceiverInfo, RtcpRrPacket,
                  RtcpRtpfbPacket, RtcpSrPacket, clamp_packets_lost,
                  datetime_from_ntp, seq_gt, seq_plus_one)
from .stats import (RTCRemoteInboundRtpStreamStats,
                    RTCRemoteOutboundRtpStreamStats)
from .utils import first_completed

logger = logging.getLogger('rtp')


class NackGenerator:
    def __init__(self, callback):
        self.callback = callback
        self.max_seq = None
        self.missing = None
        self.ssrc = None

    async def add(self, packet):
        if packet.ssrc != self.ssrc:
            self.max_seq = packet.sequence_number
            self.missing = set()
            self.ssrc = packet.ssrc
            return

        if seq_gt(packet.sequence_number, self.max_seq):
            # mark missing packets
            missed = 0
            seq = seq_plus_one(self.max_seq)
            while seq_gt(packet.sequence_number, seq):
                self.missing.add(seq)
                missed += 1
                seq = seq_plus_one(seq)
            self.max_seq = packet.sequence_number

            # trigger a NACK if needed
            if missed:
                await self.callback(self.ssrc, sorted(self.missing))
        else:
            self.missing.discard(packet.sequence_number)


class StreamStatistics:
    def __init__(self, ssrc, clockrate):
        self.base_seq = None
        self.max_seq = None
        self.cycles = 0
        self.packets_received = 0
        self.ssrc = ssrc

        # jitter
        self._clockrate = clockrate
        self._jitter_q4 = 0
        self._last_arrival = None
        self._last_timestamp = None

        # fraction lost
        self._expected_prior = 0
        self._received_prior = 0

    def add(self, packet):
        in_order = self.max_seq is None or seq_gt(packet.sequence_number, self.max_seq)
        self.packets_received += 1

        if self.base_seq is None:
            self.base_seq = packet.sequence_number

        if in_order:
            arrival = int(time.time() * self._clockrate)

            if self.max_seq is not None and packet.sequence_number < self.max_seq:
                self.cycles += RTP_SEQ_MODULO
            self.max_seq = packet.sequence_number

            if packet.timestamp != self._last_timestamp and self.packets_received > 1:
                diff = abs((arrival - self._last_arrival) -
                           (packet.timestamp - self._last_timestamp))
                self._jitter_q4 += diff - ((self._jitter_q4 + 8) >> 4)

            self._last_arrival = arrival
            self._last_timestamp = packet.timestamp

    @property
    def fraction_lost(self):
        expected_interval = self.packets_expected - self._expected_prior
        self._expected_prior = self.packets_expected
        received_interval = self.packets_received - self._received_prior
        self._received_prior = self.packets_received
        lost_interval = expected_interval - received_interval
        if (expected_interval == 0 or lost_interval <= 0):
            return 0
        else:
            return (lost_interval << 8) // expected_interval

    @property
    def jitter(self):
        return self._jitter_q4 >> 4

    @property
    def packets_expected(self):
        return self.cycles + self.max_seq - self.base_seq + 1

    @property
    def packets_lost(self):
        return clamp_packets_lost(self.packets_expected - self.packets_received)


class RemoteStreamTrack(MediaStreamTrack):
    def __init__(self, kind):
        self.kind = kind
        self._queue = asyncio.Queue()

    async def recv(self):
        return await self._queue.get()


class RTCRtpReceiver:
    """
    The :class:`RTCRtpReceiver` interface manages the reception and decoding
    of data for a :class:`MediaStreamTrack`.

    :param: kind: The kind of media (`'audio'` or `'video'`).
    :param: transport: An :class:`RTCDtlsTransport`.
    """
    def __init__(self, kind, transport):
        if transport.state == 'closed':
            raise InvalidStateError

        self.__codecs = {}
        self.__decoders = {}
        self._kind = kind
        self._jitter_buffer = JitterBuffer(capacity=128)
        self.__nack_generator = NackGenerator(self._send_rtcp_nack)
        self._track = None
        self.__rtcp_exited = asyncio.Event()
        self.__sender = None
        self.__started = False
        self._stats = {}
        self.__stopped = asyncio.Event()
        self.__transport = transport

        # RTCP
        self._ssrc = None
        self.__lsr = None
        self.__lsr_stamp = None
        self.__remote_counter = None

    @property
    def transport(self):
        """
        The :class:`RTCDtlsTransport` over which the media for the receiver's
        track is received.
        """
        return self.__transport

    async def receive(self, parameters):
        """
        Attempt to set the parameters controlling the receiving of media.

        :param: parameters: The :class:`RTCRtpParameters` for the receiver.
        """
        if not self.__started:
            for codec in parameters.codecs:
                self.__codecs[codec.payloadType] = codec
                self.__decoders[codec.payloadType] = get_decoder(codec)
            self.__transport._register_rtp_receiver(self, parameters)
            asyncio.ensure_future(self._run_rtcp())
            self.__started = True

    def setTransport(self, transport):
        self.__transport = transport

    async def stop(self):
        """
        Irreversibly stop the receiver.
        """
        if self.__started:
            self.__stopped.set()
            await self.__rtcp_exited.wait()

    async def _handle_rtcp_packet(self, packet):
        self.__log_debug('< %s', packet)

        if isinstance(packet, RtcpSrPacket):
            stats = RTCRemoteOutboundRtpStreamStats(
                # RTCStats
                timestamp=datetime.datetime.now(),
                type='remote-outbound-rtp',
                id=str(id(self)),
                # RTCStreamStats
                ssrc=packet.ssrc,
                kind=self._kind,
                transportId=str(id(self.transport)),
                # RTCSentRtpStreamStats
                packetsSent=packet.sender_info.packet_count,
                bytesSent=packet.sender_info.octet_count,
                # RTCRemoteOutboundRtpStreamStats
                localId='TODO',
                remoteTimestamp=datetime_from_ntp(packet.sender_info.ntp_timestamp)
            )
            self._stats[stats.type] = stats
            self.__lsr = ((packet.sender_info.ntp_timestamp) >> 16) & 0xffffffff
            self.__lsr_time = time.time()

        if isinstance(packet, (RtcpRrPacket, RtcpSrPacket)):
            for report in packet.reports:
                stats = RTCRemoteInboundRtpStreamStats(
                    # RTCStats
                    timestamp=datetime.datetime.now(),
                    type='remote-inbound-rtp',
                    id=str(id(self)),
                    # RTCStreamStats
                    ssrc=packet.ssrc,
                    kind=self._kind,
                    transportId=str(id(self.transport)),
                    # RTCReceivedRtpStreamStats
                    packetsReceived=0,  # FIXME: where do we get this?
                    packetsLost=report.packets_lost,
                    jitter=report.jitter,
                    # RTCRemoteInboundRtpStreamStats
                    localId='TODO',
                    roundTripTime=0,  # FIXME: where do we get this?
                    fractionLost=report.fraction_lost
                )
                self._stats[stats.type] = stats

        if isinstance(packet, RtcpRtpfbPacket) and self.__sender:
            for seq in packet.lost:
                await self.__sender._retransmit(seq)

    async def _handle_rtp_packet(self, packet):
        self.__log_debug('< %s', packet)
        if packet.payload_type in self.__decoders:
            decoder = self.__decoders[packet.payload_type]
            loop = asyncio.get_event_loop()

            # RTCP
            if self.__remote_counter is None or self.__remote_counter.ssrc != packet.ssrc:
                codec = self.__codecs[packet.payload_type]
                self.__remote_counter = StreamStatistics(packet.ssrc, codec.clockRate)
            self.__remote_counter.add(packet)

            if self._kind == 'audio':
                # FIXME: audio should use a jitter buffer!
                audio_frame = await loop.run_in_executor(None, decoder.decode, packet.payload)
                await self._track._queue.put(audio_frame)
            else:
                if packet.payload:
                    # Parse codec-specific information
                    decoder.parse(packet)
                else:
                    # Firefox sends empty frames
                    packet._data = b''
                    packet._first_in_frame = False
                    packet._picture_id = None

                # check if we are missing any packets
                await self.__nack_generator.add(packet)

                # check if we have a complete video frame
                encoded_frame = self._jitter_buffer.add(packet)
                if encoded_frame is not None:
                    video_frames = await loop.run_in_executor(None, decoder.decode,
                                                              encoded_frame.data)
                    for video_frame in video_frames:
                        await self._track._queue.put(video_frame)

    async def _run_rtcp(self):
        self.__log_debug('- RTCP started')

        while not self.__stopped.is_set():
            # The interval between RTCP packets is varied randomly over the
            # range [0.5, 1.5] times the calculated interval.
            sleep = 0.5 + random.random()
            result = await first_completed(asyncio.sleep(sleep), self.__stopped.wait())
            if result is True:
                break

            # RTCP RR
            if self._ssrc is not None and self.__remote_counter is not None:
                lsr = 0
                dlsr = 0
                if self.__lsr is not None:
                    lsr = self.__lsr
                    delay = time.time() - self.__lsr_time
                    if delay > 0 and delay < 65536:
                        dlsr = int(delay * 65536)

                packet = RtcpRrPacket(
                    ssrc=self._ssrc,
                    reports=[RtcpReceiverInfo(
                        ssrc=self.__remote_counter.ssrc,
                        fraction_lost=self.__remote_counter.fraction_lost,
                        packets_lost=self.__remote_counter.packets_lost,
                        highest_sequence=self.__remote_counter.max_seq,
                        jitter=self.__remote_counter.jitter,
                        lsr=lsr,
                        dlsr=dlsr)])
                await self._send_rtcp(packet)

        self.__log_debug('- RTCP finished')
        self.__rtcp_exited.set()

    async def _send_rtcp(self, packet):
        self.__log_debug('> %s', packet)
        try:
            await self.transport._send_rtp(bytes(packet))
        except ConnectionError:
            pass

    async def _send_rtcp_nack(self, media_ssrc, lost):
        if self._ssrc is not None:
            packet = RtcpRtpfbPacket(fmt=1, ssrc=self._ssrc, media_ssrc=media_ssrc)
            packet.lost = lost
            await self._send_rtcp(packet)

    def _set_sender(self, sender):
        self.__sender = sender

    def __log_debug(self, msg, *args):
        logger.debug('receiver(%s) ' + msg, self._kind, *args)
