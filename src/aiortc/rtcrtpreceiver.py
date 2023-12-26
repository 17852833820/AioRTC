import asyncio
import datetime
import logging
import queue
import random
import threading
import time
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Set
import collections

from av.frame import Frame

from . import clock
from .codecs import depayload, get_capabilities, get_decoder, is_rtx
from .exceptions import InvalidStateError
from .jitterbuffer import JitterBuffer
from .mediastreams import MediaStreamError, MediaStreamTrack
from .rate import RemoteBitrateEstimator
from .rtcdtlstransport import RTCDtlsTransport
from .rtcrtpparameters import (RTCRtpCapabilities, RTCRtpCodecParameters,
                               RTCRtpReceiveParameters)
from .rtp import (RTCP_PSFB_APP, RTCP_PSFB_PLI, RTCP_RTPFB_NACK,
                  RTP_HISTORY_SIZE, AnyRtcpPacket, RtcpByePacket,
                  RtcpPsfbPacket, RtcpReceiverInfo, RtcpRrPacket,
                  RtcpRtpfbPacket, RtcpSrPacket, RtpPacket, clamp_packets_lost,
                  pack_remb_fci, unwrap_rtx)
from .stats import (RTCInboundRtpStreamStats, RTCRemoteOutboundRtpStreamStats,
                    RTCStatsReport)
from .utils import uint16_add, uint16_gt

logger = logging.getLogger(__name__)
# abs-send-time estimator
INTER_ARRIVAL_SHIFT = 26
TIMESTAMP_GROUP_LENGTH_MS = 5
TIMESTAMP_TO_MS = 1000.0 / (1 << INTER_ARRIVAL_SHIFT)

# 解码线程主任务
def decoder_worker(loop, input_q, output_q):
    #初始化：用于追踪当前的解码器和编码器名称
    codec_name = None
    decoder = None
    __log_debug: Callable[..., None] = lambda *args: None
    if logger.isEnabledFor(logging.DEBUG):
        __log_debug = lambda msg, *args: logger.debug(
            f"RTCRtpReceiver(%s) {msg}", 'decoder_worker', *args
        )

    while True:#任务处理无限循环
        task = input_q.get() #等待从输入队列 (input_q) 获取任务
        if task is None:# 如果获取到的任务为 None，表示线程结束，将 None 放入输出队列 (output_q) 并终止循环。
            # inform the track that is has ended
            asyncio.run_coroutine_threadsafe(output_q.put(None), loop)
            break
        codec, encoded_frame = task
        if codec.name != codec_name: #检查编解码器是否匹配
            decoder = get_decoder(codec)# 实际解码器
            codec_name = codec.name
        #使用解码器对编码帧进行解码，得到解码后的帧
        for frame in decoder.decode(encoded_frame):
            
            asyncio.run_coroutine_threadsafe(output_q.put(frame), loop) #将解码后的帧（frame）放入输出队列
            __log_debug('[DECODE] Add Render Frame...Stream id: %s, Number: %d, Type: %d', encoded_frame.stream_id, frame.index, frame.pict_type)

        now = clock.current_ms()
        dec_dur = now - encoded_frame.times_dur['time_in_dec_q']
        __log_debug('[FRAME_INFO] T: %d, dec_dur: %d, Bytes: %d', encoded_frame.timestamp, dec_dur, len(encoded_frame.data))

    if decoder is not None:
        del decoder


class NackGenerator:
    def __init__(self) -> None:
        self.max_seq: Optional[int] = None
        self.missing: Set[int] = set()

    def add(self, packet: RtpPacket) -> bool:
        """
        Mark a new packet as received, and deduce missing packets.
        """
        missed = False

        if self.max_seq is None:
            self.max_seq = packet.sequence_number
            return missed

        # mark missing packets
        if uint16_gt(packet.sequence_number, self.max_seq):
            seq = uint16_add(self.max_seq, 1)
            while uint16_gt(packet.sequence_number, seq):
                self.missing.add(seq)
                missed = True
                seq = uint16_add(seq, 1)
            self.max_seq = packet.sequence_number
        else:
            self.missing.discard(packet.sequence_number)

        # limit number of tracked packets
        self.truncate()

        return missed

    def truncate(self) -> None:
        """
        Limit the number of missing packets we track.

        Otherwise, the size of RTCP FB messages grows indefinitely.
        """
        if self.max_seq is not None:
            min_seq = uint16_add(self.max_seq, -RTP_HISTORY_SIZE)
            for seq in list(self.missing):
                if uint16_gt(min_seq, seq):
                    self.missing.discard(seq)


class StreamStatistics:
    def __init__(self, clockrate: int) -> None:
        self.base_seq: Optional[int] = None
        self.max_seq: Optional[int] = None
        self.cycles = 0
        self.packets_received = 0
        self.bytes_received=0

        # jitter
        self._clockrate = clockrate
        self._jitter_q4 = 0
        self._last_arrival: Optional[int] = None
        self._last_timestamp: Optional[int] = None

        # fraction lost
        self._expected_prior = 0
        self._received_prior = 0

    def add(self, packet: RtpPacket) -> None:
        in_order = self.max_seq is None or uint16_gt(
            packet.sequence_number, self.max_seq
        )
        self.packets_received += 1
        self.bytes_received+=len(packet.payload) + packet.padding_size
        if self.base_seq is None:
            self.base_seq = packet.sequence_number

        if in_order:
            arrival = int(time.time() * self._clockrate)

            if self.max_seq is not None and packet.sequence_number < self.max_seq:
                self.cycles += 1 << 16
            self.max_seq = packet.sequence_number

            if packet.timestamp != self._last_timestamp and self.packets_received > 1:
                diff = abs(
                    (arrival - self._last_arrival)
                    - (packet.timestamp - self._last_timestamp)
                )
                self._jitter_q4 += diff - ((self._jitter_q4 + 8) >> 4)

            self._last_arrival = arrival
            self._last_timestamp = packet.timestamp

    @property
    def fraction_lost(self) -> int:
        expected_interval = self.packets_expected - self._expected_prior
        self._expected_prior = self.packets_expected
        received_interval = self.packets_received - self._received_prior
        self._received_prior = self.packets_received
        lost_interval = expected_interval - received_interval
        if expected_interval == 0 or lost_interval <= 0:
            return 0
        else:
            return (lost_interval << 8) // expected_interval

    @property
    def jitter(self) -> int:
        return self._jitter_q4 >> 4

    @property
    def packets_expected(self) -> int:
        return self.cycles + self.max_seq - self.base_seq + 1

    @property
    def packets_lost(self) -> int:
        return clamp_packets_lost(self.packets_expected - self.packets_received)
class QueueRepeat(Exception):
    """Raised when the Queue.put_nowait() method is called on a full Queue."""
    pass

class UniqueQueue(asyncio.Queue):
    def __init__(self, maxsize=0):
        self._maxsize = maxsize
        
        # Futures.
        self._getters = collections.deque()
        # Futures.
        self._putters = collections.deque()
        self._unfinished_tasks = 0
        self._finished = asyncio.locks.Event()
        self._finished.set()
        self._init(maxsize)
        self.index_list=[]
    def put_nowait(self, item):
        """Put an item into the queue without blocking.

        If no free slot is immediately available, raise QueueFull.
        """
        
        if self.full():
            raise asyncio.QueueFull
        # check index 
        if item.index not in self.index_list: # fix join bug
            self._put(item)
            self.index_list.append(item.index)
            self._unfinished_tasks += 1
            self._finished.clear()
            self._wakeup_next(self._getters)
        
class RemoteStreamTrack(MediaStreamTrack):
    def __init__(self, kind: str, id: Optional[str] = None) -> None:
        super().__init__()
        self.kind = kind
        if id is not None:
            self._id = id
        self._queue: UniqueQueue = UniqueQueue()

    async def recv(self) -> Frame:
        """
        Receive the next frame.
        """
        if self.readyState != "live":
            raise MediaStreamError

        frame = await self._queue.get()
        if frame is None:
            self.stop()
            raise MediaStreamError
        
        return frame


class TimestampMapper:
    def __init__(self) -> None:
        self._last: Optional[int] = None
        self._origin: Optional[int] = None

    def map(self, timestamp: int) -> int:
        if self._origin is None:
            # first timestamp
            self._origin = timestamp
        elif timestamp < self._last:
            # RTP timestamp wrapped
            self._origin -= 1 << 32

        self._last = timestamp
        return timestamp - self._origin


@dataclass
class RTCRtpContributingSource:
    """
    The :class:`RTCRtpContributingSource` dictionary contains information about
    a contributing source (CSRC).
    """

    timestamp: datetime.datetime
    "The timestamp associated with this source."
    source: int
    "The CSRC identifier associated with this source."


@dataclass
class RTCRtpSynchronizationSource:
    """
    The :class:`RTCRtpSynchronizationSource` dictionary contains information about
    a synchronization source (SSRC).
    """

    timestamp: datetime.datetime
    "The timestamp associated with this source."
    source: int
    "The SSRC identifier associated with this source."


class RTCRtpReceiver:
    """
    The :class:`RTCRtpReceiver` interface manages the reception and decoding
    of data for a :class:`MediaStreamTrack`.

    :param kind: The kind of media (`'audio'` or `'video'`).
    :param transport: An :class:`RTCDtlsTransport`.
    """

    def __init__(self, kind: str, transport: RTCDtlsTransport) -> None:
        if transport.state == "closed":
            raise InvalidStateError

        self.__active_ssrc: Dict[int, datetime.datetime] = {}
        self.__codecs: Dict[int, RTCRtpCodecParameters] = {}
        self.__decoder_queue: queue.Queue = queue.Queue()
        self.__decoder_queue2: queue.Queue = queue.Queue()
        self.__decoder_thread: Optional[threading.Thread] = None
        self.__decoder_thread2: Optional[threading.Thread] = None
        self.__kind = kind
        if kind == "audio":
            self.__jitter_buffer = JitterBuffer(capacity=16, prefetch=4)
            self.__nack_generator = None
            self.__remote_bitrate_estimator = None
        else:
            self.__jitter_buffer = JitterBuffer(capacity=128, is_video=True)
            self.__nack_generator = NackGenerator()
            self.__remote_bitrate_estimator = RemoteBitrateEstimator()
        self._track: Optional[RemoteStreamTrack] = None
        self.__rtcp_exited = asyncio.Event()
        self.__rtcp_started = asyncio.Event()
        self.__rtcp_task: Optional[asyncio.Future[None]] = None
        self.__rtx_ssrc: Dict[int, int] = {}
        self.__started = False
        self.__stats = RTCStatsReport()
        self.__timestamp_mapper = TimestampMapper()
        self.__transport = transport

        # RTCP
        self.__lsr: Dict[int, int] = {}
        self.__lsr_time: Dict[int, float] = {}
        self.__remote_streams: Dict[int, StreamStatistics] = {}
        self.__rtcp_ssrc: Optional[int] = None

        # logging
        self.__log_debug: Callable[..., None] = lambda *args: None
        if logger.isEnabledFor(logging.DEBUG):
            self.__log_debug = lambda msg, *args: logger.debug(
                f"RTCRtpReceiver(%s) {msg}", self.__kind, *args
            )
        # 计算frame 传输时间
        self.first_recv_time=0
        self.last_recv_time=0
        # 统计接收速率
        self.last_arrival_time=0
        self.recv_rate=0#bps
        self.last_arrival_bytes=0
        self.last_count=0
        self.recv_count=0
        # 是否采用多流
        self.use_multistream =True
    @property
    def track(self) -> MediaStreamTrack:
        """
        The :class:`MediaStreamTrack` which is being handled by the receiver.
        """
        return self._track

    @property
    def transport(self) -> RTCDtlsTransport:
        """
        The :class:`RTCDtlsTransport` over which the media for the receiver's
        track is received.
        """
        return self.__transport

    @classmethod
    def getCapabilities(self, kind) -> Optional[RTCRtpCapabilities]:
        """
        Returns the most optimistic view of the system's capabilities for
        receiving media of the given `kind`.

        :rtype: :class:`RTCRtpCapabilities`
        """
        return get_capabilities(kind)

    async def getStats(self) -> RTCStatsReport:
        """
        Returns statistics about the RTP receiver.

        :rtype: :class:`RTCStatsReport`
        """
        for ssrc, stream in self.__remote_streams.items():
            self.__stats.add(
                RTCInboundRtpStreamStats(
                    # RTCStats
                    timestamp=clock.current_datetime(),
                    type="inbound-rtp",
                    id="inbound-rtp_" + str(id(self)),
                    # RTCStreamStats
                    ssrc=ssrc,
                    kind=self.__kind,
                    transportId=self.transport._stats_id,
                    # RTCReceivedRtpStreamStats
                    packetsReceived=stream.packets_received,
                    packetsLost=stream.packets_lost,
                    jitter=stream.jitter,
                    # RTPInboundRtpStreamStats
                )
            )
        self.__stats.update(self.transport._get_stats())

        return self.__stats

    def getSynchronizationSources(self) -> List[RTCRtpSynchronizationSource]:
        """
        Returns a :class:`RTCRtpSynchronizationSource` for each unique SSRC identifier
        received in the last 10 seconds.
        """
        cutoff = clock.current_datetime() - datetime.timedelta(seconds=10)
        sources = []
        for source, timestamp in self.__active_ssrc.items():
            if timestamp >= cutoff:
                sources.append(
                    RTCRtpSynchronizationSource(source=source, timestamp=timestamp)
                )
        return sources

    async def receive(self, parameters: RTCRtpReceiveParameters) -> None:
        """
        Attempt to set the parameters controlling the receiving of media.

        :param parameters: The :class:`RTCRtpParameters` for the receiver.
        """
        if not self.__started:
            for codec in parameters.codecs:
                self.__codecs[codec.payloadType] = codec
            for encoding in parameters.encodings:
                if encoding.rtx:
                    self.__rtx_ssrc[encoding.rtx.ssrc] = encoding.ssrc

            # start decoder thread 启动一个用于解码传入媒体数据包的单独线程，该线程运行 decoder_worker 函数
            self.__decoder_thread = threading.Thread(
                target=decoder_worker,
                name=self.__kind + "-decoder1",
                args=(
                    asyncio.get_event_loop(),
                    self.__decoder_queue,#待解码的队列
                    self._track._queue,#解码后输出的队列
                ),
            )
            self.__decoder_thread.start()
            #多流解码
            if self.use_multistream:
                self.__decoder_thread2 = threading.Thread(
                    target=decoder_worker,
                    name=self.__kind + "-decoder2",
                    args=(
                        asyncio.get_event_loop(),
                        self.__decoder_queue2,#待解码的队列
                        self._track._queue,#解码后输出的队列
                    ),
                )
                self.__decoder_thread2.start()

            self.__transport._register_rtp_receiver(self, parameters)
            self.__rtcp_task = asyncio.ensure_future(self._run_rtcp())
            self.__started = True

    def setTransport(self, transport: RTCDtlsTransport) -> None:
        self.__transport = transport

    async def stop(self) -> None:
        """
        Irreversibly stop the receiver.
        """
        if self.__started:
            self.__transport._unregister_rtp_receiver(self)
            self.__stop_decoder()

            # shutdown RTCP task
            await self.__rtcp_started.wait()
            self.__rtcp_task.cancel()
            await self.__rtcp_exited.wait()

    def _handle_disconnect(self) -> None:
        self.__stop_decoder()

    async def _handle_rtcp_packet(self, packet: AnyRtcpPacket) -> None:
        self.__log_debug("< %s", packet)
        if isinstance(packet, RtcpSrPacket):
            self.__stats.add(
                RTCRemoteOutboundRtpStreamStats(#记录远程传输统计信息，如发送的数据包数、发送的字节数等
                    # RTCStats
                    timestamp=clock.current_datetime(),
                    type="remote-outbound-rtp",
                    id=f"remote-outbound-rtp_{id(self)}",
                    # RTCStreamStats
                    ssrc=packet.ssrc,
                    kind=self.__kind,
                    transportId=self.transport._stats_id,
                    # RTCSentRtpStreamStats
                    packetsSent=packet.sender_info.packet_count,
                    bytesSent=packet.sender_info.octet_count,
                    # RTCRemoteOutboundRtpStreamStats
                    remoteTimestamp=clock.datetime_from_ntp(
                        packet.sender_info.ntp_timestamp
                    ),
                )
            )
            self.__lsr[packet.ssrc] = (#更新最后 SR（LSR）时间戳和时间
                (packet.sender_info.ntp_timestamp) >> 16
            ) & 0xFFFFFFFF
            self.__lsr_time[packet.ssrc] = time.time()
        elif isinstance(packet, RtcpByePacket):
            self.__stop_decoder()

    async def _handle_rtp_packet(self, packet: RtpPacket, arrival_time_ms: int,arrival_24NTP_time:int) -> None:
        """
        Handle an incoming RTP packet. 处理接收的RTP数据包
        """
        self.__log_debug("< %s", packet)

        # feed bitrate estimato 如果启用了远程带宽估计：收到RTP包时更新带宽估计
        if self.__remote_bitrate_estimator is not None:
            if packet.extensions.abs_send_time is not None:#RTP 包包含绝对发送时间
                remb = self.__remote_bitrate_estimator.add(
                    abs_send_time=packet.extensions.abs_send_time,
                    arrival_time_ms=arrival_time_ms,
                    payload_size=len(packet.payload) + packet.padding_size,
                    ssrc=packet.ssrc,
                )
                if self.__rtcp_ssrc is not None and remb is not None: 
                    # send Receiver Estimated Maximum Bitrate feedback 发送REMB包反馈估计的最大比特率
                    rtcp_packet = RtcpPsfbPacket(
                        fmt=RTCP_PSFB_APP,
                        ssrc=self.__rtcp_ssrc,
                        media_ssrc=0,
                        fci=pack_remb_fci(*remb),
                    )
                    await self._send_rtcp(rtcp_packet)
        # 计算 frame 传输延迟
        # if packet.extensions.abs_send_time is not None and packet.extensions.marker_first=="1": #第一个packet的发送时间
        #     self.first_send_time=packet.extensions.abs_send_time
        # if packet.marker: #最后一个packet的接收时间
        #     current_24NTP_time= ( clock.current_ntp_time() >> 14 ) & 0x00FFFFFF 
        #     self.last_recv_time=current_24NTP_time #current ms
        #     self.__log_debug('[FRAME_INFO]  transport dur: %d ms', self.last_recv_time-self.first_send_time)
        # 计算frame 延迟：最后一个包的接收时间-第一个包的接收时间+rtt/2    
        if  packet.marker: #
            self.last_recv_time=arrival_time_ms
            self.__log_debug('[FRAME_INFO] T: %d ,  frame packet dur: %d ms',packet.timestamp,self.last_recv_time-self.first_recv_time)
        if packet.extensions.marker_first:
            self.first_recv_time=arrival_time_ms
           
        # keep track of sources
        self.__active_ssrc[packet.ssrc] = clock.current_datetime()

        # check the codec is known 检查编解码器
        codec = self.__codecs.get(packet.payload_type)
        if codec is None:
            self.__log_debug(
                "x RTP packet with unknown payload type %d", packet.payload_type
            )
            return

        # feed RTCP statistics
        if packet.ssrc not in self.__remote_streams:
            self.__remote_streams[packet.ssrc] = StreamStatistics(codec.clockRate)
        self.__remote_streams[packet.ssrc].add(packet)
        # 统计接收速率
        if arrival_time_ms-self.last_arrival_time>1000: 
                for ssrc, stream in self.__remote_streams.items():
                    self.recv_rate=((stream.bytes_received-self.last_arrival_bytes)*8)/((arrival_time_ms-self.last_arrival_time)/1000)
                    self.recv_count=stream.packets_received-self.last_count
                    self.__log_debug('[Recv_INFO] ssrc: %d, timestamp: %d,recv rate: %f bps, count_received: %d', ssrc,arrival_time_ms,self.recv_rate, self.recv_count)
                    self.last_arrival_bytes=stream.bytes_received
                    self.last_count=stream.packets_received
                    self.last_arrival_time=arrival_time_ms
                    
        # unwrap retransmission packet 
        if is_rtx(codec):
            original_ssrc = self.__rtx_ssrc.get(packet.ssrc)
            if original_ssrc is None:
                self.__log_debug("x RTX packet from unknown SSRC %d", packet.ssrc)
                return

            if len(packet.payload) < 2:
                return

            codec = self.__codecs[codec.parameters["apt"]]
            packet = unwrap_rtx(
                packet, payload_type=codec.payloadType, ssrc=original_ssrc
            )

        # send NACKs for any missing any packets
        if self.__nack_generator is not None and self.__nack_generator.add(packet):
            await self._send_rtcp_nack(
                packet.ssrc, sorted(self.__nack_generator.missing)
            )

        # parse codec-specific information 解析 RTP 包的负载，获取编码帧的相关信息
        try:
            if packet.payload:
                packet._data = depayload(codec, packet.payload)  # type: ignore
            else:
                packet._data = b""  # type: ignore
        except ValueError as exc:
            self.__log_debug("x RTP payload parsing failed: %s", exc)
            return

        # try to re-assemble encoded frame 尝试重新组装编码帧
        pli_flag, (encoded_frame, jit_dur) = self.__jitter_buffer.add(packet)#向抖动缓冲区（Jitter Buffer）添加 RTP 包
        if jit_dur is not None: 
            self.__log_debug('[FRAME_INFO] T: %d, jit_dur: %d, Bytes: %d', encoded_frame.timestamp, jit_dur, len(encoded_frame.data))
        # check if the PLI should be sent 如果成功获得完整的编码帧，检查是否需要发送PLI RTCP包
        if pli_flag:
            await self._send_rtcp_pli(packet.ssrc)

        # if we have a complete encoded frame, decode it 将解码请求放入解码器队列__decoder_queue
        if encoded_frame is not None and self.__decoder_thread:
            encoded_frame.timestamp = self.__timestamp_mapper.map(
                encoded_frame.timestamp
            )#获得了frame的pts
            encoded_frame.times_dur['time_in_dec_q'] = clock.current_ms()
            if self.use_multistream and encoded_frame.stream_id == "2":
                self.__decoder_queue2.put((codec, encoded_frame))
            else:
                self.__decoder_queue.put((codec, encoded_frame))


    async def _run_rtcp(self) -> None:
        self.__log_debug("- RTCP started")
        self.__rtcp_started.set()

        try:
            while True:
                # The interval between RTCP packets is varied randomly over the
                # range [0.5, 1.5] times the calculated interval.
                await asyncio.sleep(0.5 + random.random())

                # RTCP RR
                reports = []
                for ssrc, stream in self.__remote_streams.items():
                    lsr = 0
                    dlsr = 0
                    if ssrc in self.__lsr:
                        lsr = self.__lsr[ssrc]
                        delay = time.time() - self.__lsr_time[ssrc]
                        if delay > 0 and delay < 65536:
                            dlsr = int(delay * 65536)

                    reports.append(
                        RtcpReceiverInfo(
                            ssrc=ssrc,
                            fraction_lost=stream.fraction_lost,
                            packets_lost=stream.packets_lost,
                            highest_sequence=stream.max_seq,
                            jitter=stream.jitter,
                            lsr=lsr,
                            dlsr=dlsr,
                        )
                    )

                if self.__rtcp_ssrc is not None and reports:
                    packet = RtcpRrPacket(ssrc=self.__rtcp_ssrc, reports=reports)
                    await self._send_rtcp(packet)

        except asyncio.CancelledError:
            pass

        self.__log_debug("- RTCP finished")
        self.__rtcp_exited.set()

    async def _send_rtcp(self, packet) -> None:
        self.__log_debug("> %s", packet)
        try:
            await self.transport._send_rtp(bytes(packet))
        except ConnectionError:
            pass

    async def _send_rtcp_nack(self, media_ssrc: int, lost: List[int]) -> None:
        """
        Send an RTCP packet to report missing RTP packets.
        """
        if self.__rtcp_ssrc is not None:
            packet = RtcpRtpfbPacket(
                fmt=RTCP_RTPFB_NACK, ssrc=self.__rtcp_ssrc, media_ssrc=media_ssrc
            )
            packet.lost = lost
            await self._send_rtcp(packet)

    async def _send_rtcp_pli(self, media_ssrc: int) -> None:
        """
        Send an RTCP packet to report picture loss.
        """
        if self.__rtcp_ssrc is not None:
            packet = RtcpPsfbPacket(
                fmt=RTCP_PSFB_PLI, ssrc=self.__rtcp_ssrc, media_ssrc=media_ssrc
            )
            await self._send_rtcp(packet)

    def _set_rtcp_ssrc(self, ssrc: int) -> None:
        self.__rtcp_ssrc = ssrc

    def __stop_decoder(self) -> None:
        """
        Stop the decoder thread, which will in turn stop the track.
        """
        if self.__decoder_thread:
            self.__decoder_queue.put(None)
            self.__decoder_thread.join()
            self.__decoder_thread = None
        if self.__decoder_thread2:
            self.__decoder_queue2.put(None)
            self.__decoder_thread2.join()
            self.__decoder_thread2 = None
