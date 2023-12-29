import asyncio
import logging
import random
import time
import traceback
import uuid
from typing import Callable, Dict, List, Optional, Union

import cv2
from av import AudioFrame
from av.frame import Frame
from .pacer.pacedsender import PacedSender
from .pacer.roundrobinpacketqueue import RtpPacketToSend,RtpPacketMediaType
from . import clock, rtp
from .codecs import get_capabilities, get_encoder, is_rtx
from .codecs.base import Encoder
from .exceptions import InvalidStateError
from .mediastreams import MediaStreamError, MediaStreamTrack
from .rtcrtpparameters import RTCRtpCodecParameters, RTCRtpSendParameters
from .rtp import (RTCP_PSFB_APP, RTCP_PSFB_PLI, RTCP_RTPFB_NACK,
                  RTP_HISTORY_SIZE, AnyRtcpPacket, RtcpByePacket,
                  RtcpPsfbPacket, RtcpRrPacket, RtcpRtpfbPacket,
                  RtcpSdesPacket, RtcpSenderInfo, RtcpSourceInfo, RtcpSrPacket,
                  RtpPacket, unpack_remb_fci, wrap_rtx)
from .stats import (RTCOutboundRtpStreamStats, RTCRemoteInboundRtpStreamStats,
                    RTCStatsReport)
from .utils import random16, random32, uint16_add, uint32_add
from .codecs.h264 import DEFAULT_BITRATE
logger = logging.getLogger(__name__)

RTT_ALPHA = 0.85

class MultiEncodeMode:
    def __init__(self):
        self.current_state = "S0"

    def transition(self):
        if self.current_state == "S0":
            self.current_state = "S1"
        elif self.current_state == "S1":
            self.current_state = "S2"
        elif self.current_state == "S2":
            self.current_state = "S3"
        elif self.current_state == "S3":
            self.current_state = "S1"

class RTCEncodedFrame:
    def __init__(self, payloads: List[bytes], timestamp: int, audio_level: int):
        self.payloads = payloads
        self.timestamp = timestamp
        self.audio_level = audio_level
  
class RTCRtpSender():
    """
    The :class:`RTCRtpSender` interface provides the ability to control and
    obtain details about how a particular :class:`MediaStreamTrack` is encoded
    and sent to a remote peer.

    :param trackOrKind: Either a :class:`MediaStreamTrack` instance or a
                         media kind (`'audio'` or `'video'`).
    :param transport: An :class:`RTCDtlsTransport`.
    """

    def __init__(self, trackOrKind: Union[MediaStreamTrack, str], transport) -> None:
        if transport.state == "closed":
            raise InvalidStateError
        
        if isinstance(trackOrKind, MediaStreamTrack):
            self.__kind = trackOrKind.kind
            self.replaceTrack(trackOrKind)
        else:
            self.__kind = trackOrKind
            self.replaceTrack(None)
        self.__cname: Optional[str] = None
        self._ssrc = random32()
        self._rtx_ssrc = random32()
        # FIXME: how should this be initialised?
        self._stream_id = str(uuid.uuid4())
        self.__encoder: Optional[Encoder] = None
        self.__force_keyframe = False
        self.__loop = asyncio.get_event_loop()
        self.__mid: Optional[str] = None
        self.__rtp_exited = asyncio.Event()
        self.__rtp_header_extensions_map = rtp.HeaderExtensionsMap()
        self.__rtp_started = asyncio.Event()
        self.__rtp_task: Optional[asyncio.Future[None]] = None
        self.__rtp_history: Dict[int, RtpPacket] = {}
        self.__rtcp_exited = asyncio.Event()
        self.__rtcp_started = asyncio.Event()
        self.__rtcp_task: Optional[asyncio.Future[None]] = None
        self.__rtx_payload_type: Optional[int] = None
        self.__rtx_sequence_number = random16()
        self.__started = False
        self.__stats = RTCStatsReport()
        self.__transport = transport

        # stats
        self.__lsr: Optional[int] = None
        self.__lsr_time: Optional[float] = None
        self.__ntp_timestamp = 0
        self.__rtp_timestamp = 0
        self.__octet_count = 0
        self.__packet_count = 0
        self.__rtt = None
        self.send_rate=0 #bps
        self.last_octet=0
        self.timestamp=0
        self.last_timestamp=0
        self.last_count=0
        # logging
        self.__log_debug: Callable[..., None] = lambda *args: None
        if logger.isEnabledFor(logging.DEBUG):
            self.__log_debug = lambda msg, *args: logger.debug(
                f"RTCRtpSender(%s) {msg}", self.__kind, *args
            )
        # Multi Encoder
        # self.condition_IDR_event = asyncio.Event()
        # self.condition_Finish_event = asyncio.Event()
        self.lock = asyncio.Lock()
        self._data=None
        self.__encoder_two: Optional[Encoder] = None
        # self.IDR_receive_finished=False
        self.use_multistream =True
        self.encode_mode=MultiEncodeMode()
        self.encode_role_forwart=False #前向：stream1向stream2切换，否则stream2向stream1切换
        # Pacer
        self.pace_sender:PacedSender=PacedSender()
    @property
    def kind(self):
        return self.__kind

    @property
    def track(self) -> MediaStreamTrack:
        """
        The :class:`MediaStreamTrack` which is being handled by the sender.
        """
        return self.__track

    @property
    def transport(self):
        """
        The :class:`RTCDtlsTransport` over which media data for the track is
        transmitted.
        """
        return self.__transport

    @classmethod
    def getCapabilities(self, kind):
        """
        Returns the most optimistic view of the system's capabilities for
        sending media of the given `kind`.

        :rtype: :class:`RTCRtpCapabilities`
        """
        return get_capabilities(kind)

    async def getStats(self) -> RTCStatsReport:
        """
        Returns statistics about the RTP sender.

        :rtype: :class:`RTCStatsReport`
        """
        self.__stats.add(
            RTCOutboundRtpStreamStats(
                # RTCStats
                timestamp=clock.current_datetime(),
                type="outbound-rtp",
                id="outbound-rtp_" + str(id(self)),
                # RTCStreamStats
                ssrc=self._ssrc,
                kind=self.__kind,
                transportId=self.transport._stats_id,
                # RTCSentRtpStreamStats
                packetsSent=self.__packet_count,
                bytesSent=self.__octet_count,
                # RTCOutboundRtpStreamStats
                trackId=str(id(self.track)),
            )
        )
        # self.__log_debug('[Sender_INFO] timestamp: %d, packetsSent: %d ,bytesSent: %d',timestamp_origin, sequence_number)
        self.__stats.update(self.transport._get_stats())

        return self.__stats

    def replaceTrack(self, track: Optional[MediaStreamTrack]) -> None:
        self.__track = track
        if track is not None:
            self._track_id = track.id
        else:
            self._track_id = str(uuid.uuid4())

    def setTransport(self, transport) -> None:
        self.__transport = transport

    async def send(self, parameters: RTCRtpSendParameters) -> None:
        """
        Attempt to set the parameters controlling the sending of media.

        :param parameters: The :class:`RTCRtpSendParameters` for the sender.
        """
        if not self.__started:#如果发送器未启动，设置相关发送参数
            self.__cname = parameters.rtcp.cname
            self.__mid = parameters.muxId

            # make note of the RTP header extension IDs 
            self.__transport._register_rtp_sender(self, parameters)#注册RTP发送器
            self.__rtp_header_extensions_map.configure(parameters)#配置RTP头扩展

            # make note of RTX payload type
            for codec in parameters.codecs: #遍历传入参数中的编解码器
                if (
                    is_rtx(codec)
                    and codec.parameters["apt"] == parameters.codecs[0].payloadType
                ):#查找RTX编解码器并记录其负载类型
                    self.__rtx_payload_type = codec.payloadType
                    break
            # Version 2:设置Pacer初始比特率
            self.pace_sender.set_pacing_rates(DEFAULT_BITRATE,DEFAULT_BITRATE)
            #启动RTP和RTCP任务：分别启动异步任务_run_rtp和_run_rtcp
            if self.use_multistream:
                self.__rtp_task = asyncio.ensure_future(self._run_rtp_multi_stream(parameters.codecs[0]))
            else:
                self.__rtp_task = asyncio.ensure_future(self._run_rtp(parameters.codecs[0]))
            self.__rtcp_task = asyncio.ensure_future(self._run_rtcp())
            self.__started = True

    async def stop(self):
        """
        Irreversibly stop the sender.
        """
        if self.__started:
            self.__transport._unregister_rtp_sender(self)

            # shutdown RTP and RTCP tasks
            await asyncio.gather(self.__rtp_started.wait(), self.__rtcp_started.wait())
            self.__rtp_task.cancel()
            self.__rtcp_task.cancel()
            await asyncio.gather(self.__rtp_exited.wait(), self.__rtcp_exited.wait())

    async def _handle_rtcp_packet(self, packet):
        #处理 RR 和 SR 类型的 RTCP 包
        if isinstance(packet, (RtcpRrPacket, RtcpSrPacket)):
            for report in filter(lambda x: x.ssrc == self._ssrc, packet.reports):
                # estimate round-trip time
                if self.__lsr == report.lsr and report.dlsr:
                    rtt = time.time() - self.__lsr_time - (report.dlsr / 65536)#接收RR包的时间-发送上一个SR包的时间-dlsr（接收端发送RR包-接收端接收SR包）
                    self.__log_debug('[FRAME_INFO] RTT: %f ms', rtt*1000)
                    if self.__rtt is None:
                        self.__rtt = rtt
                    else:
                        self.__rtt = RTT_ALPHA * self.__rtt + (1 - RTT_ALPHA) * rtt
                # self.__log_debug('[FRAME_INFO] loss rate: %f %', report.packets_lost)

                self.__stats.add(
                    RTCRemoteInboundRtpStreamStats(
                        # RTCStats
                        timestamp=clock.current_datetime(),
                        type="remote-inbound-rtp",
                        id="remote-inbound-rtp_" + str(id(self)),
                        # RTCStreamStats
                        ssrc=packet.ssrc,
                        kind=self.__kind,
                        transportId=self.transport._stats_id,
                        # RTCReceivedRtpStreamStats
                        packetsReceived=self.__packet_count - report.packets_lost,
                        packetsLost=report.packets_lost,
                        jitter=report.jitter,
                        # RTCRemoteInboundRtpStreamStats
                        roundTripTime=self.__rtt,
                        fractionLost=report.fraction_lost,
                    )
                )
        #处理 NACK 类型的 RTCP 包：请求重传
        elif isinstance(packet, RtcpRtpfbPacket) and packet.fmt == RTCP_RTPFB_NACK:
            for seq in packet.lost:
                await self._retransmit(seq)
        #处理 PLI 类型的 RTCP 包：请求关键帧
        elif isinstance(packet, RtcpPsfbPacket) and packet.fmt == RTCP_PSFB_PLI:
            self._send_keyframe()
        #处理 APP 类型的 RTCP 包：REMB反馈包（包含估计的带宽信息）
        elif isinstance(packet, RtcpPsfbPacket) and packet.fmt == RTCP_PSFB_APP:
            try:
                bitrate, ssrcs = unpack_remb_fci(packet.fci)#REMB反馈的带宽估计
                if self._ssrc in ssrcs:
                    self.__log_debug(
                        "- receiver estimated maximum bitrate %d bps", bitrate
                    )
                    if self.__encoder and hasattr(self.__encoder, "target_bitrate"):
                        self.__encoder.target_bitrate = bitrate
            except ValueError:
                pass

    async def _next_encoded_frame(self, codec: RTCRtpCodecParameters):
            # 设置关键帧
            if not self.encode_role_forwart and self.encode_mode.current_state=="S1":
                self._send_keyframe()
            #获取下一个媒体帧或数据包
            audio_level = None
            #如果编码器未初始化，获取一个适用于给定编解码器参数的编码器
            if self.__encoder is None:
                self.__encoder = get_encoder(codec)
            
            ts = clock.current_ms()#获取当前时间戳
            if isinstance(self._data, Frame):#如果获取的数据是帧类型
                # encode frame 编码帧
                if isinstance(self._data, AudioFrame):
                    audio_level = rtp.compute_audio_level_dbov(self._data)

                force_keyframe = self.__force_keyframe
                self.__force_keyframe = False
                #调用编码器的encode方法执行编码，返回编码的payloads（packet列表）和时间戳
                payloads, timestamp ,frametype,framesize= await self.__loop.run_in_executor(
                    None, self.__encoder.encode, self._data, force_keyframe
                )
                
            else:#调用编码器的pack方法执行编码
                payloads, timestamp = self.__encoder.pack(self._data)
            te = clock.current_ms()#获取编码结束时间戳
           
            return RTCEncodedFrame(payloads, timestamp, audio_level), te-ts,frametype,framesize
        # else:
        #     return None,None,None
    """重传丢失的RTP包"""
    async def _retransmit(self, sequence_number: int) -> None:
        """
        Retransmit an RTP packet which was reported as lost.
        """
        packet = self.__rtp_history.get(sequence_number % RTP_HISTORY_SIZE)
        if packet and packet.sequence_number == sequence_number:
            if self.__rtx_payload_type is not None:
                packet = wrap_rtx(
                    packet,
                    payload_type=self.__rtx_payload_type,
                    sequence_number=self.__rtx_sequence_number,
                    ssrc=self._rtx_ssrc,
                )
                self.__log_debug("> %s", packet)
                # Version 2 
                packet.set_packet_type(RtpPacketMediaType.kRetransmission)
                packet.set_retransmitted_sequence_number(self.__rtx_sequence_number)
                self.__rtx_sequence_number = uint16_add(self.__rtx_sequence_number, 1)

            
            # packet_bytes = packet.serialize(self.__rtp_header_extensions_map)
            # Version2:
            # await self.transport._send_rtp(packet_bytes)
            # 组装RtpPacketToSend并入队
            self.pace_sender.enqueue_packets(packet)

    def _send_keyframe(self) -> None:
        """
        Request the next frame to be a keyframe.
        """
        self.__force_keyframe = True
    async def _next_encoded_frame_two(self, codec: RTCRtpCodecParameters)->None:
            # 设置关键帧
            if  self.encode_role_forwart and self.encode_mode.current_state=="S1":
                self._send_keyframe()
            audio_level = None
            #如果编码器未初始化，获取一个适用于给定编解码器参数的编码器
            if self.__encoder_two is None:
                self.__encoder_two = get_encoder(codec)
            
            ts = clock.current_ms()#获取当前时间戳
            
            if isinstance(self._data, Frame):#如果获取的数据是帧类型
                # encode frame 编码帧
                if isinstance(self._data, AudioFrame):
                    audio_level = rtp.compute_audio_level_dbov(self._data)

                force_keyframe = self.__force_keyframe
                self.__force_keyframe = False
                #调用编码器的encode方法执行编码，返回编码的payloads（packet列表）和时间戳
                payloads, timestamp ,frametype,framesize= await self.__loop.run_in_executor(
                    None, self.__encoder_two.encode, self._data, force_keyframe
                )
                
            else:#调用编码器的pack方法执行编码
                payloads, timestamp = self.__encoder_two.pack(self._data)
            te = clock.current_ms()#获取编码结束时间戳
            
            return RTCEncodedFrame(payloads, timestamp, audio_level), te-ts,frametype,framesize
        
    async def _run_rtp(self, codec: RTCRtpCodecParameters) -> None:
        self.__log_debug("- RTP started")
        self.__rtp_started.set()
        # 初始化序列号和起始时间戳
        sequence_number = random16()
        timestamp_origin = random32()#初始化一个随机的初始时间和随机的初始包序号
        self.__log_debug('[FRAME_INFO] Timestamp_origin: %d, Sequence_number: %d',timestamp_origin, sequence_number)
        frame_number=0
        try:
            while True:#主循环：不断获取下一个编码帧，遍历帧中的payload并创建RTP数据包发送
                if not self.__track:
                    await asyncio.sleep(0.02)
                    continue
                
                # 编码下一帧
                self._data = await self.__track.recv()
                enc_frame, enc_dur ,frame_type= await self._next_encoded_frame(codec) #返回了一帧图像编码后产生的数据：编码打包后的packet列表和时间戳，enc_dur为编码一张图像花费的时间
                # 对于正常P帧编码的stream，正常传输直到编码器停止返回None
                timestamp = uint32_add(timestamp_origin, enc_frame.timestamp)
                self.__log_debug('[FRAME_INFO] Stream id : 1, Number: %d, PTS: %d, enc_dur: %d Type: %d', frame_number,timestamp, enc_dur,frame_type.value)
                # 遍历每个packet并为其创建一个RTP数据包
                packets=[]
                for i, payload in enumerate(enc_frame.payloads):
                    # Version 1
                    # packet = RtpPacket(
                    #     payload_type=codec.payloadType,
                    #     sequence_number=sequence_number,
                    #     timestamp=timestamp,
                    # )
                    
                    # Version2
                    packet=RtpPacketToSend(
                        payload_type=codec.payloadType,
                        sequence_number=sequence_number,
                        timestamp=timestamp,
                    )
                    packet.set_packet_type(RtpPacketMediaType.kVideo)
                    packet.set_allow_retransmission(True)
                    packet.set_first_packet_of_frame(i==0)
                    if frame_type.value==0 or frame_type.value==4:
                        packet.set_is_key_frame(True)
                    else:
                        packet.set_is_key_frame(False)
                    packet.ssrc = self._ssrc
                    packet.payload = payload
                    packet.marker = (i == len(enc_frame.payloads) - 1) and 1 or 0 #用于指示当前数据包是否是一个帧（frame）的最后一个数据包
                    # set header extensions 添加头部扩展
                    packet.extensions.abs_send_time = ( #RTP包的发送时间：获取当前的NTP时间戳，将64位的 NTP timestamps转圜为24位
                        clock.current_ntp_time() >> 14
                    ) & 0x00FFFFFF 
                    packet.extensions.mid = self.__mid #用于唯一标识发送的媒体流
                    packet.extensions.marker_first= "1" #stream id
                    if enc_frame.audio_level is not None:
                        packet.extensions.audio_level = (False, -enc_frame.audio_level)
                    # 记录第一个数据包的发送时间
                    
                    # send packet 调用_send_rtp发送RTP数据包
                    self.__log_debug("> %s", packet)
                    self.__rtp_history[
                        packet.sequence_number % RTP_HISTORY_SIZE
                    ] = packet
                    # packet_bytes = packet.serialize(self.__rtp_header_extensions_map) #一个RTP packet
                    # Version2:
                    # await self.transport._send_rtp(packet_bytes)
                    # 组装RtpPacketToSend并入队
                    packets.append(packet)
                    # 更新统计信息
                    self.__ntp_timestamp = clock.current_ntp_time()
                    self.timestamp=clock.current_ms()
                    self.__rtp_timestamp = packet.timestamp
                    self.__octet_count += len(payload)
                    self.__packet_count += 1
                    sequence_number = uint16_add(sequence_number, 1)
                self.pace_sender.enqueue_packets(packets)
                # 计算发送速率
                if self.timestamp-self.last_timestamp>1000:
                    self.send_rate=((self.__octet_count-self.last_octet)*8)/((self.timestamp-self.last_timestamp)/1000)
                    self.__log_debug('[Send_INFO] timestamp: %d, send_rate: %f bps, packet_count: %d', self.timestamp,self.send_rate, self.__packet_count-self.last_count)
                    self.last_octet=self.__octet_count
                    self.last_timestamp=self.timestamp
                    self.last_count=self.__packet_count
                frame_number = uint16_add(frame_number, 1)
        except (asyncio.CancelledError, ConnectionError, MediaStreamError):
            pass
        except Exception:
            # we *need* to set __rtp_exited, otherwise RTCRtpSender.stop() will hang,
            # so issue a warning if we hit an unexpected exception
            self.__log_warning(traceback.format_exc())

        # stop track
        if self.__track:
            self.__track.stop()
            self.__track = None

        self.__log_debug("- RTP finished")
        self.__rtp_exited.set()
    async def _run_rtp_multi_stream(self, codec: RTCRtpCodecParameters) -> None:
        self.__log_debug("- RTP started")
        self.__rtp_started.set()
        # 初始化序列号和起始时间戳
        sequence_number = random16()
        timestamp_origin = random32()#初始化一个随机的初始时间和随机的初始包序号
        self.__log_debug('[FRAME_INFO] Timestamp_origin: %d, Sequence_number: %d',timestamp_origin, sequence_number)
        frame_number=0
        try:
            while True:#主循环：不断获取下一个编码帧，遍历帧中的payload并创建RTP数据包发送
                if not self.__track:
                    await asyncio.sleep(0.02)
                    continue
                
                # 编码下一帧
                self._data = await self.__track.recv()
                if self._data.index%10==0 and self._data.index>=10:
                        self.encode_mode.transition()#转换到S1:出现关键帧
                        self.encode_role_forwart=not self.encode_role_forwart
                if self._data.index%10==1 and self._data.index>=10:
                    self.encode_mode.transition()#转换到S2：一张关键帧编码完成开始传输
                if  self._data.index%10==5 and self._data.index>=10:
                    self.encode_mode.transition()#转换到S3：关键帧传输完成
               
                # 对于正常P帧编码的stream，正常传输直到编码器停止返回None
                # if not self.condition_Finish_event.is_set() and enc_frame:
                if  (self.encode_role_forwart and self.encode_mode.current_state!="S3") or (not self.encode_role_forwart and self.encode_mode.current_state!="S2"):
                    enc_frame, enc_dur ,frame_type,frame_size= await self._next_encoded_frame(codec) #返回了一帧图像编码后产生的数据：编码打包后的packet列表和时间戳，enc_dur为编码一张图像花费的时间
                    timestamp = uint32_add(timestamp_origin, enc_frame.timestamp)
                    self.__log_debug('[FRAME_INFO] Stream id : 1, Number: %d, PTS: %d, enc_dur: %d Type: %s, size: %d', frame_number,timestamp, enc_dur,frame_type.name,frame_size)
                    # 遍历每个packet并为其创建一个RTP数据包
                    packets=[]
                    for i, payload in enumerate(enc_frame.payloads):
                        # Version 2
                        # packet=RtpPacketToSend(
                        # payload_type=codec.payloadType,
                        # sequence_number=sequence_number,
                        # timestamp=timestamp,
                        # )
                        packet=RtpPacket(
                        payload_type=codec.payloadType,
                        sequence_number=sequence_number,
                        timestamp=timestamp,
                        )
                        # packet.set_packet_type(RtpPacketMediaType.kVideo)
                        # packet.set_allow_retransmission(True)
                        # packet.set_first_packet_of_frame(i==0)
                        # if frame_type.value==0 or frame_type.value==4:
                        #     packet.set_is_key_frame(True)
                        # else:
                        #     packet.set_is_key_frame(False)
                        packet.ssrc = self._ssrc
                        packet.payload = payload
                        packet.payload_size=len(payload)
                        packet.marker = (i == len(enc_frame.payloads) - 1) and 1 or 0 #用于指示当前数据包是否是一个帧（frame）的最后一个数据包
                        # set header extensions 添加头部扩展
                        packet.extensions.abs_send_time = ( #RTP包的发送时间：获取当前的NTP时间戳，将64位的 NTP timestamps转圜为24位
                            clock.current_ntp_time() >> 14
                        ) & 0x00FFFFFF 
                        packet.extensions.mid = self.__mid #用于唯一标识发送的媒体流
                        packet.extensions.marker_first= "1" #stream id
                        if enc_frame.audio_level is not None:
                            packet.extensions.audio_level = (False, -enc_frame.audio_level)
                        # 记录第一个数据包的发送时间
                        
                        # send packet 调用_send_rtp发送RTP数据包
                        self.__log_debug("> %s", packet)
                        self.__rtp_history[
                            packet.sequence_number % RTP_HISTORY_SIZE
                        ] = packet
                        packet_bytes = packet.serialize(self.__rtp_header_extensions_map) #一个RTP packet
                        await self.transport._send_rtp(packet_bytes)
                        # Version2:
                        # 组装RtpPacketToSend并入队
                        packets.append(packet)
                        # 更新统计信息
                        self.__ntp_timestamp = clock.current_ntp_time()
                        self.timestamp=clock.current_ms()
                        self.__rtp_timestamp = packet.timestamp
                        self.__octet_count += len(payload)
                        self.__packet_count += 1
                        sequence_number = uint16_add(sequence_number, 1)
                    # self.pace_sender.enqueue_packets(packets)
                # 对于强制I帧编码的stream，传输I帧，跳过I帧后的5个P帧后再开启传输
                # if (enc_frame_two and (frame_type_two.value==0 or frame_type_two.value == 4)) or (enc_frame_two and self.condition_Finish_event.is_set()):
                if  (self.encode_role_forwart and self.encode_mode.current_state!="S2") or (not self.encode_role_forwart and self.encode_mode.current_state!="S3" and self.encode_mode.current_state!="S0"):
                    enc_frame_two, enc_dur_two ,frame_type_two,frame_size_two= await self._next_encoded_frame_two(codec) #返回了一帧图像编码后产生的数据：编码打包后的packet列表和时间戳，enc_dur为编码一张图像花费的时间
                    timestamp = uint32_add(timestamp_origin, enc_frame_two.timestamp)
                    self.__log_debug('[FRAME_INFO] Stream id : 2, Number: %d, PTS: %d, enc_dur: %d Type: %s, size: %d', frame_number,timestamp, enc_dur_two,frame_type_two.name,frame_size_two)
                    packets=[]
                    for i, payload in enumerate(enc_frame_two.payloads):
                        #Version 2
                        # packet=RtpPacketToSend(
                        # payload_type=codec.payloadType,
                        # sequence_number=sequence_number,
                        # timestamp=timestamp,
                        # )
                        packet=RtpPacket(
                        payload_type=codec.payloadType,
                        sequence_number=sequence_number,
                        timestamp=timestamp,
                        )
                        # packet.set_packet_type(RtpPacketMediaType.kVideo)
                        # packet.set_allow_retransmission(True)
                        # packet.set_first_packet_of_frame(i==0)
                        # if frame_type.value==0 or frame_type.value==4:
                            # packet.set_is_key_frame(True)
                        # else:
                            # packet.set_is_key_frame(False)
                        packet.ssrc = self._ssrc
                        packet.payload = payload
                        packet.payload_size=len(payload)
                        packet.marker = (i == len(enc_frame_two.payloads) - 1) and 1 or 0 #用于指示当前数据包是否是一个帧（frame）的最后一个数据包
                        # set header extensions 添加头部扩展
                        packet.extensions.abs_send_time = ( #RTP包的发送时间：获取当前的NTP时间戳，将64位的 NTP timestamps转圜为24位
                            clock.current_ntp_time() >> 14
                        ) & 0x00FFFFFF 
                        packet.extensions.mid = self.__mid #用于唯一标识发送的媒体流
                        packet.extensions.marker_first= "2" #stream id
                        if enc_frame_two.audio_level is not None:
                            packet.extensions.audio_level = (False, -enc_frame_two.audio_level)
                        # 记录第一个数据包的发送时间
                        
                        # send packet 调用_send_rtp发送RTP数据包
                        self.__log_debug("> %s", packet)
                        self.__rtp_history[
                            packet.sequence_number % RTP_HISTORY_SIZE
                        ] = packet
                        packet_bytes = packet.serialize(self.__rtp_header_extensions_map) #一个RTP packet
                        await self.transport._send_rtp(packet_bytes)
                        # Version2:
                        # 组装RtpPacketToSend并入队
                        # 更新统计信息
                        packets.append(packet)
                        self.__ntp_timestamp = clock.current_ntp_time()
                        self.timestamp=clock.current_ms()
                        self.__rtp_timestamp = packet.timestamp
                        self.__octet_count += len(payload)
                        self.__packet_count += 1
                        sequence_number = uint16_add(sequence_number, 1)
                    # self.pace_sender.enqueue_packets(packets)
                # 计算发送速率
                if self.timestamp-self.last_timestamp>1000:
                    self.send_rate=((self.__octet_count-self.last_octet)*8)/((self.timestamp-self.last_timestamp)/1000)
                    self.__log_debug('[Send_INFO] timestamp: %d, send_rate: %f bps, packet_count: %d', self.timestamp,self.send_rate, self.__packet_count-self.last_count)
                    self.last_octet=self.__octet_count
                    self.last_timestamp=self.timestamp
                    self.last_count=self.__packet_count
                frame_number = uint16_add(frame_number, 1)
        except (asyncio.CancelledError, ConnectionError, MediaStreamError):
            pass
        except Exception:
            # we *need* to set __rtp_exited, otherwise RTCRtpSender.stop() will hang,
            # so issue a warning if we hit an unexpected exception
            self.__log_warning(traceback.format_exc())

        # stop track
        if self.__track:
            self.__track.stop()
            self.__track = None

        self.__log_debug("- RTP finished")
        self.__rtp_exited.set()

    async def _run_rtcp(self) -> None:
        self.__log_debug("- RTCP started")
        self.__rtcp_started.set()

        try:
            while True:
                # The interval between RTCP packets is varied randomly over the
                # range [0.5, 1.5] times the calculated interval.
                await asyncio.sleep(0.5 + random.random())

                # RTCP SR
                packets: List[AnyRtcpPacket] = [
                    RtcpSrPacket(
                        ssrc=self._ssrc,
                        sender_info=RtcpSenderInfo(
                            ntp_timestamp=self.__ntp_timestamp,
                            rtp_timestamp=self.__rtp_timestamp,
                            packet_count=self.__packet_count,
                            octet_count=self.__octet_count,
                        ),
                    )
                ]
                self.__lsr = ((self.__ntp_timestamp) >> 16) & 0xFFFFFFFF
                self.__lsr_time = time.time()

                # RTCP SDES
                if self.__cname is not None:
                    packets.append(
                        RtcpSdesPacket(
                            chunks=[
                                RtcpSourceInfo(
                                    ssrc=self._ssrc,
                                    items=[(1, self.__cname.encode("utf8"))],
                                )
                            ]
                        )
                    )

                await self._send_rtcp(packets)
        except asyncio.CancelledError:
            pass

        # RTCP BYE
        packet = RtcpByePacket(sources=[self._ssrc])
        await self._send_rtcp([packet])

        self.__log_debug("- RTCP finished")
        self.__rtcp_exited.set()

    async def _send_rtcp(self, packets: List[AnyRtcpPacket]) -> None:
        payload = b""
        for packet in packets:
            self.__log_debug("> %s", packet)
            payload += bytes(packet)

        try:
            await self.transport._send_rtp(payload)
        except ConnectionError:
            pass

    def __log_warning(self, msg: str, *args) -> None:
        logger.warning(f"RTCRtpsender(%s) {msg}", self.__kind, *args)
    