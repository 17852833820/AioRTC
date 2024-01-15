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
from collections import deque
import heapq
from . import clock
from .codecs import depayload, get_capabilities, get_decoder, is_rtx
from .exceptions import InvalidStateError
from .jitterbuffer import JitterBuffer,JitterFrame
from .mediastreams import MediaStreamError, MediaStreamTrack
from .rate import RemoteBitrateEstimator
from .rtcdtlstransport import RTCDtlsTransport
from .rtcrtpparameters import (RTCRtpCapabilities, RTCRtpCodecParameters,
                               RTCRtpReceiveParameters)
from .rtp import (RTCP_PSFB_APP, RTCP_PSFB_PLI, RTCP_RTPFB_NACK,RTCP_PSFB_IF,
                  RTP_HISTORY_SIZE, AnyRtcpPacket, RtcpByePacket,
                  RtcpPsfbPacket, RtcpReceiverInfo, RtcpRrPacket,
                  RtcpRtpfbPacket, RtcpSrPacket, RtpPacket, RtpPacketToSend,clamp_packets_lost,
                  pack_remb_fci, unwrap_rtx)
from .stats import (RTCInboundRtpStreamStats, RTCRemoteOutboundRtpStreamStats,
                    RTCStatsReport)
from .utils import uint16_add, uint16_gt

logger = logging.getLogger(__name__)
# abs-send-time estimator
INTER_ARRIVAL_SHIFT = 26
TIMESTAMP_GROUP_LENGTH_MS = 5
TIMESTAMP_TO_MS = 1000.0 / (1 << INTER_ARRIVAL_SHIFT)
kMaxAllowedFrameDelayMs = 5
class VCMTiming:
    def __init__(self) -> None:
        self._crit_sect=None
        self._render_delay_ms:int =10 # 渲染耗时,默认10ms
        self._min_playout_delay_ms:int =0 # 最小播放延迟
        self._max_playout_delay_ms:int =10000# 最大播放延迟
        self._jitter_delay_ms:int =0 # 网络抖动
        self._current_delay_ms:int =0 # 当前延迟，用于计算视频帧渲染时间,表示需要等待的时间
        self._prev_frame_timestamp:int =0 
        self._num_decoded_frames:int =0
        self._master:bool=True
        self._decoder_time_ms:int=0
        self.codec_timer:VCMCodecTimer=VCMCodecTimer()
        # if self._master:
        #     self._ts_extrapolator=TimestampExtrapolator(current_ms())
    def reset(self):
        self._render_delay_ms = 10
        self._min_playout_delay_ms = 0
        self._jitter_delay_ms = 0
        self._current_delay_ms = 0
        self._prev_frame_timestamp = 0
        self.codec_timer:VCMCodecTimer=VCMCodecTimer()
        # self._ts_extrapolator.reset(current_ms())
    def set_render_delay(self,render_delay_ms:int):
        self._render_delay_ms=render_delay_ms
    def set_jitter_delay(self,jitter_delay_ms:int):
        if jitter_delay_ms!=self._jitter_delay_ms:
            self._jitter_delay_ms=jitter_delay_ms
            if self._current_delay_ms==0:
                self._current_delay_ms=self._jitter_delay_ms
                logger.info("current delay before:0,after:{0}".format(self._current_delay_ms))
    def set_min_playout_delay(self,min_playout_delay:int)->None:
        self._min_playout_delay_ms=min_playout_delay
    def min_playout_delay(self)->int:
        return self._min_playout_delay_ms
    def set_max_playout_delay(self,max_playout_delay:int)->None:
        self._max_playout_delay_ms=max_playout_delay
    def max_playout_delay(self)->int:
        return self._max_playout_delay_ms
  
    def required_decode_time_ms(self)->int:
        decode_time_ms=self.codec_timer.required_decode_time_ms()
        assert decode_time_ms>=0
        self._decoder_time_ms=decode_time_ms
        return self._decoder_time_ms
    def stop_decode_timer(self,decode_time_ms:int,now_ms:int):
        self.codec_timer.add_timing(decode_time_ms,now_ms)
        assert decode_time_ms>=0
    # def set_decode_time_ms(self,decode_time_ms):
    #     self._decoder_time_ms=decode_time_ms
    def max_waiting_time(self,render_time_ms:int,now_ms:int)->int:
        max_wait_time_ms=render_time_ms-now_ms-self.required_decode_time_ms()-self._render_delay_ms
        return max_wait_time_ms
    # 每次获得一个可解码帧调用一次，更新当前延迟，最终用于计算渲染时间
    def update_current_delay(self,render_time_ms:int,actual_decode_time_ms:int):#render_time_ms帧期望渲染时间,实际解码时间actual_decode_time_ms
        # 1. 获得目标延迟
        target_delay_ms=self.target_delay_internal()
        # render_time_ms:期望渲染时间
        # 期望解码时刻=帧期望渲染时间-解码耗时-渲染耗时
        # 实际产生的延迟delay_ms=实际解码时刻actual_decode_time_ms-期望解码时刻
        delay_ms=actual_decode_time_ms-(render_time_ms-self.required_decode_time_ms()-self._render_delay_ms)
        logger.debug("current delay:{0},target_delay_ms:{1},jitter delay:{2},delay ms:{3}".format(self._current_delay_ms,target_delay_ms,self._jitter_delay_ms,delay_ms))
        if delay_ms<0:
            return 
        current_delay_before=self._current_delay_ms
        # 2. 如果有延迟，上个时刻的当前延迟+实际产生的延迟 仍然 <=目标延迟
        if self._current_delay_ms + delay_ms <= target_delay_ms:
            # 更新当前延迟，逼近目标延迟，会增加当前延迟
            self._current_delay_ms+=delay_ms
            logger.info("Update 1:current delay before:{0},after:{1}".format(current_delay_before,self._current_delay_ms))
        else:# 如果上个时刻的当前延迟 + 实际产生的延迟仍然超过目标延迟，以目标延迟为上限.可以降低当前延迟
            self._current_delay_ms=target_delay_ms
            logger.info("Update 2:current delay before:{0},after:{1}".format(current_delay_before,self._current_delay_ms))
    # def update_current_delay(self,frame_timestamp:int)->None:
    # frame_timestamp为帧pts预期渲染时间
    def render_time_ms(self,frame_timestamp:int,now_ms:int)->int:
        return self.render_time_ms_internal(frame_timestamp,now_ms)
    # 计算视频帧最终渲染时间 = 帧平滑时间 + 当前延迟
    def render_time_ms_internal(self,frame_timestamp:int,now_ms:int)->int:# frame_timestamp帧时间戳
        # 1. 如果这两个播放延迟都是0，要求立即渲染 
        if self._min_playout_delay_ms ==0 and self._max_playout_delay_ms == 0:
            return 0
        # 2. 使用卡尔曼滤波器估计帧平滑时间
        # estimated_complele_time_ms=self._ts_extrapolator.extrapolate_local_time(frame_timestamp)
        estimated_complele_time_ms=-1
        if estimated_complele_time_ms ==-1:
            estimated_complele_time_ms=now_ms
        # 3. 当前延迟限定在(min_playout_delay_ms_, max_playout_delay_ms_)范围内
        actual_delay=max(self._current_delay_ms,self._min_playout_delay_ms)
        logger.debug("actual_delay:{0}".format(actual_delay))
        actual_delay=min(actual_delay,self._max_playout_delay_ms)
        # 4. 视频帧的最终渲染时间 = 帧平滑时间 + 当前延迟
        return estimated_complele_time_ms+actual_delay
    def target_video_delay(self)->int:
        return self.target_delay_internal()
    # 目标延迟=max（抖动延迟+解码时间+渲染时间，播放延迟）
    def target_delay_internal(self)->int:
        # 目标延迟
        logger.info("_min_playout_delay_ms:{0}, _jitter_delay_ms:{1}, _decoder_time_ms:{2}, _render_delay_ms:{3}".format(self._min_playout_delay_ms,self._jitter_delay_ms,self.required_decode_time_ms(),self._render_delay_ms))
        return max(self._min_playout_delay_ms,
                  self._jitter_delay_ms + self.required_decode_time_ms() + self._render_delay_ms)
class VCMCodecTimer:
    def __init__(self):
        self.kIgnoredSampleCount = 5
        self.kPercentile = 0.95
        self.kTimeLimitMs = 10000 # 10s
        self.ignored_sample_count_ = 0
        self.filter_ = PercentileFilter(self.kPercentile)
        self.history_ = deque()

    def add_timing(self, decode_time_ms, now_ms):
        if self.ignored_sample_count_ < self.kIgnoredSampleCount:
            self.ignored_sample_count_ += 1
            return

        self.filter_.insert(decode_time_ms)
        self.history_.append(Sample(decode_time_ms, now_ms))

        while self.history_ and now_ms - self.history_[0].sample_time_ms > self.kTimeLimitMs:
            self.filter_.erase(self.history_[0].decode_time_ms)
            self.history_.popleft()

    def required_decode_time_ms(self):
        return self.filter_.get_percentile_value()
class Sample:
    def __init__(self, decode_time_ms, sample_time_ms):
        self.decode_time_ms = decode_time_ms
        self.sample_time_ms = sample_time_ms


class PercentileFilter:
    def __init__(self, percentile):
        self.percentile = percentile
        self.data = []

    def insert(self, value):
        heapq.heappush(self.data, value)

    def erase(self, value):
        self.data.remove(value)
        heapq.heapify(self.data)

    def get_percentile_value(self):
        index = int(len(self.data) * self.percentile)
        if index==0:return 0
        return sorted(self.data)[index]


class FrameDecoder:
    def __init__(self,thread_name_) -> None:
        self.latest_return_time_ms_:int=0 # 最晚返回绝对时间戳
        self.max_wait_time_ms:int =0 # 最大等待时间
        self.last_decoded_frame_timestamp:int=0
        self.lock_ = threading.Lock()
        self.thread_=None
        self.stop_ = False
        self.thread_name_:str=thread_name_
        self.timing_:VCMTiming = VCMTiming()
        self.jitter_buffer = JitterBuffer(capacity=128, is_video=True)
    def set_frame_rate(self,fps:int)->None:
        self.jitter_buffer._jitter_estimator.set_frame_rate(fps)
    def start(self,decoer_queue,output_queue):
        self.thread_ = threading.Thread(target=self.run_sync, name=self.thread_name_,args=(
                    asyncio.get_event_loop(),
                    decoer_queue,#待解码的队列
                    output_queue,#解码后输出的队列
                ))
        self.thread_.start()
    def run_sync(self,loop, input_q, output_q):
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(self.decoder_worker(loop, input_q, output_q))
    def join(self):
         with self.lock_:
            self.stop_ = True
         self.thread_.join()
         self.stop=False
         self.thread_=None
    def HasBadRenderTiming(self,now_ms:int,frame:JitterFrame)->bool:
        if frame.render_time_ms == 0:
            return False
        if frame.render_time_ms < 0:
            return True
        kMaxVideoDelayMs = 10000
        if abs(frame.render_time_ms -now_ms ) >kMaxVideoDelayMs:
            frame_delay=abs(frame.render_time_ms -now_ms )
            logger.warning("A frame about to be decoded is out of the configured: delay bounds ({0}>{1}). Resetting the video jitter buffer.".format(frame_delay,kMaxVideoDelayMs))
            return True
        # with self.lock_:
        if self.timing_.target_video_delay()> kMaxVideoDelayMs:
            logger.warning("The video target delay has grown larger than :{0} ms.".format(kMaxVideoDelayMs))
            return True
        return False
    def find_next_frame(self,now_ms:int,frame:JitterFrame)->int:
        with self.lock_:
            wait_ms = self.latest_return_time_ms_ - now_ms
            if self.last_decoded_frame_timestamp !=0 and self.last_decoded_frame_timestamp>frame.timestamp:
                return -1
            if frame.render_time_ms == -1:
                frame.render_time_ms=self.timing_.render_time_ms(frame.timestamp,now_ms)
            wait_ms=self.timing_.max_waiting_time(frame.render_time_ms,now_ms)
        if wait_ms < - kMaxAllowedFrameDelayMs:
            return -1
        wait_ms=min(wait_ms,self.latest_return_time_ms_-now_ms)
        wait_ms=max(wait_ms,0)
        return wait_ms
    def get_next_frame(self,encoded_frame:JitterFrame)->JitterFrame:
        now_ms=clock.current_ms()
        render_time_ms=encoded_frame.render_time_ms
        with self.lock_:
            if self.HasBadRenderTiming(now_ms,encoded_frame):
                self.jitter_buffer._jitter_estimator.reset()
                self.timing_.reset()
                render_time_ms=self.timing_.render_time_ms(encoded_frame.timestamp,now_ms)
            encoded_frame.render_time_ms=render_time_ms
            self.jitter_buffer._jitter_estimator.update_estimate(encoded_frame.frame_delay_ms,len(encoded_frame.data),False)
            jitterDelay=self.jitter_buffer._jitter_estimator.calculate_estimate() # 网络抖动
            encoded_frame.jitter_delay_ms=jitterDelay
            self.timing_.set_jitter_delay(jitterDelay)
            self.timing_.update_current_delay(render_time_ms,now_ms)
        return encoded_frame
    # 解码线程主任务
    async def decoder_worker(self, loop, input_q, output_q):
        #初始化：用于追踪当前的解码器和编码器名称
        codec_name = None
        decoder = None
        __log_debug: Callable[..., None] = lambda *args: None
        if logger.isEnabledFor(logging.DEBUG):
            __log_debug = lambda msg, *args: logger.debug(
                f"RTCRtpReceiver(%s) {msg}", 'decoder_worker', *args
            )

        while True:#任务处理无限循环
            # logger.info("input queue size:{0}".format(input_q.qsize()))
            task = input_q.get() #等待从输入队列 (input_q) 获取任务
            logger.info("Decode: get task from input frame queue!{0}".format(task))
            if task is None:# 如果获取到的任务为 None，表示线程结束，将 None 放入输出队列 (output_q) 并终止循环。
                # inform the track that is has ended
                asyncio.run_coroutine_threadsafe(output_q.put(None), loop)
                break
            codec, encoded_frame = task
            if codec.name != codec_name: #检查编解码器是否匹配
                decoder = get_decoder(codec)# 实际解码器
                codec_name = codec.name
            # 计算延迟等待时间
            if encoded_frame.is_key_frame:
                self.max_wait_time_ms=200
            else:
                self.max_wait_time_ms=3000
            self.latest_return_time_ms_=clock.current_ms()+self.max_wait_time_ms
            wait_time=self.find_next_frame(clock.current_ms(),encoded_frame)
            # __log_debug('[DECODE] wait time %d', wait_time)
            if wait_time == -1:
                wait_time=0   
            # 启动延迟执行的异步任务
            encoded_frame=self.get_next_frame(encoded_frame)
            t1=clock.current_ms()
            # future=asyncio.run_coroutine_threadsafe(self.delayed_decode(decoder, encoded_frame, wait_time, output_q,loop), loop)
            future= await self.delayed_decode(decoder, encoded_frame, wait_time, output_q,loop)
            t2=clock.current_ms()
            logger.info("duration1:{0},t2:{1},t1:{2}".format(t2-t1,t2,t1))
            # dec_dur=future.result()
            dec_dur=future
            t3=clock.current_ms()
            logger.info("duration2:{0},t3:{1},t2:{2}".format(t3-t2,t3,t2))
            self.timing_.stop_decode_timer(dec_dur,clock.current_ms())
            end_time=clock.current_ms()
        if decoder is not None:
            del decoder
    async def delayed_decode(self,decoder,encoded_frame,wait_time,output_q,loop):
        __log_debug: Callable[..., None] = lambda *args: None
        if logger.isEnabledFor(logging.DEBUG):
            __log_debug = lambda msg, *args: logger.debug(
                f"RTCRtpReceiver(%s) {msg}", 'decoder_worker', *args
            )
        ts= clock.current_ms()
        await asyncio.sleep(wait_time/1000.0)
        # await asyncio.sleep(wait_time % 1000)
        __log_debug("actual wait time:{0},expect wait time:{1}".format(clock.current_ms()-ts,wait_time))
        t1= clock.current_ms()
        for frame in decoder.decode(encoded_frame):
            asyncio.run_coroutine_threadsafe(output_q.put(frame), loop) #将解码后的帧（frame）放入输出队列
            __log_debug('[DECODE] Add Render Frame...Stream id: %s, Number: %d, Type: %d', encoded_frame.stream_id, frame.index, frame.pict_type)
        t2 = clock.current_ms()
        dec_dur =t2-t1
        __log_debug('[DECODE] stream_id: %s, is_key_frame: %d, T: %d, dec_dur: %d, wait_time: %d, jitter_delay_ms: %d, frame_delay_ms: %d, render_time_ms: %d, frame_delay_ms: %d, receive_time_ms: %d, Bytes: %d', encoded_frame.stream_id,encoded_frame.is_key_frame,encoded_frame.timestamp, dec_dur, wait_time,encoded_frame.jitter_delay_ms,encoded_frame.frame_delay_ms,encoded_frame.render_time_ms,encoded_frame.frame_delay_ms,encoded_frame.receive_time_ms,len(encoded_frame.data))
        return dec_dur

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
        self.fps:int=0
        self.counter:int=0
        self.last_arrival_time:int=clock.current_ms()
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
        # 记录fps
        self.counter+=1
        if clock.current_ms()-self.last_arrival_time>1000: 
            self.fps=self.counter
            self.counter=0
            self.last_arrival_time=clock.current_ms()
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
        # self.__decoder_thread: Optional[threading.Thread] = None
        # self.__decoder_thread2: Optional[threading.Thread] = None
        self.__decoder_thread:FrameDecoder=FrameDecoder("-decoder1")
        self.__decoder_thread2:FrameDecoder=FrameDecoder("-decoder2")
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
            # self.__decoder_thread = threading.Thread(
            #     target=FrameDecoder.decoder_worker,
            #     name=self.__kind + "-decoder1",
            #     args=(
            #         asyncio.get_event_loop(),
            #         self.__decoder_queue,#待解码的队列
            #         self._track._queue,#解码后输出的队列
            #     ),
            # )
            # self.__decoder_thread.start()
            self.__decoder_thread.start(self.__decoder_queue,self._track._queue)
            #多流解码
            # if self.use_multistream:
            #     self.__decoder_thread2 = threading.Thread(
            #         target=FrameDecoder.decoder_worker,
            #         name=self.__kind + "-decoder2",
            #         args=(
            #             asyncio.get_event_loop(),
            #             self.__decoder_queue2,#待解码的队列
            #             self._track._queue,#解码后输出的队列
            #         ),
            #     )
            #     self.__decoder_thread2.start()
            if self.use_multistream:
                self.__decoder_thread2.start(self.__decoder_queue2,self._track._queue)
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

    async def _handle_rtp_packet(self, packet: RtpPacketToSend, arrival_time_ms: int,arrival_24NTP_time:int) -> None:
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
        if  packet.marker: #最后一个包
            self.last_recv_time=arrival_time_ms
            self.__log_debug('[FRAME_INFO] T: %d ,  frame packet dur: %d ms',packet.timestamp,self.last_recv_time-self.first_recv_time)
        if packet._is_first_packet_of_frame:# 第一个包
            self.first_recv_time=arrival_time_ms
        packet.set_recv_time_ms(arrival_time_ms)
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
        pli_flag, encoded_frame, jit_dur ,is_key_frame= self.__jitter_buffer.add(packet)#向抖动缓冲区（Jitter Buffer）添加 RTP 包
        # 多流编码时检测是否需要发送I帧接收完成信号
        if self.use_multistream and is_key_frame:
            await self._send_rtcp_ifkey(packet.ssrc)
        # if jit_dur is not None: # 非jit 只是组帧jitter buffer的时间
            # self.__log_debug('[FRAME_INFO] T: %d, jit_dur: %d, Bytes: %d', encoded_frame.timestamp, jit_dur, len(encoded_frame.data))
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
                self.__decoder_thread2.set_frame_rate(self._track.fps)
                logger.debug("Receive Frame timestamp:{0},Push frame queue2".format(encoded_frame.timestamp))
                logger.info("input queue2 size:{0}".format(self.__decoder_queue2.qsize()))
            else:
                self.__decoder_queue.put((codec, encoded_frame))
                self.__decoder_thread.set_frame_rate(self._track.fps)
                logger.debug("Receive Frame timestamp:{0},Push frame queue".format(encoded_frame.timestamp))
                logger.info("input queue size:{0}".format(self.__decoder_queue.qsize()))


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
    async def _send_rtcp_ifkey(self, media_ssrc: int) -> None:
            """
            Send an RTCP packet to report picture loss.
            """
            if self.__rtcp_ssrc is not None:
                packet = RtcpPsfbPacket(
                    fmt=RTCP_PSFB_IF, ssrc=self.__rtcp_ssrc, media_ssrc=media_ssrc
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
