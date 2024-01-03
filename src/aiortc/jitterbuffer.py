from typing import List, Optional, Tuple
import numpy as np

from .rtp import RtpPacket
from .utils import uint16_add
from .clock import current_ms
import time
MAX_MISORDER = 100


class JitterFrame:
    def __init__(self, data: bytes, timestamp: int,stream_id:str) -> None:
        self.data = data
        self.timestamp = timestamp
        self.times_dur = {}
        self.stream_id=stream_id


class JitterBuffer:
    def __init__(
        self, capacity: int, prefetch: int = 0, is_video: bool = False
    ) -> None:
        assert capacity & (capacity - 1) == 0, "capacity must be a power of 2"
        self._capacity = capacity #容量
        self._origin: Optional[int] = None #起始位置
        self._packets: List[Optional[RtpPacket]] = [None for i in range(capacity)]
        self._prefetch = prefetch
        self._is_video = is_video
        self._packet_times_in = {}

    @property
    def capacity(self) -> int:
        return self._capacity

    def add(self, packet: RtpPacket) -> Tuple[bool, Optional[JitterFrame],int]:
        pli_flag = False
        if self._origin is None:# # 如果缓冲区的起始序列号为空，表示这是第一个数据包，设置起始序列号和相关变量
            self._origin = packet.sequence_number
            delta = 0
            misorder = 0
        else: # 计算当前数据包的序列号和起始序列号的差值
            delta = uint16_add(packet.sequence_number, -self._origin)
            misorder = uint16_add(self._origin, -packet.sequence_number)
        # 如果乱序的数量小于差值，表示数据包是按序到达的，直接返回
        if misorder < delta:
            if misorder >= MAX_MISORDER: #如果包的序列号超出一定范围（MAX_MISORDER），则移除部分数据包
                self.remove(self.capacity)# 如果乱序数量超过阈值，移除部分数据包，重置起始序列号
                self._origin = packet.sequence_number
                delta = misorder = 0
                if self._is_video:
                    pli_flag = True
            else:
                return pli_flag, None,None
        # 如果差值超过缓冲区的容量，移除多余的帧，确保缓冲区中只存储最新的数据
        if delta >= self.capacity: #如果包的序列号超过缓冲区的容量（self.capacity），则移除多余的帧，确保缓冲区中只存储最新的数据
            # remove just enough frames to fit the received packets
            excess = delta - self.capacity + 1
            if self.smart_remove(excess):
                self._origin = packet.sequence_number
            if self._is_video:
                pli_flag = True
        # 计算数据包在缓冲区中的位置
        pos = packet.sequence_number % self._capacity
        self._packets[pos] = packet
        if packet.timestamp not in self._packet_times_in: self._packet_times_in[packet.timestamp] = []  #_packet_times_in记录该帧所有数据包的接收时间
        self._packet_times_in[packet.timestamp].append(current_ms())
        self.t1=current_ms()
        # if packet.timestamp not in self._packet_times_in: self._packet_times_in[packet.timestamp] = 0  #_packet_times_in记录该帧所有数据包的接收时间
        # self._packet_times_in[packet.timestamp]=current_ms()
        encodeframe,jit_dur=self._remove_frame(packet.sequence_number)
        return pli_flag, encodeframe,jit_dur
    """从缓冲区中移除一个完整的 RTP 帧"""
    def _remove_frame(self, sequence_number: int) -> Tuple[Optional[JitterFrame],int]:
        frame = None
        frames = 0
        packets = []
        remove = 0
        timestamp = None
        import logging
        logger = logging.getLogger(__name__)

        for count in range(self.capacity): #遍历缓冲区中的每个位置，获取对应位置的 RTP 包
            pos = (self._origin + count) % self._capacity
            packet = self._packets[pos]
            if packet is None:
                
                break
            if timestamp is None:
                timestamp = packet.timestamp
            elif packet.timestamp != timestamp:#如果发现不同时间戳的包，表示已经获取到一个完整的帧，将这一帧的数据包合并为 JitterFrame 对象
                # we now have a complete frame, only store the first one
                if frame is None:
                    frame = JitterFrame(
                        data=b"".join([x._data for x in packets]), timestamp=timestamp,stream_id=packets[0].extensions.marker_first
                    )
                    remove = count
                    # avg_time_in = np.mean(self._packet_times_in[timestamp]) #计算当前 RTP 帧的平均到达时间
                    # jit_dur = current_ms() - avg_time_in #帧的到达时间与其平均到达时间的差异

                # check we have prefetched enough
                frames += 1
                if frames >= self._prefetch: #检查是否已经预取足够数量的帧（self._prefetch），如果是，则移除之前的数据包，返回合并的帧和相关的抖动延迟
                    self.remove(remove)
                    # t2=time.perf_counter()
                    # print("jitter search time: {0}".format(t2-t1))
                    ts=current_ms()
                    # logger.info("jitter frame output: {0} s".format(timestamp))
                    # logger.info("jitter frame output: {0} s".format(ts-self._packet_times_in[timestamp][-1]))
                    # logger.info("jitter frame output: {0} s".format(ts-self.t1))
                    # logger.info("jitter frame output: {0} s".format(max(list(self._packet_times_in.keys()))))
                    
                    jit_dur=ts-self._packet_times_in[timestamp][-1]
                    return frame, jit_dur
                
                # start a new frame
                packets = []
                timestamp = packet.timestamp

            packets.append(packet)

        return None, None

    def remove(self, count: int) -> None:
        assert count <= self._capacity
        for i in range(count):
            pos = self._origin % self._capacity
            self._packets[pos] = None
            self._origin = uint16_add(self._origin, 1)

    def smart_remove(self, count: int) -> bool:
        """
        Makes sure that all packages belonging to the same frame are removed
        to prevent sending corrupted frames to the decoder.
        """
        timestamp = None
        for i in range(self._capacity):
            pos = self._origin % self._capacity
            packet = self._packets[pos]
            if packet is not None:
                if i >= count and timestamp != packet.timestamp:
                    break
                timestamp = packet.timestamp
            self._packets[pos] = None
            self._origin = uint16_add(self._origin, 1)
            if i == self._capacity - 1:
                return True
        return False
