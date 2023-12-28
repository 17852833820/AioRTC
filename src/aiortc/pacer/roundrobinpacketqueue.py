"""
RoundRobinPacketQueue:pacer优先级队列相关代码
"""
import heapq
from enum import Enum,auto
from typing import List, Dict
from typing import Optional, Iterator
from queue import PriorityQueue,Empty
from typing import Iterator
from datetime import datetime, timedelta
from ..rtp import RtpPacket
from typing import Optional,List
kMaxLeadingSize=1400 #Byte
class RtpPacketMediaType(Enum):
    kAudio = auto()                    # Audio media packets.
    kVideo = auto()                    # Video media packets.
    kRetransmission = auto()           # Retransmissions, sent as response to NACK.
    kForwardErrorCorrection = auto()   # FEC packets.
    kPadding = auto()                  # RTX or plain padding sent to maintain BWE.

class RtpPacketToSend(RtpPacket):
    def __init__(self, payload_type: int = 0, marker: int = 0, sequence_number: int = 0, timestamp: int = 0, ssrc: int = 0, payload: bytes = b"") -> None:
        super().__init__(payload_type, marker, sequence_number, timestamp, ssrc, payload)
        self.capture_time_ms:int =0
        self._packet_type:Optional[RtpPacketMediaType]=None
        self._retransmitted_sequence_number: Optional[int] = None
        self._allow_retransmission: bool = False
        self._application_data: List[int] = []
        self._is_first_packet_of_frame: bool = False
        self._is_key_frame: bool = False
    def set_capture_time_ms(self, time: int):
        self._capture_time_ms = time

    def set_packet_type(self, packet_type: RtpPacketMediaType):
        self._packet_type = packet_type

    def packet_type(self) -> Optional[RtpPacketMediaType]:
        return self._packet_type

    def set_retransmitted_sequence_number(self, sequence_number: int):
        self._retransmitted_sequence_number = sequence_number

    def retransmitted_sequence_number(self) -> Optional[int]:
        return self._retransmitted_sequence_number

    def set_allow_retransmission(self, allow_retransmission: bool):
        self._allow_retransmission = allow_retransmission

    def allow_retransmission(self) -> bool:
        return self._allow_retransmission

    def application_data(self) -> bytes:
        return bytes(self._application_data)
    # 应用程序特定数据
    def set_application_data(self, data: bytes):
        self._application_data = bytearray(data)

    # def set_packetization_finish_time_ms(self, time: int):
    #     delta = VideoSendTiming.GetDeltaCappedMs(self._capture_time_ms, time)
    #     self.set_extension(VideoTimingExtension, delta, VideoTimingExtension.kPacketizationFinishDeltaOffset)

    # def set_pacer_exit_time_ms(self, time: int):
    #     delta = VideoSendTiming.GetDeltaCappedMs(self.capture_time_ms, time)
    #     self.set_extension(VideoTimingExtension, delta, VideoTimingExtension.kPacerExitDeltaOffset)

    # def set_network_time_ms(self, time: int):
    #     delta = VideoSendTiming.GetDeltaCappedMs(self.capture_time_ms, time)
    #     self.set_extension(VideoTimingExtension, delta, VideoTimingExtension.kNetworkTimestampDeltaOffset)

    # def set_network2_time_ms(self, time: int):
    #     delta = VideoSendTiming.GetDeltaCappedMs(self.capture_time_ms, time)
    #     self.set_extension(VideoTimingExtension, delta, VideoTimingExtension.kNetwork2TimestampDeltaOffset)

    def set_first_packet_of_frame(self, is_first_packet: bool):
        self._is_first_packet_of_frame = is_first_packet

    def is_first_packet_of_frame(self) -> bool:
        return self._is_first_packet_of_frame

    def set_is_key_frame(self, is_key_frame: bool):
        self._is_key_frame = is_key_frame

    def is_key_frame(self) -> bool:
        return self._is_key_frame
class Stream:
    def __init__(self):
        # 默认构造函数
        self.size = 0  # 已经发送的报文总大小，
        self.ssrc = 0  # 数据流的同步源标识符
        self.packet_queue = PriorityPacketQueue()  # 使用优先级队列管理数据包的队列
        self.priority_it = Dict[StreamPrioKey, List[int]]  # 迭代器，用于指向某个元素，这里可能是一个索引或其他标识

    # 拷贝构造函数（这里没有特别的拷贝构造函数，因为 Python 通常使用引用计数来处理对象的复制）

    def __del__(self):
        # 虚析构函数，用于执行必要的清理工作
        pass
"""简化Stream的优先级比较"""
class StreamPrioKey:
    def __init__(self, priority, size):
        self.priority = priority
        self.size = size
     #优先级不等时比较优先级，优先级相等时，发送过较少的stream优先级更高
    def __lt__(self, other):
        if self.priority != other.priority:
            return self.priority < other.priority
        return self.size < other.size

    

class QueuedPacket:
    def __init__(self, priority: int, enqueue_time: int, enqueue_order: int,
                 enqueue_time_it: Optional[Iterator[int]], packet: RtpPacketToSend):
        self._priority = priority
        self._enqueue_time = enqueue_time
        self._enqueue_order = enqueue_order
        self._is_retransmission = (packet.packet_type == RtpPacketMediaType.kRetransmission)
        self._enqueue_time_it = enqueue_time_it
        self._owned_packet = packet.release()

    def __lt__(self, other: 'QueuedPacket') -> bool:
        if (self._priority != other.priority_):
            return self._priority > other.priority_
        if (self._is_retransmission != other.is_retransmission_):
            return other.is_retransmission_

        return self._enqueue_order > other.enqueue_order_

    def priority(self) -> int:
        return self._priority

    def type(self) -> RtpPacketMediaType:
        # Placeholder for the actual logic
        return self._owned_packet.packet_type()

    def ssrc(self) -> int:
        # Placeholder for the actual logic
        return self._owned_packet.ssrc()

    def enqueue_time(self) -> int:
        return self._enqueue_time

    def is_retransmission(self) -> bool:
        return  self.type() == RtpPacketMediaType.kRetransmission

    def enqueue_order(self) -> int:
        return self._enqueue_order

    def rtp_packet(self) -> RtpPacketToSend:
        return self._owned_packet

    def enqueue_time_iterator(self) -> Optional[Iterator[int]]:
        return self._enqueue_time_it

    def update_enqueue_time_iterator(self, it: Optional[Iterator[int]]):
        self._enqueue_time_it = it

    def subtract_pause_time(self, pause_time_sum: int):
        # Placeholder for the actual logic
        self._enqueue_time-=pause_time_sum
"""优先级队列"""
class PriorityPacketQueue():
    def __init__(self):
        self.heap = []

    def push(self, packet:QueuedPacket):
        priority = packet.priority()  # Assuming Priority() returns an integer
        heapq.heappush(self.heap, (-priority, packet))

    def pop(self):
        if self.heap:
            _, packet = heapq.heappop(self.heap)
            return packet
        else:
            raise IndexError("pop from an empty PriorityPacketQueue")

    def empty(self):
        return not bool(self.heap)

class RoundRobinPacketQueue:
    def __init__(self,start_time) -> None:
        self._transport_overhead_per_packet_:int=0
        self._time_last_updated:int=start_time
        self._paused:bool=False
        self._size_packets:int=0 # 报文数量
        self._max_size:int=kMaxLeadingSize #最大报文数量
        self._size:int=0 #报文总字节数
        self._queue_time_sum:int=0
        self._pause_time_sum: int=0
        self._stream_priorities: dict[StreamPrioKey, int] # streampropkey-ssrc map
        self._streams: dict[int, Stream] #ssrc stream map
        self._enqueue_times: PriorityQueue[int] #报文入队时间，用于找到最老的报文时间
        self._single_packet_queue: Optional[QueuedPacket] #存一个报文的queue
        self._include_overhead: bool=False
    def packet_size(self,packet:QueuedPacket)->int:
        packet_size=packet.rtp_packet().payload_size()+packet.rtp_packet.padding_size()
        return packet_size
    def maybe_promote_single_packet_to_normal_queue(self)->None:
        if self._single_packet_queue:
            self.push(self._single_packet_queue)
            self._single_packet_queue.reset()
    def get_highest_priority_stream(self):
        if not self._stream_priorities:
            return None
        ssrc=next(iter(self._stream_priorities.begin()))
        stream_info=self._streams.get(ssrc)
        if stream_info and stream_info.priority_it==next(iter(self._stream_priorities)):
            return stream_info
        return None
    # 插入不同优先级的报文
    def push(self,priority:int,enqueue_time:int,enqueue_order:int,packet:RtpPacketToSend)->None:
        assert packet.packet_type is not None, "Packet type must have a value."
        # 没有存书任何报文时直接存在_single_packet_queue中，不进队列
        if self._size_packets == 0:
            # Single packet fast-path.
            self._single_packet_queue = QueuedPacket(
                priority, enqueue_time, enqueue_order,
                self._enqueue_times.end() if self._enqueue_times else None,
                packet
            )
            self.update_queue_time(enqueue_time)
            self._single_packet_queue.subtract_pause_time(self._pause_time_sum)
            self._size_packets = 1
            self._size += self.packet_size(self._single_packet_queue)
        else:
            # 如果_single_packet_queue有数据，先push到queue中
            self.maybe_promote_single_packet_to_normal_queue()
            #调用另外一个push函数插入到队列中
            self.push(QueuedPacket(
                priority, enqueue_time, enqueue_order,
                self._enqueue_times.insert(enqueue_time),
                packet
            ))
    # 插入数据到队列中
    def push(self,packet:QueuedPacket):
        # 1. 根据报文ssrc找到对应的stream，没有找到则创建一个新的stream
        stream_info=self._streams.get(packet.ssrc())
        if stream_info is None:
            stream_info = self._streams[packet.ssrc()]=Stream()
            stream_info.priority_it=None #暂时不确定该stream的优先级，即没有被调度
            stream_info.ssrc=packet.ssrc()
        stream=stream_info
        # 2. 若该stream没有被调度，则加入stream_priorities
        if stream.priority_it is None:
            assert not self.is_ssrc_scheduled(stream.ssrc)
            stream.priority_it=self._stream_priorities[StreamPrioKey(packet.priority,stream.size)] = packet.ssrc
        # 3. 若当前报文优先级比之前的小，说明优先级有变化，更新优先级
        elif packet.priority < stream.priority_it.priority:
            del self._stream_priorities[stream.priority_it]
            stream.priority_it=self._stream_priorities[StreamPrioKey(packet.priority,stream.size)]=packet.ssrc
        assert stream.priority_it is not None
        
        if packet.enqueue_time_it==self._enqueue_times.end():
            packet.update_enqueue_time_it(self._enqueue_times.insert(packet.enqueue_time()))
        else:
            self.update_queue_time(packet.enqueue_time)
            packet.subtract_pause_time(self._pause_time_sum)
            self._size_packets+=1
            self._size+=self.packet_size(packet)
        stream.packet_queue.push(packet)
    # 弹出一个即将发送的报文
    def pop(self)->RtpPacketToSend:
        # 1. 如果_single_packet_queue中有数据，则直接返回该数据
        if(self._single_packet_queue is not None):
            assert self._stream_priorities.empty()==False
            rtp_packet=self._single_packet_queue.rtp_packet()
            self._single_packet_queue.reset()
            self._queue_time_sum=0
            self._size_packets=0
            self._size=0
            return rtp_packet
        # 2. 否则，返回优先级最高的stream，获取最前面的报文
        assert self.empty()==False
        stream = self.get_highest_priority_stream()#获取优先级最高的stream
        queued_packet = stream.packet_queue.top()# 获取最前面的报文
        del self._stream_priorities[stream.priority_it]# 因为有弹出报文，优先级需要更新
        # 计算该报文在queue中的时间
        time_in_non_paused_state=(self._time_last_updated-queued_packet.enqueue_time()-self._pause_time_sum)
        self._queue_time_sum-=time_in_non_paused_state
        # 删除该报文的queue time
        assert queued_packet.enqueue_time()!=self._enqueue_times.end()
        del self._enqueue_times[queued_packet.enque_time()]
        # 报文发送后，更新stream发送的报文字节数，为了避免发送码率较低的stream一直处于较高优先级发送过多，限制了最低发送字节数
        packet_size=self.packet_size(queued_packet)
        stream.size=max(stream.size+packet_size,self._max_size-kMaxLeadingSize)
        self._max_size=max(self._max_size,stream.size)
        
        self._size-=packet_size
        self._size_packets-=1
        assert self._size_packets>0 or self._queue_time_sum==0
        rtp_packet=queued_packet.rtppacket()
        stream.packet_queue.pop()
        # 如果剩余报文需要发送，更新调度优先级
        assert not self.is_ssrc_scheduled(stream.ssrc)
        if stream.packet_queue.empty():
            stream.priority_it=None
        else:
            priority=stream.packet_queue.top().priority()
            stream.priority_it=self._stream_priorities[StreamPrioKey(priority,stream.size)]=stream.ssrc
        return rtp_packet
    def is_ssrc_scheduled(self,ssrc:int)->bool:
        return ssrc in self._stream_priorities.values()
    # queue是否为空
    def empty(self)->bool:
        if self._size_packets == 0:
            assert not self._single_packet_queue and not self._stream_priorities, "Assertion failed: single_packet_queue_ is not None and stream_priorities_ is not empty."
            return True
        assert  self._single_packet_queue or  self._stream_priorities, "Assertion failed: single_packet_queue_ is not None and stream_priorities_ is not empty."
        return False

    # 报文总数
    def size_in_packets(self)->int:
        return self._size_packets
    # 报文总字节数
    def size(self):
        return self._size
    # 队列中最旧的报文时间戳
    def oldest_queuetime(self):
       if self._single_packet_queue :
           return self._single_packet_queue.enqueue_time()
       if self.empty():
           return datetime.min
       assert not self._enqueue_times.empty()
       return min(self._enqueue_times)
    #队列中的报文按照当前码率发送出去所用到的时间 
    def average_queuetime(self):
        if self.empty():
            return 0
        return self._queue_time_sum/self._size_packets
    # 更新内部状态
    def update_queue_time(self,now:int):
        assert self._time_last_updated <= now
        if now ==self._time_last_updated:
            return 
        delta=now-self._time_last_updated
        if self._paused:
            self._pause_time_sum+=delta
        else:
            self._queue_time_sum+=timedelta(microseconds=delta.microseconds*self._size_packets)
        self._time_last_updated=now
    # 暂停queue处理    
    def set_pause_state(self,paused:bool,now:int):
        if self._paused == paused:
            return
        self.update_queue_time(now)
        self._paused=paused

    
    def __del__(self):
        # 虚析构函数，用于执行必要的清理工作
        while(not self.empty()):
            self.pop()