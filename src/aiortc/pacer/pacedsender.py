from threading import Lock
from enum import Enum,auto
from typing import Optional, List,Callable
from dataclasses import dataclass
from .roundrobinpacketqueue import RoundRobinPacketQueue
from ..rtp import RtpPacketToSend,RtpPacketMediaType
import logging
import math
from datetime import timedelta
from ..import clock
from .intervalbudget import IntervalBudget
import threading
logger = logging.getLogger(__name__)
"""
Pacer模块: 规定了报文的优先级，报文类型，Pacer调度模式
"""
kDefaultMinPacketLimit=timedelta(milliseconds=5)
kCongestedPacketInterval = timedelta(milliseconds=500)
kMaxDebtInTime=timedelta(milliseconds=500)
kMaxElapsedTime = timedelta(seconds=2)
kMaxProcessingInterval=timedelta(seconds=30)

kMaxExpectedQueueLength=timedelta(milliseconds=2000)
kDefaultPaceMultiplier=2.5
kPausedProcessInterval =kCongestedPacketInterval
kMinSleepTime=timedelta(milliseconds=1)
kMaxQueueLengthMs=2000

class ProcessMode(Enum):
        kDynamic = "kDynamic"
        kPeriodic = "kPeriodic"

def get_priority_for_type(packet_type):
    # Lower number takes priority over higher.
    k_first_priority = 0
    if packet_type == RtpPacketMediaType.kAudio: # 音频第一优先级
        # Audio is always prioritized over other packet types.
        return k_first_priority + 1
    elif packet_type == RtpPacketMediaType.kRetransmission: # 重传报第二优先级
        # Send retransmissions before new media.
        return k_first_priority + 2
    elif packet_type in [RtpPacketMediaType.kVideo, RtpPacketMediaType.kForwardErrorCorrection]: #视频包和FEC优先级为3
        # Video has "normal" priority, in the old speak.
        # Send redundancy concurrently to video. If it is delayed it might have a
        # lower chance of being useful.
        return k_first_priority + 3
    elif packet_type == RtpPacketMediaType.kPadding: #padding优先级最低
        # Packets that are in themselves likely useless, only sent to keep the
        # BWE high.
        return k_first_priority + 4
    else:
        # Handle unknown packet type or provide a default priority.
        return k_first_priority

class PacedSender:
    def __init__(self):
        self.process_mode_ :ProcessMode= ProcessMode.kPeriodic  # 假设有一个名为 PacingController 的类，并有一个 ProcessMode 枚举
        self.critsect_ = Lock()  # Python 中的锁对象
        self.pacing_controller_ = PacingController(self,self.process_mode_)  # 假设有一个名为 PacingController 的类
        self.process_thread_ :Optional[threading.Thread] 
        
    # 1. 输入报文：将报文交给PacingController.enqueue_packet处理
    def enqueue_packets(self, packets:List[RtpPacketToSend]):
        with self.critsect_:
            for packet in packets:
                self.pacing_controller_.enqueue_packet(packet)
        self.maybe_wakeup_process_thread()
    def pause(self):
        with self.critsect_:
            self.pacing_controller_.pause()
        if self.process_thread_:
            self.process_thread_.stop()
            self.process_thread_.start()
    def resume(self):
        with self.critsect_:
            self.pacing_controller_.resume()
        if self.process_thread_:
            self.process_thread_.wakeup()
    def set_congestion_window(self,congestion_window_size:int):
        with self.critsect_:
            self.pacing_controller_.set_congestion_window(congestion_window_size)
        self.maybe_wakeup_process_thread()
    def update_outstanding_data(self,outstanding_data:int):
        with self.critsect_:
            self.pacing_controller_.update_outstanding_data(outstanding_data)
        self.maybe_wakeup_process_thread()
    def set_pacing_rates(self,pacing_rate:int,padding_rate:int):
        with self.critsect_:
            self.pacing_controller_.set_pacing_rates(pacing_rate,padding_rate)     
        self.maybe_wakeup_process_thread() # 唤醒出队发送线程  
    def set_account_for_audio(self,account_for_audio:bool):
        with self.critsect_:
            self.pacing_controller_.set_account_for_audio(account_for_audio)
    def set_include_overhead(self):
         with self.critsect_:
            self.pacing_controller_.set_include_overhead()
    def oldest_packet_wait_time(self):
        with self.critsect_:
            return self.pacing_controller_.oldest_packet_wait_time()
    def queue_size_data(self):
        with self.critsect_:
            self.pacing_controller_.queue_size_data()
    def expected_queue_time(self):
        with self.critsect_:
            return self.pacing_controller_.expected_queue_time()
    def time_until_next_process(self):
        with self.critsect_:
            next_send_time=self.pacing_controller_.next_send_time()
            sleep_time=max(0,next_send_time - clock.CurrentTime())
            return sleep_time#ms
    def process(self):
        with self.critsect_:
            self.pacing_controller_.process_packets()
    def process_thread_attached(self,process_thread:threading.Thread):
        logging.info(f"ProcessThreadAttached {hex(id(process_thread))}")
        assert not process_thread or process_thread == self.process_thread_
    def send_rtp_packet(self,packet:RtpPacketToSend):
        return 
    def generate_padding(self):
        return 
    def set_queue_time_limit(self,limit:timedelta):
        with self.critsect_:
            self.pacing_controller_.set_queue_time_limit(limit)
        self.maybe_wakeup_process_thread()
    def maybe_wakeup_process_thread(self):
        # 在这里添加可能的唤醒处理线程逻辑
        # 告诉处理线程调用我们的 TimeUntilNextProcess() 方法以获取调用 Process() 的新时间。
        # if self.process_thread_ and self.process_mode_ == ProcessMode.kDynamic:
        #     self.process_thread_.wake_up(module_proxy=self.module_proxy_)
        return
    def __del__(self):
        if self.process_thread_:
            self.process_thread_.put(None)
            self.process_thread_.join()
            self.process_thread_ = None
class PacingController:
    def __init__(self, packet_sender:PacedSender, mode:ProcessMode):
        self._mode:ProcessMode = mode
        self._packet_sender:PacedSender = packet_sender
        self._padding_target_duration:int = self._get_dynamic_padding_target()
        self._min_packet_limit:int = kDefaultMinPacketLimit
        self._transport_overhead_per_packet:int = 0
        self._last_timestamp:int = clock.current_ms()
        self._paused:bool= False
        self._media_budget:IntervalBudget =  IntervalBudget(0,False) # 周期调度模式中媒体数据的budget
        self._padding_budget:IntervalBudget = IntervalBudget(0,False)# 周期调度模式中padding数据的budget
        self._media_debt:int = 0
        self._padding_debt:int = 0
        self._media_rate:int = 0
        self._padding_rate:int = 0
        self._pacing_bitrate:int = 0
        self._last_process_time:int = clock.current_ms()
        self._last_send_time:int = self._last_process_time
        self._congestion_window_size:int = (float('inf'))
        self._outstanding_data:int= 0
        self._queue_time_limit:int = kMaxExpectedQueueLength
        self._account_for_audio:bool = False
        self._include_overhead:bool = False
        self._packet_queue:RoundRobinPacketQueue = RoundRobinPacketQueue(self._last_process_time)  # 报文队列
        self._packet_counter:int = 0 # 报文数量
        # logging
        self.__log_debug: Callable[..., None] = lambda *args: None
        if logger.isEnabledFor(logging.DEBUG):
            self.__log_debug = lambda msg, *args: logger.debug(
                f"RTCRtpReceiver(%s) {msg}", self.__kind, *args
            )
    def pause(self):
        if not self._paused:
            self.__log_debug("PacedSender paused.")
        self._paused = True
        self._packet_queue.set_pause_state(True, self.clock.CurrentTime())
    def resume(self):
        if self._paused:
            self.__log_debug("PacedSender resumed.")
        self._paused = False
        self._packet_queue.set_pause_state(False, self.clock.CurrentTime())
    def is_paused(self)->bool:
        return self._paused
    def _get_dynamic_padding_target(self):
        padding_target = timedelta(milliseconds=5)
        return padding_target
    def _get_min_packet_limit(self):
        min_packet_limit_ms = 5  # Default value
        config = self.field_trials.Lookup("WebRTC-Pacer-MinPacketLimitMs")
        self._parse_field_trial(config, min_packet_limit_ms)
        return timedelta(milliseconds=min_packet_limit_ms)
    def set_congestion_window(self, congestion_window_size)->None:
        was_congested = self.congested()
        self._congestion_window_size = congestion_window_size
        if was_congested and not self.congested():
            elapsed_time = self._update_time_and_get_elapsed(self.clock.CurrentTime())
            self.update_budget_with_elapsed_time(elapsed_time)
    def congested(self)->bool:
        if math.ifinite(self._congestion_window_size):
            return self._outstanding_data>=self._congestion_window_size
        return False
    def update_outstanding_data(self, outstanding_data)->None:
        was_congested=self.congested()
        self._outstanding_data=outstanding_data
        if was_congested and not self.congested:
            elapsed_time=self._update_time_and_get_elapsed(self.current_time())
            self.update_budget_with_elapsed_time(elapsed_time)
    def current_time(self)->int:
        time=clock.current_ms()
        if time<self._last_timestamp:
            self.__log_debug( "Non-monotonic clock behavior observed. Previous timestamp: %d , new timestamp: %d",self._last_timestamp_.ms(),time.ms())
            time=self._last_timestamp
        self._last_timestamp=time
        return time
    def set_pacing_rates(self,pacing_rate:int,padding_rate:int)->None:
        self._media_rate=pacing_rate #kbps
        self._padding_rate=padding_rate #kbps
        self._pacing_bitrate=pacing_rate #kbps
        self._padding_budget.set_target_rate_kbps(padding_rate)#bps to kbps
        logging.info( "bwe:pacer_updated pacing_kbps= %d , padding_budget_kbps= %d ",self._pacing_bitrate,self._padding_rate)
    def expected_queue_time(self)->int:
        assert self.pacing_bitrate > 0, "Pacing bitrate must be greater than zero."
        return timedelta(milliseconds=(self.queue_size_data()*8*1000)/self._pacing_bitrate)

    def queue_size_packets(self)->int:
        return self._packet_queue.size_in_packets()
    def queue_size_data(self)->int:
        return self._packet_queue.size()
    def current_buffer_level(self)->int:
        return max(self._media_debt,self._padding_debt)
    
    def set_include_overhead(self)->None:
        self._include_overhead = True
        self._packet_queue.set_include_overhead()

    def set_account_for_audio(self, account_for_audio:bool)->None:
        self._account_for_audio = account_for_audio

    def SetMediaBudget(self, media_budget, media_debt):
        self.media_budget = media_budget
        self.media_debt = media_debt

    def SetPaddingBudget(self, padding_budget, padding_debt):
        self.padding_budget = padding_budget
        self.padding_debt = padding_debt
    def oldest_packet_wait_time(self)->int:
        oldest_packet=self._packet_queue.oldest_queuetime()
        if oldest_packet.is_infinite():
            return 0
        return self.current_time()-oldest_packet

    def update_time_and_get_elapsed(self, now:int)->int:
        if self._last_process_time.is_minus_infinity():
            return 0
        assert now >= self._last_process_time
        elapsed_time = now - self._last_process_time
        self._last_process_time_ = now
        if elapsed_time>kMaxElapsedTime:
            elapsed_time = kMaxElapsedTime
        return elapsed_time
    def set_queue_time_limit(self,limit:int)->None:
        self._queue_time_limit=limit
    def next_send_time(self)->int:
        now=self.current_time()
        if self._paused:
            return self._last_send_time+kPausedProcessInterval
        if self._mode==ProcessMode.kPeriodic:
            return self._last_process_time+self._min_packet_limit
        audio_enqueue_time=self._packet_queue.leading_audio_packet_enqueue_time()
        if audio_enqueue_time.has_value():
            return audio_enqueue_time
        if self.congested() or self._packet_counter==0:
            return self._last_send_time+kCongestedPacketInterval
        if self._media_rate>0 and not self._packet_queue.empty():
            return min(self._last_send_time+kPausedProcessInterval,self._last_process_time+self._media_debt/self._media_rate)
        if self._padding_rate>0 and self._packet_queue.empty():
            drain_time=max(self.media_debt/self._media_rate,self.padding_debt/self._padding_rate)
            return min(self._last_send_time+kPausedProcessInterval,self._last_process_time+drain_time)
        return self._last_process_time+kPausedProcessInterval
    def padding_to_add(self,recommended_probe_size:int,data_sent:int)->int:
        if not self._packet_queue.empty():
            return 0
        if self.congested():
            return 0
        if self._packet_counter==0:
            return 0
        if recommended_probe_size:
            if recommended_probe_size>data_sent:
                return recommended_probe_size-data_sent
            return 0
        if self._mode==ProcessMode.kPeriodic:
            return self._padding_budget.bytes_remaining()
        elif self._padding_rate>0 and self._padding_debt==0:
            return self._padding_target_duration*self._padding_rate
        return 0
    def get_pending_packet(self,target_send_time:int,now:int)->RtpPacketToSend:
        if self._packet_queue.empty():
            return None
        unpaced_audio_packet=self._packet_queue.leading_audio_packet_enqueue_time().has_value()
        if not unpaced_audio_packet:
            if self.congested():
                return None
            if self._mode==ProcessMode.kPeriodic:
                if self._media_budget.bytes_remaining()<=0:
                    return None
            else:
                if now<=target_send_time:
                    flush_time=self._media_debt/self._media_rate
                    if now+flush_time>target_send_time:
                        return None
        return self._packet_queue.pop()
    def on_packet_sent(self,packet_type:RtpPacketMediaType,packet_size:int,send_time:int)->None:
        audio_packet=packet_type==RtpPacketMediaType.kAudio
        if not audio_packet or self._account_for_audio:
            self.update_budget_with_sent_data(packet_size)
        self._last_send_time=send_time
        self._last_process_time=send_time
    def on_padding_sent(self,data_sent:int)->None:
        if data_sent>0:
            self.update_budget_with_sent_data(data_sent)
        self._last_send_time=self.current_time()
        self._last_process_time=self.current_time()
    def update_budget_with_elapsed_time(self,delta:int)->None:
        if self._mode==ProcessMode.kPeriodic:
            delta=min(kMaxProcessingInterval,delta)
            self._media_budget.increase_budget(delta)
            self._padding_budget.increase_budget(delta)
        else:
            self._media_debt-=min(self._media_debt,self._media_rate*delta)
            self._padding_debt-=min(self.padding_debt,self._padding_rate*delta)
    def update_budget_with_sent_data(self,size:int)->None:
        self._outstanding_data+=size
        if self._mode==ProcessMode.kPeriodic:
            self._media_budget.use_budget(size)
            self._padding_budget.use_budget(size)
        else:
            self._media_debt+=size
            self._media_debt=min(self._media_debt,self._media_rate*kMaxDebtInTime)
            self._padding_debt+=size
            self._padding_debt=min(self._padding_debt,self._padding_rate* kMaxDebtInTime)
    # 报文输入：决策优先级并插入queue
    def enqueue_packet(self, packet:RtpPacketToSend):
        assert self._pacing_bitrate > 0, "SetPacingRate must be called before InsertPacket."
        assert packet._packet_type, "Packet type must be set."
        # 获取当前报文的优先级
        priority = get_priority_for_type(packet._packet_type)
        self.enqueue_packet_internal(packet, priority)#probing相关处理以及会根据Pacing的处理模式（动态和周期两种模式）做budget的更新
    # 报文输入：按照报文优先级，将报文插入queue
    def enqueue_packet_internal(self, packet:RtpPacketToSend, priority:int):

        # TODO: Make sure tests respect this, replace with DCHECK.
        now = self.current_time()# 入队时间
        if packet.capture_time_ms() < 0:
            packet.set_capture_time_ms(now.ms())
        self._packet_queue.push(priority, now, self._packet_counter, packet)
        self._packet_counter += 1
    def should_send_keepalive(self,now:int)->bool:
        if self._paused or self.congested() or self._packet_counter==0:
            elapsed_since_last_send=now-self._last_send_time
            if elapsed_since_last_send>=kCongestedPacketInterval:
                return True
        return False
    # Pacer核心逻辑：周期调度queue中的报文，budget足够的时候从queue中取出报文发送,删除了动态调度的处理部分
    def process_packets(self)->None:
        now=currenttime()
        target_send_time=now
        previous_process_time=self._last_process_time
        # 距离上次处理的时间，限制不超过2s
        elapsed_time=self.update_time_and_get_elapsed(now)
        # 1. 保活
        if(self.should_send_keepalive(now)):
            if(self._packet_counter==0):
                self._last_send_time=now
            else:
                keepalive_data_sent=0
                keepalive_packets=self._packet_sender.generate_padding(1)
                for packet in keepalive_packets:
                    keepalive_data_sent+=packet.payload_size()+packet.padding_size()
                    self._packet_sender.send_rtp_packet(packet.PacedPacketInfo())
                self.on_padding_sent(keepalive_data_sent)
        if self._paused:
            return 
        # 2. 如果开启了drain_large_queues,queue中的数据难以以当前速率在剩余时间内发送出去，则适当提高当前发送码率（通过修改budget）
        if elapsed_time>0:
            target_rate=self._pacing_bitrate
            queue_size_data=self._packet_queue.size()
            if queue_size_data>0:
                self._packet_queue.update_queue_time()
                if self._drain_large_queues:
                    avg_time_left=max(timedelta(milliseconds=1),self._queue_time_limit-self._packet_queue.average_queuetime)
                    min_rate_need=queue_size_data/avg_time_left
                    if min_rate_need>target_rate:
                        target_rate=min_rate_need
                        self.__log_debug ("bwe:large_pacing_queue pacing_rate_kbps=%d"
                              ,target_rate.kbps())
            # 提高当前budget，用于尽快排空
            if self._mode==ProcessMode.kPeriodic:
                self._media_budget.set_target_rate_kbps(target_rate.kbps())
                self.update_budget_with_elapsed_time(elapsed_time)
            else:
                self._media_rate=target_rate
        data_sent=0
        while (not self._paused):
            # 3. 获取需要发送的报文，需要检查是否拥塞，budget是否足够
            rtp_packet=self.get_pending_packet(pacing_info,target_send_time,now)
            # 4. 当前无法发送媒体包，检查是否发送padding
            if rtp_packet==None:
                # // No packet available to send, check if we should send padding.
                break
            # 5. 封装发送报文，通过回调发送报文
            packet_type=rtp_packet.packet_type()
            packet_size=rtp_packet.payload_size()+rtp_packet.padding_size()
            if self._include_overhead:
                packet_size+=rtp_packet.headers_size()+self._transport_overhead_per_packet
            self._packet_sender.send_packet(rtp_packet,pacing_info)
            data_send+=packet_size
            # 6. 发送完成后，更新一些统计以及budget
            self.on_packet_sent(packet_type,packet_size,target_send_time)
        self._last_process_time=max(self._last_process_time,previous_process_time)
        
            
            