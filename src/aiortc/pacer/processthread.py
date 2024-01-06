import threading
import time
from ..import clock
import queue
import datetime
import asyncio
from datetime import timedelta
kCallProcessTmmediately = -1
class Module():
    def time_until_next_process(self):
        pass
    def process(self):
        pass
    def process_thread_attached(self):
        pass
class ProcessThreadImpl:
    def __init__(self, thread_name):
        self.stop_ = False
        self.thread_name_ = thread_name
        self.modules_ = []
        self.lock_ = threading.Lock()
        # self.queue_ = queue.Queue()
        # self.delayed_tasks_ = queue.PriorityQueue()
        self.wake_up_ = threading.Event()
        self.k_call_process_immediately:int =-1
        self.thread_=None
    def start(self):
        # Start the thread
        if self.thread_ is not None:
            return 
        self.stop_=False
        for m in self.modules_:
            m.module.process_thread_attached(self)
        self.thread_ = threading.Thread(target=self.run_sync, name=self.thread_name_)
        self.thread_.start()
    def __del__(self):
        # while not self.delayed_tasks_.empty():
        #     task = self.delayed_tasks_.get().task
        #     del task
        # while not self.queue_.empty():
        #     task=self.queue_.get()
        #     del task
        pass
    # 唤醒出队线程：立即发送数据报文
    def wake_up(self,module:Module):
        with self.lock_:
            for m in self.modules_:
                if m.module == module:
                    m.next_callback=kCallProcessTmmediately
        self.wake_up_.set()
                    
    # def post_task(self,task:QueuedTask):
    #     with self.lock_:
    #         self.queue_.push(task)
    #     self.wake_up.set()
    # def post_delayed_task(self,task:QueuedTask,milliseconds:int):
    #     run_at_ms=int(time.time()*1000) + milliseconds
    #     recalculate_wakeup_time=False
    #     with self.lock_:
    #         recalculate_wakeup_time=(self.delayed_tasks_.empty() or run_at_ms < self.delayed_tasks_.queue[0].run_at_ms)
    #         self.delayed_tasks_.put(DelayedTask(run_at_ms,task))
    #     if recalculate_wakeup_time:
    #         self.wake_up_.set()
    def delete(self):
        self.stop()
    def stop(self):
        # Stop the thread
        with self.lock_:
            self.stop_ = True
        self.wake_up_.set()
        self.thread_.join()
        self.stop=False
        self.thread_=None
        for m in self.modules_:
            m.module.process_thread_attached(None) 
    def run_sync(self):
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(self.run_async())
    async def run_async(self):
        while await self.process():
            pass

    async def process(self):
        # Main processing loop
        # now = int(time.time() * 1000)  # Current time in milliseconds
        now = clock.current_datetime()
        next_checkpoint = now + timedelta(milliseconds=60*1000)# 设置下一个检查点的时间

        with self.lock_:
            if self.stop_:
                return False

            # Process modules
            for module_callback in self.modules_:
                if isinstance(module_callback.next_callback, int) and module_callback.next_callback == 0: # 直接获取下一次发送时间
                    module_callback.next_callback = self.get_next_callback_time(module_callback.module, now)
                # 需要立即发送数据
                if (isinstance(module_callback.next_callback, datetime.datetime) and module_callback.next_callback <= now) or (isinstance(module_callback.next_callback, int) and module_callback.next_callback == self.k_call_process_immediately):
                    await module_callback.module.process() # 调用process回调函数执行发送
                    new_now = clock.current_datetime()
                    module_callback.next_callback = self.get_next_callback_time(module_callback.module, new_now)# 获取下一次发送时间

                if isinstance(module_callback.next_callback, datetime.datetime) and module_callback.next_callback < next_checkpoint:
                    next_checkpoint = module_callback.next_callback

            # # Process delayed tasks 处理延迟任务队列：将到期的任务移动到普通任务队列
            # while not self.delayed_tasks_.empty() and self.delayed_tasks_.top().run_at_ms <= now:
            #     self.queue_.put(self.delayed_tasks_.get().task)
            
            # if not self.delayed_tasks_.empty():
            #     next_checkpoint = min(next_checkpoint, self.delayed_tasks_.top().run_at_ms)
            # # 处理普通任务队列：执行任务的run方法然后删除任务对象
            # # Process tasks in the queue
            # while not self.queue_.empty():
            #     task = self.queue_.get()
            #     task.run()

        time_to_wait = next_checkpoint - clock.current_datetime() #ms
        if time_to_wait.total_seconds() * 1000 > 0:
            self.wake_up_.wait(time_to_wait.total_seconds() * 1000 )

        return True
    # 获取下一次pacer发送的时间 
    def get_next_callback_time(self, module:Module, time_now:datetime.datetime)->datetime.datetime:
        interval = module.time_until_next_process()# 获取等待的时间间隔
        if interval.total_seconds() * 1000 < 0:
            return time_now
        return time_now + interval

    def register_module(self, module):
        if self.thread_:
            module.process_thread_attached(self)
        with self.lock_:
            self.modules_.append(ModuleCallback(module))
        self.wake_up_.set()

    def deregister_module(self, module):
        with self.lock_:
            self.modules_ = [mc for mc in self.modules_ if mc.module != module]
        module.process_thread_attached(None)


# Define ModuleCallback class if not already defined
class ModuleCallback():
    def __init__(self, module:Module):
        self.module = module # 回调函数模块
        self.next_callback :int=0 # Initialize to 0 指示了何时触发发送：0表示直接获取下一次的发送时间，-1表示立即发送
# class DelayedTask():
#     def __init__(self,run_at_ms:int,task:QueuedTask) -> None:
#         self.run_at_ms:int=run_at_ms
#         self.task:QueuedTask=task