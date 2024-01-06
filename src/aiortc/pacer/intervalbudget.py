kWindowMs=500
class IntervalBudget:
    def __init__(self,initial_target_rate_kbps:int,can_build_up_underuse:bool) -> None:
        self._target_rate_kbps:int
        self._max_bytes_in_budget:int
        self._bytes_remaining:int=0
        self._can_build_up_underuse:bool=can_build_up_underuse
        self.set_target_rate_kbps(initial_target_rate_kbps)
    # 更新目标比特率
    def set_target_rate_kbps(self,target_rate_kbps:int)->None:
        self._target_rate_kbps=target_rate_kbps
        self._max_bytes_in_budget=(kWindowMs*target_rate_kbps)/8# 时间*速率=预算的最大数据字节数
        self._bytes_remaining=min(max(-self._max_bytes_in_budget,self._bytes_remaining),self._max_bytes_in_budget)
    # 根据时间流逝增加预算，距离上次更新时间间隔delta_time_ms
    def increase_budget(self,delta_time_ms:int)->None:
        bytes=self._target_rate_kbps*delta_time_ms/8 # 这段时间增长的预算
        #     1） 如果上次的budget没有用完（bytes_remaining_ > 0），如果没有设置can_build_up_underuse_
        # // 不会对上次的补偿，直接清空所有预算，开始新的一轮

        # // 2） 如果设置了can_build_up_underuse_标志，那意味着要考虑上次的underuse，
        # // 如果上次没有发送完，则本次需要补偿，见上面if逻辑
        if self.bytes_remaining()<0 or self._can_build_up_underuse:# 如果上次发送的过多（bytes_remaining_ < 0），那么本次发送的数据量会变少
            self._bytes_remaining=min(self._bytes_remaining+bytes,self._max_bytes_in_budget)
        else:
            self._bytes_remaining=min(bytes,self._max_bytes_in_budget)
    # 报文发送后减少预算
    def use_budget(self,bytes:int)->None:
        self._bytes_remaining=max(self._bytes_remaining-bytes,-self._max_bytes_in_budget)
    def bytes_remaining(self)->int:
        return max(0,self._bytes_remaining)
    def budget_ratio(self)->float:
        if self._max_bytes_in_budget==0:
            return 0.0
        return float(self._bytes_remaining/self._max_bytes_in_budget)
    def target_rate_kbps(self)->int:
        return self._target_rate_kbps