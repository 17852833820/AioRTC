kWindowMs=500
class IntervalBudget:
    def __init__(self,initial_target_rate_kbps:int,can_build_up_underuse:bool) -> None:
        self._target_rate_kbps:int
        self._max_bytes_in_budget:int
        self._bytes_remaining:int=0
        self._can_build_up_underuse:bool=can_build_up_underuse
        self.set_target_rate_kbps(initial_target_rate_kbps)
    def set_target_rate_kbps(self,target_rate_kbps)->None:
        self._target_rate_kbps=target_rate_kbps
        self._max_bytes_in_budget=(kWindowMs*target_rate_kbps)/8
        self._bytes_remaining=min(max(-self._max_bytes_in_budget,self._bytes_remaining),self._max_bytes_in_budget)
    def increase_budget(self,delta_time_ms:int)->None:
        bytes=self._target_rate_kbps*delta_time_ms/8
        if self.bytes_remaining<0 or self._can_build_up_underuse:
            self._bytes_remaining=min(self._bytes_remaining+bytes,self._max_bytes_in_budget)
        else:
            self._bytes_remaining=min(bytes,self._max_bytes_in_budget)
    def use_budget(self,bytes:int)->None:
        self._bytes_remaining=max(self._bytes_remaining-bytes,self._max_bytes_in_budget)
    def bytes_remaining(self)->int:
        return max(0,self._bytes_remaining)
    def budget_ratio(self)->float:
        if self._max_bytes_in_budget==0:
            return 0.0
        return float(self._bytes_remaining/self._max_bytes_in_budget)
    def target_rate_kbps(self)->int:
        return self._target_rate_kbps