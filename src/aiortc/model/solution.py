import numpy
import numpy as np
import torch
import onnxruntime as ort
class MetaPolicy(object):
    def __init__(self, num_history_step: int, **kwargs):
        super(MetaPolicy, self).__init__()
        """parameters"""
        self.num_history_step = num_history_step
        """state dict"""
        self._state_dict = None
        self.history=[]
        self.send_rate_range=kwargs["send_rate_range"]
        self.reset_state()
        """constants"""
        self.constant = {
            'state_dim': 30 * 5,
            'action_dim': 21,
            'send_rate_map_bps': np.linspace(-self.send_rate_range, self.send_rate_range, 21, endpoint=True),  # alpha: -2~2
            'initial_rate_bps': 100 * 1e3,
            'min_rate': 100 * 1e3,
            'max_rate': 1500000,
            'max_bandwidth': 5.0e6,  # used to normalize throughput, bps
            'delay_jitter_smooth_factor': 16,
            'delay_jitter_sample_interval_s': 30e-3,
            'reward_func': self.reward_func,
            'interval': 0.1
        }
        """init epoch specified signals"""
        self.reset_epoch_signals()
        """global signals"""
        self.just_updated_sendrate = False
        if 'safeguard' in kwargs.keys():
            self.safeguard = kwargs['safeguard']
        else:
            self.safeguard = 0.0
        if 'record_history' in kwargs.keys():
            self.record_history = kwargs['record_history']
        else:
            self.record_history = False
        if 'use_differential_reward' in kwargs.keys():
            self.use_differential_reward = True
        else:
            self.use_differential_reward = False
        if 'is_inference' in kwargs.keys():
            self.is_inference = kwargs['is_inference']
        else:
            self.is_inference = True
        """policy"""
        self.ort_session = ort.InferenceSession(kwargs["checkpoint_path"])
    def reward_func(instant_rate_mbps, instant_delay_s, instant_loss_rate, send_rate_jitter_mbps):
        reward = 50 * instant_rate_mbps - 50 * instant_loss_rate - 200 * instant_delay_s - send_rate_jitter_mbps * 20
        return reward
    def reset_epoch_signals(self):
        """
        init epoch specified signals
        """
        self.USE_CWND = False
        self.send_rate = self.constant['initial_rate_bps'] / 8  # initial rete
        self.action = 0
        self.fd_id = 0
        # last
        self.last_send_rate_bps = 0
        self.send_rate_jitter_bps = 0
        # counter
        self.epoch_step_cnt = 0
        self.epoch_update_cnt = 0
        # history
        self.epoch_history = {
            'reward': {},
            'state': {},
            'action': {},
            'others': {}
        }
        self.device="cpu"
    @property
    def state(self):
        s = np.array(list(self._state_dict.values()))
        s = np.concatenate(s, axis=0)
        return s

    def reset_state(self):
        n_h = self.num_history_step
        self._state_dict = {
            "rtt_s": np.zeros(n_h),  # delay
            "inter_packet_delay_s": np.zeros(n_h),  # delay_jit
            "loss_rate": np.zeros(n_h),  # loss
            "recv_throughput_bps": np.zeros(n_h),  # throughput
            "last_final_estimation_rate_bps": np.zeros(n_h)  # action_history
        }

    def append_to_state(self, state: dict):
        # fifo: https://numpy.org/doc/stable/reference/generated/numpy.roll.html
        # ([oldest, ..., newest])
        for key in self._state_dict.keys():
            self._state_dict[key] = np.roll(self._state_dict[key], -1)
            self._state_dict[key][-1] = state[key]
    def append_to_history(self, field: str, data: dict):
        # add new metrics
        for key in data.keys():
            if key not in self.epoch_history[field].keys():
                self.epoch_history[field][key] = []
        # append
        for key in data.keys():
            if isinstance(data[key], list):
                self.epoch_history[field][key] += data[key]
            else:
                self.epoch_history[field][key].append(data[key])

    def estimate_bandwidth(self, cur_time,instant_delay,instant_delay_jitter,instant_loss_rate,instant_rate_bps):
            """state"""
            self.append_to_state({
                "rtt_s": instant_delay,  # delay
                "inter_packet_delay_s": instant_delay_jitter,  # delay_jit
                "loss_rate": instant_loss_rate,  # loss
                "recv_throughput_bps": instant_rate_bps / self.constant['max_bandwidth'],  # throughput
                "last_final_estimation_rate_bps": self.last_send_rate_bps / self.constant['max_bandwidth']
            })
            state = self.state
            """action"""
            probs = self.ort_session.run(None, {'input': state.astype(numpy.float32)})[0]
            actions_tensor = numpy.where(probs == numpy.max(probs))
            self.action = actions_tensor[0]
            self.just_updated_sendrate = True
            # map action to rate in bps
            alpha = self.constant['send_rate_map_bps'][self.action]
            send_rate_bps = max(self.last_send_rate_bps * np.exp(alpha), self.constant['min_rate'])

            send_rate_bps = np.clip(send_rate_bps, self.constant['min_rate'], self.constant['max_rate'])
            # low-pass-filter
            alpha = 1
            send_rate_bps = self.last_send_rate_bps * (1 - alpha) + send_rate_bps * alpha

            # calculate jitter
            self.send_rate_jitter_bps = abs(send_rate_bps - self.last_send_rate_bps)
            self.last_send_rate_bps = send_rate_bps
            # set send rate
            self.send_rate = send_rate_bps / 8.0-self.safeguard if send_rate_bps / 8.0>self.safeguard else send_rate_bps / 8.0 # bps to Bps
            if instant_delay>0.1 and 800*1e6/8.0>self.send_rate>400*1e3/8.0:
                self.send_rate=400*1e3/8.0
            self.fd_id += 1
            


class MetaSolution(MetaPolicy):
    def __init__(self, num_history_step: int, **kwargs):
        super().__init__(num_history_step, **kwargs)
    def cc_trigger(self, cur_time,instant_delay,instant_delay_jitter,instant_loss_rate,instant_rate_bps):
        """
        The part of algorithm to make congestion control, which will be call when sender get an event about acknowledge or lost from receiver.
        See more at https://github.com/AItransCompetition/simple_emulator/tree/master#congestion_control_algorithmpy.
        """
        super().estimate_bandwidth( cur_time,instant_delay,instant_delay_jitter,instant_loss_rate,instant_rate_bps)
        res = {
            'target_bitrate': self.send_rate    #(Bps)
        }
        return res
if __name__ == '__main__':
    #保守策略选择方案1
    solution=MetaSolution(num_history_step=30,send_rate_range=2.0,is_meta_test=False,safeguard=100*1e3/8.0,checkpoint_path="models/model2.onnx",
                          record_history=True,use_differential_reward=True,is_inference=True)
    #激进策略选择方案2
    '''solution=MetaSolution(num_history_step=30,send_rate_range=0.5,is_meta_test=False,safeguard=70*1e3/8.0,checkpoint_path="models/model1.onnx",
                          record_history=True,use_differential_reward=True,is_inference=True)'''
    #example
    #per 100ms run
    cur_time=0.0
    res=solution.cc_trigger(cur_time,0.1,0.2,0.3,0.4)
    print("cur_time:{0},target_bitrate:{0}".format(cur_time,res["target_bitrate"]))
