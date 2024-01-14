from typing import List, Optional, Tuple
import numpy as np
import math
from .rtp import RtpPacket,RtpPacketToSend
from .utils import uint16_add
from .clock import current_ms,datetime_from_ntp
import time
import logging
MAX_MISORDER = 100
logger = logging.getLogger(__name__)

def VCM_MAX(a, b):
    return a if a > b else b

def VCM_MIN(a, b):
    return a if a < b else b
kStartupDelaySamples=30
kFsAccuStartupSamples=5
class JitterFrame:
    def __init__(self, data: bytes, timestamp: int,stream_id:str) -> None:
        self.data = data
        self.timestamp = timestamp
        self.times_dur = {} #pts
        self.stream_id=stream_id
        self.render_time_ms:int= -1
        self.is_key_frame:bool=False
        self.frame_delay_ms:int=0 # 帧间延迟观测值
        self.jitter_delay_ms:int=0 # 估计的最优jitter delay
        self.receive_time_ms:int=0 # 最后一个数据包的接收时间


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
        self._preSendTime:int=0
        self._preRecvTime:int=0
        self._ts_delta:int=0
        self._tr_delta:int=0
        self._curSendTime:int=0
        self._curRecvTime:int=0
        self._jitter_estimator:VCMJitterEstimator=VCMJitterEstimator()
        
    @property
    def capacity(self) -> int:
        return self._capacity
    # 计算jitter观测值 = 相邻两帧的接收时间差值-相邻两帧的发送时间差值
    def calculate_delay(self,packets:List[RtpPacketToSend],frame:JitterFrame)->JitterFrame:
        for packet in packets:
            if packet._is_first_packet_of_frame:
                abs_send_time=(packet.extensions.abs_send_time << 14) & 0xFFFFFFFFFFFFFFFF
                abs_send_time=datetime_from_ntp(abs_send_time)
                self._curSendTime=abs_send_time #第一个包的发送时间
            if packet.marker:
                self._curRecvTime=packet.recv_time_ms() #最后一个包的接收时间
            if self._preSendTime != 0 and self._preRecvTime != 0:
                self._tr_delta=self._curRecvTime-self._preRecvTime
                self._ts_delta=(self._curSendTime-self._preSendTime).total_seconds()*1000
        self._preRecvTime=self._curRecvTime
        self._preSendTime=self._curSendTime
        frame.frame_delay_ms=(self._ts_delta-self._tr_delta)
        # logger.info("ts delta:{0},tr_delta:{1},frame.frame_delay_ms:{2}".format(self._ts_delta,self._tr_delta,frame.frame_delay_ms))
        frame.receive_time_ms=self._curRecvTime
        return frame
    def add(self, packet: RtpPacketToSend) -> Tuple[bool, Optional[JitterFrame],int,bool]:
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
        encodeframe,jit_dur,is_key_frame=self._remove_frame(packet.sequence_number)
        
        return pli_flag, encodeframe,jit_dur,is_key_frame
    """从缓冲区中移除一个完整的 RTP 帧"""
    def _remove_frame(self, sequence_number: int) -> Tuple[Optional[JitterFrame],int]:
        frame = None
        frames = 0
        packets = []
        remove = 0
        timestamp = None
        import logging
        logger = logging.getLogger(__name__)
        is_key_frame=False
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
                    is_key_frame=packets[0]._is_key_frame
                    frame.is_key_frame=is_key_frame
                    # 计算wait time
                    frame=self.calculate_delay(packets,frame)
                   
                    
                    
                # check we have prefetched enough
                frames += 1
                if frames >= self._prefetch: #检查是否已经预取足够数量的帧（self._prefetch），如果是，则移除之前的数据包，返回合并的帧和相关的抖动延迟
                    self.remove(remove)
                    jit_dur=current_ms()-self._packet_times_in[timestamp][-1]
                    return frame, jit_dur,is_key_frame
                
                # start a new frame
                packets = []
                timestamp = packet.timestamp

            packets.append(packet)

        return None, None,is_key_frame

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

class VCMJitterEstimator:
    def __init__(self) -> None:
        self._prevFrameSize:int=0
        self._fsSum:int=0 # 目前接收的视频帧数据总大小
        self._fsCount:int=0 #目前接收的视频帧的数量
        self._avgFrameSize:int=0
        self._varFrameSize:int=0
        self._maxFrameSize:int=0
        self._phi:int=0.97
        self._psi:int=0.9999
        self._startupCount:int=0
        self.time_deviation_upper_bound_:float=3.5
        self._noiseStdDevs:float=2.33
        self._noiseStdDevOffset:float=30.0
        self._numStdDevDelayOutlier:int=15
        self._numStdDevFrameSizeOutlier:int=3
        self._alphaCountMax:int=400
        self._thetaLow:int=0.000001
        self._fps:int=0
        self.reset()
    def reset(self):
        self._theta = np.array([(1.0/(512e3/8.0), 0)]).ravel()
        self._varNoise = 4.0
        self._thetaCov = [
        np.array([(1e-4, 0)]).ravel(),
        np.array(([0, 1e2])).ravel()
        ]
        self._Qcov = [
        np.array(([2.5e-10, 0])).ravel(),
        np.array(([0,  1e-10])).ravel()
        ]

        self._avgFrameSize = 500
        self._maxFrameSize = 500
        self._varFrameSize = 100
        self._lastUpdateT = -1
        self._prevEstimate = -1.0
        self._prevFrameSize = 0
        self._avgNoise = 0.0
        self._alphaCount = 1
        self._filterJitterEstimate = 0.0
        self._latestNackTimestamp = 0
        self._nackCount = 0
        self._latestNackTimestamp = 0
        self._fsSum = 0
        self._fsCount = 0
        self._startupCount = 0
        # _rttFilter.Reset();
        # fps_counter_.Reset();
    #计算延迟残差(反映网络噪声的大小)，并据此计算噪声均值和方差
    def deviation_from_expected_delay(self,frameDelayMs:int,deltaFSBytes:int)->float:
        return frameDelayMs-(self._theta[0]*deltaFSBytes+self._theta[1])
    # 1. 更新frame delay和帧大小 
    def update_estimate(self,frameDelayMS,frameSizeBytes,incompleteFrame)->None:
        if frameSizeBytes==0:
            return 
        deltaFS=frameSizeBytes-self._prevFrameSize
        # 1. 计算平均每帧的大小
        if self._fsCount<kFsAccuStartupSamples:
            self._fsSum+=frameSizeBytes
            self._fsCount+=1
        elif self._fsCount==kFsAccuStartupSamples:
            self._avgFrameSize=float(self._fsSum)/float(self._fsCount)
            self._fsCount+=1
        # 帧大小的 平均值和方差 的估计
        if not incompleteFrame or frameSizeBytes>self._avgFrameSize:
            avgFrameSize=self._phi*self._avgFrameSize+(1-self._phi)*frameSizeBytes
            # 当前帧和平均值差距不大时才通过以上0.97的平滑方式更新帧平均值
            if frameSizeBytes<self._avgFrameSize+2*math.sqrt(self._varFrameSize):
                # 如果帧的小超过平均值+2*标准差（95%置信区间），则认为是关键帧
                self._avgFrameSize=avgFrameSize
            self._varFrameSize=VCM_MAX(self._phi*self._varFrameSize+(1-self._phi)*(frameSizeBytes-avgFrameSize)*(frameSizeBytes-avgFrameSize),1.0)
        #更新最大帧大小
        self._maxFrameSize=VCM_MAX(self._psi*self._maxFrameSize,float(frameSizeBytes))
        logger.info("frameSizeBytes:{0},self._prevFrameSize:{1},self._fsSum:{2},self._fsCount:{3},self._avgFrameSize:{4},self._varFrameSize:{5}".format(frameSizeBytes,self._prevFrameSize,self._fsSum,self._fsCount,self._avgFrameSize,self._varFrameSize))
        if self._prevFrameSize==0:
            self._prevFrameSize=frameSizeBytes
            return 
        self._prevFrameSize=frameSizeBytes
        
        # 设定延迟抖动的上限：3.5*延迟噪声标准差+0.5，限制帧延迟在范围内
        max_time_deviation_ms=(self.time_deviation_upper_bound_*math.sqrt(self._varNoise)+0.5)
        frameDelayMS=max(min(frameDelayMS,max_time_deviation_ms),-max_time_deviation_ms)
        
        deviation=self.deviation_from_expected_delay(frameDelayMS,deltaFS)
        
        if (math.fabs(deviation) < self._numStdDevDelayOutlier * math.sqrt(self._varNoise) or
            frameSizeBytes > self._avgFrameSize + self._numStdDevFrameSizeOutlier * math.sqrt(self._varFrameSize)):
            # Update the variance of the deviation from the line given by the Kalman filter.
            self.estimate_random_jitter(deviation, incompleteFrame)
            
            # Prevent updating with frames which have been congested by a large frame,
            # and therefore arrives almost at the same time as that frame.
            # This can occur when we receive a large frame (key frame) which has been
            # delayed. The next frame is of normal size (delta frame), and thus deltaFS
            # will be << 0. This removes all frame samples which arrive after a key
            # frame.
            if (not incompleteFrame or deviation >= 0.0) and deltaFS > -0.25 * self._maxFrameSize:
                # Update the Kalman filter with the new data
                self.kalman_estimate_channel(frameDelayMS, deltaFS)
        else:
            nStdDev = self._numStdDevDelayOutlier if deviation >= 0 else -self._numStdDevDelayOutlier
            self.estimate_random_jitter(nStdDev * math.sqrt(self._varNoise), incompleteFrame)
        if self._startupCount>=kStartupDelaySamples:
            self.post_process_estimate()
        else:
            self._startupCount+=1
    def kalman_estimate_channel(self,frameDelayMs:int,deltaFSBytes:int):
        # Kalman filtering

        # Prediction，先验协方差预测
        # M = M + Q
        self._thetaCov[0][0] += self._Qcov[0][0]
        self._thetaCov[0][1] += self._Qcov[0][1]
        self._thetaCov[1][0] += self._Qcov[1][0]
        self._thetaCov[1][1] += self._Qcov[1][1]

        # Kalman gain 卡尔曼增益k的更新，
        # K = M * h' / (sigma2n + h * M * h') = M * h' / (1 + h * M * h')
        # h = [dFS 1]
        # Mh = M * h'
        # hMh_sigma = h * M * h' + R
        Mh = [self._thetaCov[0][0] * deltaFSBytes + self._thetaCov[0][1],
              self._thetaCov[1][0] * deltaFSBytes + self._thetaCov[1][1]]

        if self._maxFrameSize < 1.0:
            return
        # 观测噪声方差的计算使用指数滤波的方式从噪声协方差获得
        sigma = (300.0 * math.exp(-math.fabs(deltaFSBytes) / (1e0 * self._maxFrameSize)) + 1) * math.sqrt(self._varNoise)
        sigma = max(1.0, sigma)

        hMh_sigma = deltaFSBytes * Mh[0] + Mh[1] + sigma

        if -1e-9 <= hMh_sigma <= 1e-9:
            assert False
            return

        kalmanGain = [Mh[0] / hMh_sigma, Mh[1] / hMh_sigma]

        # Correction 后验期望的计算
        # theta = theta + K * (dT - h * theta)
        measureRes = frameDelayMs - (deltaFSBytes * self._theta[0] + self._theta[1])
        self._theta[0] += kalmanGain[0] * measureRes
        self._theta[1] += kalmanGain[1] * measureRes
        logger.info("theta[0]:{0}, kalmanGain[0]:{1}, measureRes{2}".format(self._theta[0],kalmanGain[0],measureRes))
        if self._theta[0] < self._thetaLow:
            self._theta[0] = self._thetaLow

        # M = (I - K * h) * M 后验方差的计算
        t00, t01 = self._thetaCov[0][0], self._thetaCov[0][1]
        self._thetaCov[0][0] = (1 - kalmanGain[0] * deltaFSBytes) * t00 - kalmanGain[0] * self._thetaCov[1][0]
        self._thetaCov[0][1] = (1 - kalmanGain[0] * deltaFSBytes) * t01 - kalmanGain[0] * self._thetaCov[1][1]
        self._thetaCov[1][0] = self._thetaCov[1][0] * (1 - kalmanGain[1]) - kalmanGain[1] * deltaFSBytes * t00
        self._thetaCov[1][1] = self._thetaCov[1][1] * (1 - kalmanGain[1]) - kalmanGain[1] * deltaFSBytes * t01

        # Covariance matrix, must be positive semi-definite.
        assert self._thetaCov[0][0] + self._thetaCov[1][1] >= 0
        assert self._thetaCov[0][0] * self._thetaCov[1][1] - self._thetaCov[0][1] * self._thetaCov[1][0] >= 0
        assert self._thetaCov[0][0] >= 0

    def noise_threshold(self)->float:
        nopiseThreshold=self._noiseStdDevs*math.sqrt(self._varNoise)-self._noiseStdDevOffset
        if nopiseThreshold<1.0:
            nopiseThreshold=1.0
        return nopiseThreshold
    #  jitter计算：theta0 * 帧长度delta + 噪声阈值
    #  噪声阈值：2.33 * 噪声标准差 - 30ms，2.33是正态分布里面的99%置信区间
    def calculate_estimate(self)->float:
        ret=self._theta[0]*(self._maxFrameSize-self._avgFrameSize)+self.noise_threshold()
        logger.debug("self._theta[0]:{0}, self._maxFrameSize: {1}, self._avgFrameSize: {2}, self.noise_threshold(): {3}".format(self._theta[0],self._maxFrameSize,self._avgFrameSize,self.noise_threshold()))
        if ret<1.0:
            if self._prevEstimate<=0.01:
                ret=1.0
            else:
                ret=self._prevEstimate
        if ret>10000.0:
            ret=10000.0
        self._prevEstimate=ret
        return ret 
    def post_process_estimate(self)->None:
        self._filterJitterEstimate=self.calculate_estimate()
    # /**
    #  * @description: 根据deviation，更新观测噪声均值和噪声方差
    #  * @param {d_dT} deviation
    #  * @param {incompleteFrame} 
    #  * @return {*}
    #  */
    def set_frame_rate(self,fps:int)->None:
        self._fps=fps
    def get_frame_rate(self)->int:
        return self._fps
    def estimate_random_jitter(self,d_dT:float,incompleteFrame:bool):
        # now = clock.current
        # if self._lastUpdateT != -1:
            # self.fps_counter_.AddSample(now - self._lastUpdateT)
        # self._lastUpdateT = now
        # // alpha是用来更新噪声的，alpha*old + (1-alpha)*new
        # // _alphaCountMax 默认 400，alpha =  (cout-1)/count
        if self._alphaCount == 0:
            assert False
            return

        alpha = (self._alphaCount - 1) / self._alphaCount
        logger.info("alpha:{0},_alphaCount:{1}".format(alpha,self._alphaCount))
        self._alphaCount += 1
        if self._alphaCount > self._alphaCountMax:
            self._alphaCount = self._alphaCountMax
        #  alpha值需要根据帧率跳帧，帧率低的时候噪声越大，需要增大alpha，噪声均值和方差也越大
        fps = self.get_frame_rate()
        logger.info("fps:{0}".format(fps))
        # fps=30
        # // 在开始阶段（30个帧），估计会存在较大的噪声，会根据fps调整alpha:
        # // 30fps的时候，rate_scale=1
        # // 10fps的时候，rate_scale>1
        if fps > 0.0:
            rate_scale = 30.0 / fps
            if self._alphaCount < kStartupDelaySamples:
                rate_scale = (self._alphaCount * rate_scale + (kStartupDelaySamples - self._alphaCount)) / kStartupDelaySamples
            alpha = pow(alpha, rate_scale)
        #  更新噪声平均值以及噪声方差，使用加权平均的方式，实现了对噪声的平滑估计
        avgNoise = alpha * self._avgNoise + (1 - alpha) * d_dT
        varNoise = alpha * self._varNoise + (1 - alpha) * (d_dT - self._avgNoise) * (d_dT - self._avgNoise)
        logger.info("alpha:{0},avgNoise:{1},varNoise:{2},d_dT{3}".format(alpha,avgNoise,varNoise,d_dT))
        if not incompleteFrame or varNoise > self._varNoise:
            self._avgNoise = avgNoise
            self._varNoise = varNoise

        if self._varNoise < 1.0:
            # The variance should never be zero, since we might get stuck and consider
            # all samples as outliers.
            self._varNoise = 1.0
        