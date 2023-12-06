# 端到端视频传输
## 一、运行
### 1.启动发送端 
cd aiortc
python3 ./examples/videostream-cli/cli.py offer --play-from ./dataset/test.mp4
运行产生offer，默认交换SDP的模式是复制粘贴，需手动复制offer粘贴到接收端
{"sdp": "v=0\r\no=- 3910748632 3910748632 IN IP4 0.0.0.0\r\ns=-\r\nt=0 0\r\na=group:BUNDLE 0\r\na=msid-semantic:WMS *\r\nm=video 61128 UDP/TLS/RTP/SAVPF 97 98 99 100 101 102\r\nc=IN IP4 172.19.240.1\r\na=sendrecv\r\na=extmap:1 urn:ietf:params:rtp-hdrext:sdes:mid\r\na=extmap:3 http://www.webrtc.org/experiments/rtp-hdrext/abs-send-time\r\na=mid:0\r\na=msid:db6d556e-4b65-43dc-9b65-c5842ff7d7d0 9964603f-35ce-4037-9107-79cef798f53f\r\na=rtcp:9 IN IP4 0.0.0.0\r\na=rtcp-mux\r\na=ssrc-group:FID 3801919459 2472625064\r\na=ssrc:3801919459 cname:4f80c0c7-36c8-4f3a-9896-2c7157606068\r\na=ssrc:2472625064 cname:4f80c0c7-36c8-4f3a-9896-2c7157606068\r\na=rtpmap:97 VP8/90000\r\na=rtcp-fb:97 nack\r\na=rtcp-fb:97 nack pli\r\na=rtcp-fb:97 goog-remb\r\na=rtpmap:98 rtx/90000\r\na=fmtp:98 apt=97\r\na=rtpmap:99 H264/90000\r\na=rtcp-fb:99 nack\r\na=rtcp-fb:99 nack pli\r\na=rtcp-fb:99 goog-remb\r\na=fmtp:99 level-asymmetry-allowed=1;packetization-mode=1;profile-level-id=42001f\r\na=rtpmap:100 rtx/90000\r\na=fmtp:100 apt=99\r\na=rtpmap:101 H264/90000\r\na=rtcp-fb:101 nack\r\na=rtcp-fb:101 nack pli\r\na=rtcp-fb:101 goog-remb\r\na=fmtp:101 level-asymmetry-allowed=1;packetization-mode=1;profile-level-id=42e01f\r\na=rtpmap:102 rtx/90000\r\na=fmtp:102 apt=101\r\na=candidate:84525827d984b903f6250bb33f1836c3 1 udp 2130706431 172.19.240.1 61128 typ host\r\na=candidate:a794a0b8ea98fb9d35a09b0aa6852435 1 udp 2130706431 192.168.31.22 61129 typ host\r\na=candidate:4bc0bb2f32cea4d0dad824a6fa6164d4 1 udp 1694498815 115.156.132.20 61129 typ srflx raddr 192.168.31.22 rport 61129\r\na=end-of-candidates\r\na=ice-ufrag:tAct\r\na=ice-pwd:DxwfHDfHXgNeUJgZVFxeTG\r\na=fingerprint:sha-256 FB:60:A0:E7:20:D7:C6:2A:CC:5B:00:3F:F6:0D:EC:05:D6:A7:FF:03:4D:4A:DC:60:F0:2D:DB:C5:71:F9:54:86\r\na=setup:actpass\r\n", "type": "offer"}
offer信息解读：


### 2. 启动接收端
cd aiortc
python3 ./examples/videostream-cli/cli.py answer --record-to ./receive_data/video.mp4
将发送端产生的offer粘贴在终端，产生answer

{"sdp": "v=0\r\no=- 3910748693 3910748693 IN IP4 0.0.0.0\r\ns=-\r\nt=0 0\r\na=group:BUNDLE 0\r\na=msid-semantic:WMS *\r\nm=video 59269 UDP/TLS/RTP/SAVPF 97 98 99 100 101 102\r\nc=IN IP4 172.19.240.1\r\na=sendrecv\r\na=extmap:1 urn:ietf:params:rtp-hdrext:sdes:mid\r\na=extmap:3 http://www.webrtc.org/experiments/rtp-hdrext/abs-send-time\r\na=mid:0\r\na=msid:7a3706e2-10e7-4b75-aad6-22db48fd7e3d 4e618f8f-b68a-446f-a913-b4959abfe64c\r\na=rtcp:9 IN IP4 0.0.0.0\r\na=rtcp-mux\r\na=ssrc-group:FID 2317366329 1460064368\r\na=ssrc:2317366329 cname:4830261d-175a-4378-9833-4492f107983f\r\na=ssrc:1460064368 cname:4830261d-175a-4378-9833-4492f107983f\r\na=rtpmap:97 VP8/90000\r\na=rtcp-fb:97 nack\r\na=rtcp-fb:97 nack pli\r\na=rtcp-fb:97 goog-remb\r\na=rtpmap:98 rtx/90000\r\na=fmtp:98 apt=97\r\na=rtpmap:99 H264/90000\r\na=rtcp-fb:99 nack\r\na=rtcp-fb:99 nack pli\r\na=rtcp-fb:99 goog-remb\r\na=fmtp:99 level-asymmetry-allowed=1;packetization-mode=1;profile-level-id=42001f\r\na=rtpmap:100 rtx/90000\r\na=fmtp:100 apt=99\r\na=rtpmap:101 H264/90000\r\na=rtcp-fb:101 nack\r\na=rtcp-fb:101 nack pli\r\na=rtcp-fb:101 goog-remb\r\na=fmtp:101 level-asymmetry-allowed=1;packetization-mode=1;profile-level-id=42e01f\r\na=rtpmap:102 rtx/90000\r\na=fmtp:102 apt=101\r\na=candidate:84525827d984b903f6250bb33f1836c3 1 udp 2130706431 172.19.240.1 59269 typ host\r\na=candidate:a794a0b8ea98fb9d35a09b0aa6852435 1 udp 2130706431 192.168.31.22 59270 typ host\r\na=candidate:4bc0bb2f32cea4d0dad824a6fa6164d4 1 udp 1694498815 115.156.132.20 59270 typ srflx raddr 192.168.31.22 rport 59270\r\na=end-of-candidates\r\na=ice-ufrag:gl1l\r\na=ice-pwd:ggU8to1A3mvKruiqC2MOet\r\na=fingerprint:sha-256 6D:F9:82:22:78:06:81:EE:10:60:F6:0B:AE:73:6D:07:1D:E4:DB:3F:FA:32:14:16:FB:77:DE:A6:FA:7F:7F:D2\r\na=setup:active\r\n", "type": "answer"}
answer信息解读：

最后将协商产生的answer复制粘贴在发送端的终端，至此，SDP信令交换完成，开始视频传输

接收端输出相关配置信息：
![](.assert/20231205-160826.jpg)

## 二、代码分析
### 1. 交换SDP建立连接后如何开始视频的编码和发送
连接建立后，视频的编码和发送通常是在将本地媒体流添加到 RTCPeerConnection 后自动进行的，而不需要手动控制
连接器pc包含传输器，传输器包含发送器，发送器关联了音视频轨道
交换协商SDP完成后，可以开始视频传输

1. 查询MediaStreamTrack的state是否为live，是进入2
2. 调用传输层RTCDtlsTransport的异步start方法进行安全校验
3. 调用RTCRtpSender的异步send方法开始发送，分别启动异步任务_run_rtp和_run_rtcp
1. 在RTPSender的_run_rtp异步方法中：进行RTP包的编码和发送，编码参数配置为RTCRtpCodecParameters
2. 具体编码过程如下：
   1. （传输层）主循环不断获取RTPSender绑定的track中的媒体数据，调用传输层RTCDtlsTransport的异步_recv_next方法接收datagram，包括DTLS报文和RTP，RTCP报文，根据报文类型调用_handle_rtp_data或_handle_rtcp_data处理数据
   2. （传输层）若为RTP报文，调用_handle_rtp_data异步方法将RTP packet路由到类型一致的packet接收器
   3. 调用_next_encoded_frame进行编码
   4. （编码层）在_next_encode_frame内部，调用class PlayerStreamTrack的recv方法获取数据：VideoFrame或Packet
   5. （编码层）在_next_encode_frame内部，可以指定编码使用的编码器，如果是VideoFrame类型的数据调用编码器的encode方法执行编码，返回编码的payloads和时间戳；如果是Packet类型的数据，调用调用编码器的pack方法执行编码
      以H264编码为例：
      1. 获取到VideoFrame数据之后，调用编码且的encode()方法，输入参数为data和是否强制I帧
      2. 在encode（）内部：编码得到NAL单元，将其分片打包成packet列表，加上RTP 扩展头打包成RTP包列表，调用RTCDtlsTransport的_send_rtp方法发送RTP packet


在发送器中调用send方法启动RTP和RTCP任务

### 2. 会话和媒体描述
**SessionDescription**： 描述该会话
- MediaDescription list：包含该会话所有媒体描述信息的列表
**MediaDescription**：媒体描述信息
- kind
- fmt
- rtp：RTPParameters，包含支持的编解码器，头部扩展等
### 1. 如何控制发送速率的
接收器收到RTP数据包时调用_handle_rtp_packet方法，进行带宽估计并返回REMB反馈包RTCP_PSFB_APP
RemoteBitrateEstimator进行比特率估计
- incoming_bitrate
- incoming_bitrate_initialized
- estimator：OveruseEstimator
- detector：OveruseDetector
- rate_control：AimdRateControl
- add()：返回目标比特率
  
发送器接收到对端发来的REMB包后：receiver estimated maximum bitrate
### 2. 如何控制编码参数的
1. 修改默认的编解码器
   RTCRtpTransceiver的setCodecPreferences方法
   ```
   def force_codec(pc, sender, forced_codec):
    kind = forced_codec.split("/")[0]
    codecs = RTCRtpSender.getCapabilities(kind).codecs
    transceiver = next(t for t in pc.getTransceivers() if t.sender == sender)
    transceiver.setCodecPreferences(
        [codec for codec in codecs if codec.mimeType == forced_codec]
    )#将编解码器偏好设置为仅包含符合指定 forced_codec 的媒体类型的编解码器

    force_codec(pc, video_sender, args.video_codec)

    ```
    方法二：在pc add track之后，创建offer之前设置编码器
    ```
        capabilities = RTCRtpSender.getCapabilities("video")
        preferences = list(filter(lambda x: x.name == "H264", capabilities.codecs))
        preferences += list(filter(lambda x: x.name == "rtx", capabilities.codecs))
        transceiver = pc1.getTransceivers()[0]
        transceiver.setCodecPreferences(preferences)
    ```
### 3. 能否强制P/B帧编码，帧类型是内部还是外部决定的

### 4. 测量和计算端到端延迟和帧率


**RTCPRtpCodecParameters**用于设置编码参数


    
    The :class:`RTCRtpCodecParameters` dictionary provides information on
    codec settings.

    @dataclass
    class RTCRtpCodecParameters:
      mimeType: str
      "The codec MIME media type/subtype, for instance `'audio/PCMU'`."
      clockRate: int
      "The codec clock rate expressed in Hertz."
      channels: Optional[int] = None
      "The number of channels supported (e.g. two for stereo)."
      payloadType: Optional[int] = None
      "The value that goes in the RTP Payload Type Field."
      rtcpFeedback: List["RTCRtcpFeedback"] = field(default_factory=list)
      "Transport layer and codec-specific feedback messages for this codec."
      parameters: ParametersDict = field(default_factory=dict)
      "Codec-specific parameters available for signaling."
      """
       @property
    def name(self):
        return self.mimeType.split("/")[1]

    def __str__(self):
        s = f"{self.name}/{self.clockRate}"
        if self.channels == 2:
            s += "/2"
        return s

**RTCPeerConnection**作为RTP会话的实例
方法：
addTrack(self, track: MediaStreamTrack) -> RTCRtpSender  ：为pc添加视频流track，返回RTCRtpSender
getTransceivers（）获取包含RTCRtpTransceiver传输器的列表
addTransceiver（）用于向 PeerConnection 添加新的RTCRtpTransceiver传输器
createDataChannel（）创建数据通道，
setRemoteDescription（）
setLocalDescription（）
**RTCDataChannel** 数据通道参数：标签 (label)、最大数据包生命周期 (maxPacketLifeTime)、最大重传次数 (maxRetransmits)、是否有序 (ordered)、协议 (protocol)、是否已协商 (negotiated) 和数据通道的标识符 (id)
**RTCRtpTransceiver传输器**用于描述一个 RTCRtpSender 和一个 RTCRtpReceiver 的永久配对，以及它们之间的一些共享状态
参数：
        kind: str,传输的媒体类型
        receiver: RTCRtpReceiver,
        sender: RTCRtpSender,
        direction: str = "sendrecv",传输方向
setCodecPreferences（）设置编解码器偏好，修改默认的编解码器

**RTCRtpSender** 负责编码和发送数据，用于控制和获取关于如何编码和发送特定 MediaStreamTrack 到远程对等端的详细信息
- 输入：
  - MediaStreamTrack 实例或媒体种类字符串，如 'audio' 或 'video'
  - transport（RTCDtlsTransport 对象）
- 属性：
  - kind：媒体类型
  - encoder：编码器对象
  - force_keyframe：是否强制发送关键帧
  - rtp_exited,rtp_header_extensions_map：用于处理RTP相关的状态和任务
  - rtcp_exited,rtcp_starteed：用于处理RTCP相关的状态和任务
  - transport：与此发送器相关联的传输器对象
  - track
- 静态方法：
  - getCapabilities（）返回系统对编解码器，传输协议等的支持能力
- 异步方法
  - getStats（）获取RTP发送器的统计报告信息
  - send(parameters: RTCRtpSendParameters)：设置发送器相关参数，启动RTP和RTCP异步任务
  - stop()：停止RTP和RTCP任务
  - _handle_rtcp_packet（packet）：分别处理不同类型的RTCP包
  - _next_encoded_frame(codec: RTCRtpCodecParameters):执行视频编码
  - _retransmit(self, sequence_number: int)：重传丢失的RTP包
  - _run_rtp(self, codec: RTCRtpCodecParameters)：执行RTP异步任务，调用_next_encoded_frame进行视频编码
  - _run_rtcp（）执行RTCP任务
  - 
- 成员方法
  - replaceTrack（）
  - setTransport（）
  - 

getCapabilities（）返回指定媒体类型（音频或视频）的发送器的能力capabilities

**RTCRtpReceiver**负责接收和解码数据


**MediaPlayer**作为输入mp4文件的容器，从音频或视频文件中读取数据源，参数：file, format, options, timeout, 是否重复loop, decode
        player = MediaPlayer(args.play_from)

**MediaStreamTrack**媒体流，派生出AudioStreamTrack音频流和VideoStreamTrack视频流


**MediaRecorder**媒体接收流

**VideoFrame**：
参数：
    pts：用于表示视频帧什么时候被显式出来
    dts
    format
    height，width
    index
    key_frame
    pict_type:表示帧类型B/BI/I/P/S/SI/SP
    time_base：时间基，表示每个刻度是多少秒
### 编码器部分
基类：Encoder（base.py）
派生类：
  - Vp8Encoder
    - __target_bitrate
    - buffer
  - OpusEncoder
  - PcmaEncoder
  - PcmuEncoder
  - H264Encoder
    - buffer_data 
    - buffer_pts
    - codec
    - codec_buffering
    - __target_bitrate
### 1. 是否有pacer机制


