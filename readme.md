# 1. 端到端视频传输
## 运行
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
ea27adc3-ac6e-4b28-be9b-a27babed3e14.jpeg
