发送端
cd aiortc
python3 ./examples/videostream-cli/cli.py offer --play-from ./dataset/test.mp4 -v -s tcp-socket --signaling-port 2222

接收端
cd aiortc
sleep 3 && python3 ./examples/videostream-cli/cli.py answer --record-to ./receive_data/video.mp4 -v -s tcp-socket --signaling-port 2222

