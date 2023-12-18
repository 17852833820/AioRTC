发送端
cd aiortc
python3 ./examples/videostream-cli/cli.py offer --play-from ./dataset/anamal-time.mp4 -v -s tcp-socket --signaling-port 2223

接收端
cd aiortc
sleep 3 && python3 ./examples/videostream-cli/cli.py answer --record-to ./receive_data/video1.mp4 -v -s tcp-socket --signaling-port 2223

