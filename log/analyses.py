import re
from datetime import datetime

import matplotlib.pyplot as plt

jit_line_pattern = re.compile(r'(?P<timestamp>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}) - DEBUG - RTCRtpReceiver\(video\) \[FRAME_INFO\] T: \d+, jit_dur: (?P<jit_dur>\d+), Bytes: (?P<bytes>\d+)')
dec_line_pattern = re.compile(r'(?P<timestamp>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}) - DEBUG - RTCRtpReceiver\(decoder_worker\) \[FRAME_INFO\] T: \d+, dec_dur: (?P<dec_dur>\d+), Bytes: \d+')
enc_line_pattern = re.compile(r'(?P<timestamp>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}) - DEBUG - RTCRtpSender\(video\) \[FRAME_INFO\] Number: (?P<time>\d+), PTS: \d+, enc_dur: (?P<enc_dur>\d+)')
trans_line_pattern = re.compile(r'(?P<timestamp>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}) - DEBUG - RTCRtpReceiver\(video\) \[FRAME_INFO\]  transport dur: (?P<trans_dur>\d+)')
rtt_line_pattern = re.compile(r'(?P<timestamp>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}) - DEBUG - RTCRtpSender\(video\) \[FRAME_INFO\] RTT: (?P<rtt>\d+)')
size_line_pattern = re.compile(r'(?P<timestamp>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}) - INFO - Encodec \| Frame  Type: NONE ,size: (?P<size>\d+)')
# 用于存储 jit_dur 的列表2023-12-10 21:51:11 - INFO - Encodec | Frame  Type: NONE ,size: 1836
jit_dur_values = []
dec_dur_values = []
enc_dur_values = []
trans_dur_values = []
rtt_dur_values = []
size_dur_values = []#KB
# 读取 log 文件

log_file_path = 'log/answer/test-rtt.log'
with open(log_file_path, 'r') as file:
    for line in file:
        line = line.strip()
        jit_match = jit_line_pattern.search(line)
        dec_match= dec_line_pattern.search(line)
        trans_match= trans_line_pattern.search(line)

        if jit_match:
            timestamp_str = jit_match.group('timestamp')
            jit_dur_str = jit_match.group('jit_dur')
            bytes_frame=jit_match.group('bytes')
            # 解析时间戳字符串为 datetime 对象
            timestamp = datetime.strptime(timestamp_str, '%Y-%m-%d %H:%M:%S')

            # 将时间戳和 jit_dur 添加到列表
            jit_dur_values.append({
                'timestamp': timestamp,
                'jit_dur': int(jit_dur_str),
                'bytes_frame': int(bytes_frame),
            })
        if dec_match:
                timestamp_str = dec_match.group('timestamp')
                dec_dur_str = dec_match.group('dec_dur')
                # 解析时间戳字符串为 datetime 对象
                timestamp = datetime.strptime(timestamp_str, '%Y-%m-%d %H:%M:%S')

                # 将时间戳和 jit_dur 添加到列表
                dec_dur_values.append({
                    'timestamp': timestamp,
                    'dec_dur': int(dec_dur_str),
                })
           
        if trans_match:
            trans_dur_str = trans_match.group('trans_dur')
            # print(trans_dur_str)
            trans_dur_values.append({"trans_dur":int(trans_dur_str)})
log_file_path2 = 'log/offer/test-rtt.log'
with open(log_file_path2, 'r') as file:
    for line in file:
        line = line.strip()
        enc_match= enc_line_pattern.search(line)
        rtt_match= rtt_line_pattern.search(line)
        size_match= size_line_pattern.search(line)
        if enc_match:
                timestamp_str = enc_match.group('time')
                enc_dur_str = enc_match.group('enc_dur')
                # 解析时间戳字符串为 datetime 对象
                # timestamp = datetime.strptime(timestamp_str, '%Y-%m-%d %H:%M:%S')
                # 将时间戳和 jit_dur 添加到列表
                enc_dur_values.append({
                    'timestamp': timestamp_str,
                    'enc_dur': int(enc_dur_str),
                })
        if rtt_match:
            rtt_dur_str=rtt_match.group("rtt")
            rtt_dur_values.append({
                "rtt":float(rtt_dur_str)
            })
        if size_match:
            size_dur_str=size_match.group("size")
            size_dur_values.append({
                "size":float(float(size_dur_str)/1024.0)
            })
# 提取时间戳和 jit_dur 列表
# timestamps1 = [entry['timestamp'] for entry in jit_dur_values]
jit_durs = [entry['jit_dur'] for entry in jit_dur_values]
# bytes_frame = [entry['bytes_frame'] for entry in jit_dur_values]

# timestamps2 = [entry['timestamp'] for entry in dec_dur_values]
dec_durs = [entry['dec_dur'] for entry in dec_dur_values]
# timestamps3 = [float(entry['timestamp'])/1000.0 for entry in enc_dur_values]
enc_durs = [entry['enc_dur'] for entry in enc_dur_values]
trans_durs = [entry['trans_dur'] for entry in trans_dur_values]
rtt_durs=[enrty["rtt"] for enrty in rtt_dur_values]
size_durs=[entry["size"] for entry in size_dur_values]
enc_durs = enc_durs[:len(dec_durs)]
# 画图
plt.figure(figsize=(8, 6))

plt.plot(trans_durs, label='TRANS Duration')

plt.xlabel('Timestamp')
plt.ylabel('Duration (ms)')
plt.title(' Duration Over Time')
plt.legend()
plt.show()
plt.savefig('duration_TRANS.png')

plt.figure(figsize=(8, 6))
plt.plot( jit_durs, label='JIT Duration')
plt.plot(dec_durs, label='DEC Duration')
plt.plot(enc_durs, label='ENC Duration')
plt.xlabel('Timestamp')
plt.ylabel('Duration (ms)')
plt.title(' Duration Over Time')
plt.legend()
plt.show()
plt.savefig('duration_DEC.png')

plt.figure(figsize=(8, 6))
plt.plot(rtt_durs, label='RTT')
plt.xlabel('Timestamp')
plt.ylabel('Duration (ms)')
plt.title(' Duration Over Time')
plt.legend()
plt.show()
plt.figure(figsize=(8, 6))
plt.plot(size_durs, label='RTT')
plt.xlabel('Frame')
plt.ylabel('Size (KB)')
plt.title(' Duration Over Time')
plt.legend()
plt.show()
plt.savefig('FrameSize.png')
# plt.figure(figsize=(8, 6))
# plt.plot(timestamps1, bytes_frame, label='bytes')
# plt.xlabel('Timestamp')
# plt.ylabel('Duration (ms)')
# plt.title('Frame size')
# plt.legend()
# plt.show()
# plt.savefig('bytes.png')
