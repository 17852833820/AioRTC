

# METACC

这是一个基于AlphaRTC平台和Meta Reinforcement Learning的ABR算法

## 1. 环境配置

语言 python

环境：

​		python3.6/3.7

requirements：

​		torch>=1.10.1

​		torchvision>=0.11.2

​		numpy>=1.19.2

​		onnx=1.12.0

​		onnxruntime=1.12.0

​	



## 2. 安装部署说明

### 目录层级结构：

models: 本次测试需要的.onxx模型

solution: Meta ABR solution

### 测试说明：

1. Solution创建和运行

   ABR推理开始前创建solution，输入下列参数，其中"--"是可选参数。

```
#create solution
solution=MetaSolution(num_history_step,send_rate_range,is_meta_test,safeguard,checkpoint_path,record_history,use_different_reward,is_inference)
```

```
num_history_step:int
30
send_rate_range:float
2.0/0.5
is_meta_test:bool
False
safeguard:int(default:None)
checkpoint_path:str
"models/model1-init.onnx"
--record_history:bool(default:False)
False
--use_different_reward:bool(default:False)
True
--is_inference:bool(default:True)
True
```

2. 执行码率选择

```
#run solution
res=solution.cc_trigger(cur_time,instant_delay,instant_delay_jitter,instant_loss_rate,instant_rate_bps)
```

```
#input
cur_time:float
time:second
instant_delay: float
当前100ms内接收到的packet的平均包延时（second）
instant_delay_jitter：float
当前100ms内接收到的packet的平均delay jitter(second)
instant_loss_rate：float
当前100ms内的丢包率(%)
instant_rate_bps：float
当前100ms内接收到的平均速率，即平均每个packet的比特数(bps)
#output
res{
"target_bitrate":目标码率(Bps)
}
```



## 3. 运行

1. 保守策略选择

   - 以下列参数创建solution

   ```
   solution=MetaSolution(num_history_step=30,send_rate_range=2.0,is_meta_test=False,safeguard=100*1e3/8.0,checkpoint_path="models/model2.onnx",record_history=False,use_different_reward=True,is_inference=True)
   ```

   每间隔100ms运行一次策略选择，返回target bitrate

   ```
   res=solution.cc_trigger(cur_time,instant_delay,instant_delay_jitter,instant_loss_rate,instant_rate_bps)
   ```

   

2. 激进策略选择

   ```
   solution=MetaSolution(num_history_step=30,send_rate_range=0.5,is_meta_test=False,safeguard=70*1e3/8.0,checkpoint_path="models/model1.onnx",record_history=False,use_different_reward=True,is_inference=True)
   ```

   每间隔100ms运行一次策略选择，返回target bitrate

   ```
   res=solution.cc_trigger(cur_time,instant_delay,instant_delay_jitter,instant_loss_rate,instant_rate_bps)
   ```

   