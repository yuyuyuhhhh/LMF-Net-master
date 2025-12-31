实验进度：

2025.8.11

main.py 

```
#from nets.lmf_net import LMFNet, LMFNetModel  # 这里确保 lmf_net.py 里真的有这个类 普通
from nets.lmf_net_more import LMFNet, LMFNetModel  # 这里确保 lmf_net.py 里真的有这个类 CMEA
```

通过from ==nets.lmf_net_more== import LMFNet, LMFNetModel  # 这里确保 lmf_net.py 里真的有这个类 CMEA已经测试了moon新图训练后的效果，

```
D:\VScodedate\1_MFIF\py\uncertion\LMF-Net-master\LMF-Net-master\nets\CMEA_moon\parameters\CMEA.pkl
```

![image-20250811112531595](assets/image-20250811112531595.png)

![image-20250811112658825](assets/image-20250811112658825.png)

计划先接着训练过渡区域清晰化。







```
from nets.lmf_net_CMEA import LMFNetModel  # 自定义网络模型
修改了
        self.se_f =ImprovedCMEA(16)
        # 后面三个块只用轻量 ECA
        self.se_1 =ImprovedCMEA(16)
        self.se_2 =ImprovedCMEA(16)
        self.se_3 =ImprovedCMEA(16)
 实验保存为train_net_CMEA.py下的experiment_name = 'CMEA_four_0812'  # 实验名称，用于保存日志和模型
```
