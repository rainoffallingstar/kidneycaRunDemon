### kidneyca classification


这是一个图像识别项目，基于 tensorflow，fork自原项目four_flouwers,现有的 CNN 网络可以识别两种肾肿瘤2dCT图像。适合新手对使用 tensorflow 进行一个完整的图像识别过程有一个大致轮廓。项目包括对数据集的处理，从硬盘读取数据，CNN 网络的定义，训练过程，还实现了一个 GUI 界面用于使用训练好的网络。

#### Require

1. 安装 Anaconda

2. 导入环境 environment.yaml  
   `conda env update -f=environment.yaml`
   
3. 如不想运行以上步骤则： pip3 install -r environment_cpu.txt  #cpu用户
                       pip3 install -r environment_gpu.txt  #gpu用户
#### Quick start

1. git clone 这个项目
2. 解压 input_data.rar 到你喜欢的目录。
3. 修改 train.py、input_data.py、test.py等文件 中

```
train_dir = 'D:/DL/kidneyca/inputdata'  # 训练样本的读入路径

logs_train_dir = 'D:/DL/kidneyca/save'  # logs存储路径
```

为你本机的目录。

4. 运行dataaugrcc16.py等文件获得数据增强24倍增益。
5. 运行 train.py 开始训练。
6. 训练完成后，修改 test.py 中的`logs_train_dir = 'D:/DL/kidneyca/save/'`为你的目录。
7. 运行 test.py 或者 gui.py 查看结果。
