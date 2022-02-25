# -
最适合新手的入门学习项目，代码一看就懂
## 一、前言

  这次历时一个月飞桨训练营落下帷幕，而我一个0基础的纯小白在本次训练中也获得了许多深度学习的理论知识和基于paddlepaddle的实际操作，并完成了我的第一次训练模型。    
  该项目将通过图像分类的方法识别出一张手写的数字的分别的10个数字的概率，并取其中最大概率的数值为该模型的输出结果。希望能帮助想入门深度学习的朋友起到参考作用

## 二、数据介绍

该项目为经典的MINIST手写数字识别项目，MNIST数据集来自美国国家标准与技术研究所,训练集 (training set) 由来自 250 个不同人手写的数字构成, 其中 50% 是高中学生, 50% 来自人口普查局 (the Census Bureau) 的工作人员. 测试集(test set) 也是同样比例的手写数字数据.MNIST 数据集主要由一些手写数字的图片和相应的标签组成，图片一共有10 类，分别对应从0~9共10个阿拉伯数字。

![下载](C:\Users\86178\Desktop\下载.png)![下载 (1)](C:\Users\86178\Desktop\下载 (1).png)![下载 (2)](C:\Users\86178\Desktop\下载 (2).png)![下载 (3)](C:\Users\86178\Desktop\下载 (3).png)![下载 (4)](C:\Users\86178\Desktop\下载 (4).png)
  以上截取了数据集的一部分用于观察样本

## 三、模型介绍

该模型使用的是我自己搭建的双层全连接神经网络结构，结构简单、体积较小、方便训练和部署，这种简单的数据集采用该神经网络结构已经足以满足需求。![20191225150026841](C:\Users\86178\Desktop\20191225150026841.png)![output_10_1](https://user-images.githubusercontent.com/91590675/155743464-e3d63187-f946-4ff5-b829-466b37840dee.png)

以上为双层的全连接神经网络结构简单示意图    
损失函数为交叉熵损失函数，具体可参考链接

[基于PaddlePaddle2.0-构建SoftMax分类器](https://aistudio.baidu.com/aistudio/projectdetail/1323298)  

优化器为随机梯度下降算法
配置batch_size为1    
epoch为10   
学习率lr_schedule为0.001


## 四、数据集处理及模型训练


```python
# 首先解压我们的数据集
!unzip -oq /home/aistudio/data/data35499/minist数据集.zip
```


```python
# 导入所需要的工具包
import paddle
import numpy as np
from PIL import Image
import os
from matplotlib import pyplot as plt 
```


```python
# 将标签文件读取并放入字典中
root = "train_labs_black_background.txt"
f1 = open(root)
labels = f1.readlines()
label = {}
for i in range(55000-1):
    label[str(int(labels[i][0:-2]))] = int(labels[i][-2])
label["54999"] = 8
```


```python
# 预览图像
train_image_files = os.listdir("train_black_background")
print(train_image_files[6])
img = Image.open("train_black_background"+"/"+train_image_files[6])
print(img) # 图像大小为1*28*28
plt.imshow(img)
```

    22663.png
    <PIL.PngImagePlugin.PngImageFile image mode=L size=28x28 at 0x7FB335B7E890>





    <matplotlib.image.AxesImage at 0x7fb3356b10d0>




​    
![png](output_7_2.png)
​    



```python
# 搭建网络部分
class Net(paddle.nn.Layer):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = paddle.nn.Linear(in_features=1*28*28, out_features=128) # 输入28X28的图像点，输出结果为128个特征
        self.fc2 = paddle.nn.Linear(in_features=128, out_features=10) # 输入上层网络的结果，输出10个分类，分别为0~9的概率

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        return x
# 实例化
net = Net()
# 定义损失函数
loss_func = paddle.nn.CrossEntropyLoss() 
# 定义优化器
opt  = paddle.optimizer.SGD(parameters=net.parameters())
# 查看训练集
train_white_image_files = os.listdir("train_white_background")
train_black_image_files = os.listdir("train_black_background")
train_image_files = train_white_image_files+train_black_image_files
print(train_black_image_files[10:20])
```

    ['34700.png', '6362.png', '33233.png', '20471.png', '8877.png', '1909.png', '30648.png', '14400.png', '29952.png', '39000.png']



```python
# 组建训练程序
for epoch in range(10):
    for img_id in range(len(train_black_image_files)):
        img_name = f"{img_id}.png"
        _img = Image.open("train_black_background"+"/"+img_name)
        _img = np.array(_img).astype("float32").flatten() / 255
        _img = paddle.to_tensor([_img], dtype="float32")
        _label = label[str(img_id)]
        _label = paddle.to_tensor([_label], dtype="int64")

        number_prob = net(_img)
        loss = loss_func(number_prob, _label)
        loss.backward()
        opt.step()
        opt.clear_gradients()
    print(f"Epoch: {epoch}\t loss: {loss.numpy()}")

#保存模型为名为param的文件
paddle.save(net.state_dict(), "param")
```

    Epoch: 0	 loss: [0.22373171]
    Epoch: 1	 loss: [0.21978228]
    Epoch: 2	 loss: [0.23038626]
    Epoch: 3	 loss: [0.2418169]
    Epoch: 4	 loss: [0.2502424]
    Epoch: 5	 loss: [0.25516942]
    Epoch: 6	 loss: [0.25736782]
    Epoch: 7	 loss: [0.25783563]
    Epoch: 8	 loss: [0.25739065]
    Epoch: 9	 loss: [0.2565861]



```python
#可视化模型效果
net.set_dict(paddle.load("param"))
_img = Image.open("test_black_background"+"/"+"240.png")
plt.imshow(_img)
_img = np.array(_img).astype("float32").flatten() / 255
_img = paddle.to_tensor([_img], dtype="float32")
infer_number = net(_img)
print("240.png的预测结果为",paddle.nn.functional.softmax(infer_number).numpy()) # 使用激活函数使概率分布与0到1之间
print("最大概率为",np.argmax(infer_number.numpy()[0]))
```

    240.png的预测结果为 [[5.9594095e-05 1.5204851e-05 1.1775856e-04 5.2260727e-02 4.7045796e-05
      8.9238602e-01 1.6617357e-04 6.4850759e-08 5.4631677e-02 3.1577135e-04]]
    最大概率为 5



## 五、总结与升华

该项目成功的识别了测试集中数字，达到了目标，完成了我的第一个深度学习项目，但该项目只能识别单个数字，而且数据集的图片拍摄角度都是正方向的，后面不足的话预计就是能够识别多个数字，如银行卡号，车牌号等，在使用模型的接口可以加一个逆透视变换等处理方法，使模型能够识别出不同拍摄角度的图片

## 六、个人总结

我是机械电子工程的本科生，对深度学习有极大的兴趣，并且认为不管是机械还是电子如今都离不开人工智能，今后研究的方向应该会是CV，希望飞桨能够伴我前行   
下面是我的个人主页、感谢采纳  
https://aistudio.baidu.com/aistudio/usercenter
