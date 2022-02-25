#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# 查看当前挂载的数据集目录, 该目录下的变更重启环境后会自动还原
# View dataset directory. 
# This directory will be recovered automatically after resetting environment. 
get_ipython().system('ls /home/aistudio/data')


# In[ ]:


# 查看工作区文件, 该目录下的变更将会持久保存. 请及时清理不必要的文件, 避免加载过慢.
# View personal work directory. 
# All changes under this directory will be kept even after reset. 
# Please clean unnecessary files in time to speed up environment loading. 
get_ipython().system('ls /home/aistudio/work')


# In[ ]:


# 如果需要进行持久化安装, 需要使用持久化路径, 如下方代码示例:
# If a persistence installation is required, 
# you need to use the persistence path as the following: 
get_ipython().system('mkdir /home/aistudio/external-libraries')
get_ipython().system('pip install beautifulsoup4 -t /home/aistudio/external-libraries')


# In[ ]:


# 同时添加如下代码, 这样每次环境(kernel)启动的时候只要运行下方代码即可: 
# Also add the following code, 
# so that every time the environment (kernel) starts, 
# just run the following code: 
import sys 
sys.path.append('/home/aistudio/external-libraries')


# # 解压数据集

# In[ ]:


get_ipython().system('unzip -oq /home/aistudio/data/data35499/minist数据集.zip')


# In[ ]:


import paddle
import numpy as np
from PIL import Image
import os
from matplotlib import pyplot as plt 


# In[ ]:


root = "train_labs_black_background.txt"
f1 = open(root)
labels = f1.readlines()
label = {}
for i in range(55000-1):
    label[str(int(labels[i][0:-2]))] = int(labels[i][-2])
label["54999"] = 8


# In[ ]:


# 预览图像
train_image_files = os.listdir("train_black_background")
print(train_image_files[6])
img = Image.open("train_black_background"+"/"+train_image_files[6])
print(img) # 1*28*28
plt.imshow(img)


# In[ ]:


# 构建网络
class Net(paddle.nn.Layer):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = paddle.nn.Linear(in_features=1*28*28, out_features=128)
        self.fc2 = paddle.nn.Linear(in_features=128, out_features=10)

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

train_white_image_files = os.listdir("train_white_background")
train_black_image_files = os.listdir("train_black_background")
train_image_files = train_white_image_files+train_black_image_files
print(train_black_image_files[10:20])


# 模型结构为全连接
# 
# 学习率lr_schedule为0.001
# 
# optimize为随机梯度下降优化器
# 
# epoch为10
# 
# 
# batch_size为1
# 
#  Loss function为交叉熵
#  

# In[7]:


#组件训练程序
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

#保存模型
paddle.save(net.state_dict(), "param")


# # 可视化模型效果

# In[9]:


# 可视化模型效果
net.set_dict(paddle.load("param"))
_img = Image.open("test_black_background"+"/"+"240.png")
plt.imshow(_img)
_img = np.array(_img).astype("float32").flatten() / 255
_img = paddle.to_tensor([_img], dtype="float32")
infer_number = net(_img)
print("240.png的预测结果为",paddle.nn.functional.softmax(infer_number).numpy()) # 使用激活函数使概率分布与0到1之间
print("最大概率为",np.argmax(infer_number.numpy()[0]))

