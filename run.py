#!usr/bin/env python  
#-*- coding:utf-8 _*- 
'''
安装基础环境，初步处理数据的文件
若该问价安装所需要的的包安装失败，则需手动安装，请注意版本问题
将所有的图片统一解压至./data/garbage_classify/train_data/这个路径下，不在包含文件夹
@data  :2021.1.3
@author:李雅军 李骐兆
'''
import os
import time
from glob import glob


def env_pip():
    # 这里需要根据自身的显卡情况进行安装 我的显卡为CUDA10.1，故安装此版本
    # 具体参考https://pytorch.org/
    os.system('conda install pytorch torchvision torchaudio cudatoolkit=10.1') 
    # 安装所需要的库
    os.system('pip install matplotlib')
    os.system('pip install scikit-image')
    os.system('pip install pandas')
    os.system('pip install numpy==1.16.4')
    os.system('pip install sklearn')
    os.system('pip install torchvision==0.2.2')
    os.system('pip install adabound')
    os.system('pip install opencv-python')

# 进行数据的第一步处理，这里需要将所有的照片全部解压放在./data/garbage_classify/train_data/这个路径下
def data_device():
    data_path = './data/garbage_classify/train_data/'
    label_files = glob(os.path.join(data_path, '*.txt'))
    for filename in label_files:
        os.remove(filename)

    for filename in os.listdir(data_path):
        if filename[0] == 'g': 
            with open ('./data/garbage_classify/train_data/{}.txt'.format(filename),'w') as f:
                f.write(filename+', 0')
        if filename[0] == 'p' and filename[1] == 'l': 
            with open ('./data/garbage_classify/train_data/{}.txt'.format(filename),'w') as f:
                f.write(filename+', 1')
        if filename[0] == 'm': 
            with open ('./data/garbage_classify/train_data/{}.txt'.format(filename),'w') as f:
                f.write(filename+', 2')
        if filename[0] == 'p' and filename[1] == 'a': 
            with open ('./data/garbage_classify/train_data/{}.txt'.format(filename),'w') as f:
                f.write(filename+', 3')
    # 调用这个文件进行数据的第二部处理，将所有的图片路径抽取到一个文件内
    os.system('python preprocess.py')
    print('complated!')


# 返回当前显卡的占用情况
def CUDA_see():
    os.system("\"C:\\Program Files\\NVIDIA Corporation\\NVSMI\\nvidia-smi.exe\"")

# 执行者两个函数
env_pip()
data_device()