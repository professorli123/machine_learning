#!usr/bin/env python  
#-*- coding:utf-8 _*- 

# 图片像素 512*384
'''
图片增强程序，将图片按照一定的比例压缩裁剪
@data  :2021.1.3
@author:李雅军 李骐兆
'''
import random
import math
import torch

from PIL import Image, ImageOps, ImageFilter
from torchvision import transforms

class Resize(object):
    def __init__(self, size ,ratio):
        self.size = size
        self.ratio = ratio

    def __call__(self, img):    	
        image_w = int(self.size*self.ratio) 
        image_h = int(self.size*self.ratio) 
        img = img.resize((image_w,image_h),Image.BILINEAR)
        return img

class RandomRotate(object):
    def __init__(self, degree, p=0.5):
        self.degree = degree
        self.p = p

    def __call__(self, img):
        if random.random() < self.p:
            rotate_degree = random.uniform(-1*self.degree, self.degree)
            img = img.rotate(rotate_degree, Image.BILINEAR)
        return img

class RandomGaussianBlur(object):
    def __init__(self, p=0.5):
        self.p = p
    def __call__(self, img):
        if random.random() < self.p:
            img = img.filter(ImageFilter.GaussianBlur(
                radius=random.random()))
        return img

def get_train_transform(mean, std, size):
    train_transform = transforms.Compose([
        #Resize从左上角开始把不要的裁掉
        #RandomCrop随机的位置进行裁剪
        transforms.RandomCrop(size),
        #RandomHorizontalFlip以0.5的概率水平翻转给定的PIL图像
        #等比压缩一半
        Resize(size,0.4),
        # 图片通道转换
        transforms.ToTensor(),
        #用均值和标准差归一化PIL图片
        transforms.Normalize(mean=mean, std=std),
    ])
    return train_transform

def get_test_transform(mean, std, size):
    return transforms.Compose([
        #在图片的中间区域进行裁剪
        transforms.CenterCrop(size),
        #等比压缩一半
        Resize(size,0.4),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])

def get_transforms(input_size=224, test_size=224, backbone=None):
    mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    if backbone is not None and backbone in ['pnasnet5large', 'nasnetamobile']:
        mean, std = [0.5, 0.5, 0.5], [0.5, 0.5, 0.5]
    transformations = {}
    transformations['val_train'] = get_train_transform(mean, std, input_size)
    transformations['val_test'] = get_test_transform(mean, std, test_size)
    return transformations

