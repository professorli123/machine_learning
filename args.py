#!usr/bin/env python  
#-*- coding:utf-8 _*- 

'''
参数设置文件
@data  :2021.1.3
@author:李雅军 李骐兆
'''

import argparse
from build_net import model_names

# Parse arguments
parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')

# 导入数据的地址
# 训练集
parser.add_argument('-train', '--trainroot', default='data/new_shu_label.txt', type=str) #new_shu_label
# 测试集
parser.add_argument('-val', '--valroot', default='data/val1.txt', type=str)

# 这个应该是数据加载线程数  8
parser.add_argument('-j', '--workers', default=6, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
# Optimization options 优化选项
# 训练次数
parser.add_argument('--epochs', default=100, type=int, metavar='N',
                    help='number of total epochs to run')
# 图像的分类数
parser.add_argument('--num-classes', default=4, type=int, metavar='N',
                    help='number of classfication of image')
# 图像的大小                    
parser.add_argument('--image-size', default=384, type=int, metavar='N',
                    help='the train image size')
# 开始训练的次数 设置为0 ，设置为几就是从几开始
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
# 训练时一次输入多少的数据
parser.add_argument('--train-batch', default=10, type=int, metavar='N',
                    help='train batchsize (default: 256)')
# 检测时一次输入多少的数据
parser.add_argument('--test-batch', default=2, type=int, metavar='N',
                    help='test batchsize (default: 200)')
# 优化程序的选择
parser.add_argument('--optimizer', default='sgd',
                         choices=['sgd', 'rmsprop', 'adam', 'AdaBound', 'radam'], metavar='N',
                         help='optimizer (default=sgd)')
# 学习率  如果过大就不会收敛，如果过小则收敛速度太慢。                       
parser.add_argument('--lr', '--learning-rate', default=0.001, type=float,
                    metavar='LR', help='initial learning rate，1e-2， 1e-4, 0.001')
# 初始模型最后一层速率
parser.add_argument('--lr-fc-times', '--lft', default=5, type=int,
                    metavar='LR', help='initial model last layer rate')

parser.add_argument('--drop', '--dropout', default=0, type=float,
                    metavar='Dropout', help='Dropout ratio')
# 降低学习率
parser.add_argument('--schedule', type=int, nargs='+', default=[30, 50, 60],
                        help='Decrease learning rate at these epochs.')
# 学习率按照计划乘以gamma
parser.add_argument('--gamma', type=float, default=0.1, help='LR is multiplied by gamma on schedule.')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',help='momentum')
parser.add_argument('--no_nesterov', dest='nesterov',
                         action='store_false',
                         help='do not use Nesterov momentum')
parser.add_argument('--alpha', default=0.99, type=float, metavar='M',
                         help='alpha for ')
parser.add_argument('--beta1', default=0.9, type=float, metavar='M',
                         help='beta1 for Adam (default: 0.9)')
parser.add_argument('--beta2', default=0.999, type=float, metavar='M',
                         help='beta2 for Adam (default: 0.999)')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--final-lr', '--fl', default=1e-3,type=float,
                    metavar='W', help='weight decay (default: 1e-3)')
# 模型的保存路径和导入模型的路径
parser.add_argument('-c', '--checkpoint', default='./pth', type=str, metavar='PATH',#/res_16_288_last1
                    help='path to save checkpoint (default: checkpoint)')
parser.add_argument('--resume', default='./pth/checkpoint.pth.tar', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
# 网络模型
parser.add_argument('--arch', '-a', metavar='ARCH', default='resnext101_32x16d_wsl',
                    choices=model_names,
                    help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: resnext101_32x8d, pnasnet5large)')
# 神经网络的参数
# 这几个参数可以设置一下
parser.add_argument('--depth', type=int, default=50, help='Model depth.')
parser.add_argument('--cardinality', type=int, default=45, help='ResNet cardinality (group).')
parser.add_argument('--base-width', type=int, default=4, help='ResNet base width.')
parser.add_argument('--widen-factor', type=int, default=4, help='Widen factor. 4 -> 64, 8 -> 128, ...')

# 设置种子，保证网络初始化时是一致的
parser.add_argument('--manualSeed', type=int, help='manual seed')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')

# Device options
# 用几张显卡的问题，有的话最好是多用几张，3080ti啥的
parser.add_argument('--gpu-id', default='0', type=str,help='id(s) for CUDA_VISIBLE_DEVICES')

args = parser.parse_args()
