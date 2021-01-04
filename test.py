#!usr/bin/env python  
#-*- coding:utf-8 _*- 

'''
测试文件，直接调用的train的接口进行测试
@data  :2021.1.3
@author:李雅军 李骐兆
'''
from __future__ import print_function
import os
import time
import random
import datetime
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.utils.data as data
import torchvision.transforms as transforms
import dataset
import numpy as np
from args import args
from build_net import make_model
from transform import get_transforms
import matplotlib.pyplot as plt
from utils import Bar, AverageMeter, accuracy, mkdir_p, savefig, get_optimizer, save_checkpoint

# 使用CUDA进行加速
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
use_cuda = torch.cuda.is_available()

args.evaluate = 1

def main():
    transform = get_transforms(input_size=args.image_size, test_size=args.image_size, backbone=None)
    valset = dataset.TestDataset(root='./data/val1.txt', transform=transform['val_test'])
    val_loader = data.DataLoader(valset, batch_size=args.test_batch, shuffle=False, num_workers=args.workers, pin_memory=True)

    # 创建网络模型
    model = make_model(args)

    # 选择优化模型
    if args.arch.startswith('alexnet') or args.arch.startswith('vgg'):
        model.features = torch.nn.DataParallel(model.features)
        model.cuda()
    else:
        model = torch.nn.DataParallel(model).cuda()

    # Benchmark模式会提升计算速度，但是由于计算中有随机性，每次网络前馈结果略有差异
    cudnn.benchmark = True
    print('    Total params: %.2fM' % (sum(p.numel() for p in model.parameters())/1000000.0))

    # 交叉熵损失
    # 交叉熵主要是用来判定实际的输出与期望的输出的接近程度
    criterion = nn.CrossEntropyLoss().cuda()
    optimizer = get_optimizer(model, args)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, patience=4, verbose=False)


    # 是否采用已经训练成功的模型进行测试
    if args.resume:
        # 导入模型
        print('==> Resuming from checkpoint..')
        # 如果导入失败，则抛出错误
        assert os.path.isfile(args.resume), 'Error: no checkpoint directory found!'
        args.checkpoint = os.path.dirname(args.resume)
        checkpoint = torch.load(args.resume)
        best_acc = checkpoint['best_acc']
        start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])

    # 进行测试
    if args.evaluate:
        print('\nEvaluation only')
        test_loss, test_acc , test_4 = test(val_loader, model, criterion, start_epoch, use_cuda)
        print(' Test Loss:  %.8f, Test Acc:  %.2f' % (test_loss, test_acc))
        return

def test(val_loader, model, criterion, epoch, use_cuda):

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # 转化为测试模式
    model.eval()

    end = time.time()
    bar = Bar('Processing', max=len(val_loader))
    for batch_idx, (inputs, targets) in enumerate(val_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        inputs, targets = torch.autograd.Variable(inputs), torch.autograd.Variable(targets)

        # 计算输出
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        # 测量精度和记录损耗
        prec1, prec4 = accuracy(outputs.data, targets.data, topk=(1, 4))
        losses.update(loss.item(), inputs.size(0))
        top1.update(prec1.item(), inputs.size(0))
        top5.update(prec4.item(), inputs.size(0))

        # 更新时间程序
        batch_time.update(time.time() - end)
        end = time.time()

        # 画图程序
        bar.suffix  = '({batch}/{size}) Data: {data:.3f}s | Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:} | Loss: {loss:.4f} | top1: {top1: .4f} | top5: {top5: .4f}'.format(
                    batch=batch_idx + 1,
                    size=len(val_loader),
                    data=data_time.avg,
                    bt=batch_time.avg,
                    total=bar.elapsed_td,
                    eta=bar.eta_td,
                    loss=losses.avg,
                    top1=top1.avg,
                    top5=top5.avg,
                    )
        bar.next()
    bar.finish()
    return (losses.avg, top1.avg, top5.avg)

if __name__ == "__main__":
    main()