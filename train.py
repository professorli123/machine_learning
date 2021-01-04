#!usr/bin/env python  
#-*- coding:utf-8 _*- 

'''
训练神经网络的主函数
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
from ast import literal_eval
from utils import Bar, AverageMeter, accuracy, mkdir_p, savefig, get_optimizer, save_checkpoint

state = {k: v for k, v in args._get_kwargs()}

args.resume=''

# 这五个列表记录下来每一次训练的 损失值和准确率还有学习率
train_loss_list = []
test_loss_list = []
train_acc_list = []
test_acc_list = []
Lr_list = []

# 使用CUDA进行加速
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
use_cuda = torch.cuda.is_available()

# 设计随机初始化种子，保证每次神经网络初始化时的保持一致
if args.manualSeed is None:
    args.manualSeed = random.randint(1, 10000)
random.seed(args.manualSeed)
torch.manual_seed(args.manualSeed)
if use_cuda:
    torch.cuda.manual_seed_all(args.manualSeed)
# 将最佳的准确率置为0
best_acc = 0  

def main():
    # 将最佳准确率设置为全局参数
    global best_acc
    # 参数赋值，决定了是从0开始还是从上一次记录的神经网络训练次数值继续
    start_epoch = args.start_epoch  

    # 如果保存模型的路径不存在，那么就重新创建一个路径使之存在
    if not os.path.isdir(args.checkpoint):
        mkdir_p(args.checkpoint)
    # 创建日志文件，记录每一次训练的 损失值和准确率还有学习率
    with open ('./data/log.txt','w') as f:
        f.write('train_loss')
        f.write('\t')       
        f.write('test_loss')
        f.write('\t')  
        f.write('train_acc')
        f.write('\t')
        f.write('test_acc')
        f.write('\t') 
        f.write('LR')
        f.write('\t') 
        f.write('\n') 
    # 将输入和检测的图片进行等比压缩和裁剪的转化过程
    transform = get_transforms(input_size=args.image_size, test_size=args.image_size, backbone=None)

    # 导入训练文件
    print('==> Preparing dataset %s' % args.trainroot)
    trainset = dataset.Dataset(root=args.trainroot, transform=transform['val_train'])
    train_loader = data.DataLoader(trainset, batch_size=args.train_batch, shuffle=True, num_workers=args.workers, pin_memory=True)
    
    # 导入测试文件
    valset = dataset.TestDataset(root=args.valroot, transform=transform['val_test'])
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

    # 训练和测试
    for epoch in range(start_epoch, args.epochs):
        # 输出当前的训练次数和期望训练次数以及学习率
        print('\nEpoch: [%d | %d] LR: %f' % (epoch + 1, args.epochs, optimizer.param_groups[0]['lr']))

        # 训练 && 测试
        train_loss, train_acc, train_4 = train(train_loader, model, criterion, optimizer, epoch, use_cuda)
        test_loss, test_acc, test_4 = test(val_loader, model, criterion, epoch, use_cuda)

        scheduler.step(test_loss)
        print('train_loss:%f, val_loss:%f, train_acc:%f, train_4:%f, val_acc:%f, val_4:%f' % (train_loss, test_loss, train_acc, train_4, test_acc, test_4))
        # 将数据添加入列表 画图使用
        train_loss_list.append(train_loss)
        test_loss_list.append(test_loss)
        train_acc_list.append(train_acc)
        test_acc_list.append(test_acc)
        Lr_list.append(optimizer.param_groups[0]['lr'])
        # 将希望保存的数据写入log日志文件，防止丢失
        with open ('./data/log.txt','a') as f:
            f.write('%f'%(train_loss))
            f.write('\t')       
            f.write('%f'%(test_loss))
            f.write('\t')  
            f.write('%f'%(train_acc))
            f.write('\t')
            f.write('%f'%(test_acc))
            f.write('\t') 
            f.write('%f'%(optimizer.param_groups[0]['lr']))
            f.write('\t') 
            f.write('\n') 
            # 将上面保存数据的列表写入TXT文件，防止丢失
        with open ('./data/list.txt','w') as f:
            f.write(str(train_loss_list))
            f.write('|')
            f.write(str(test_loss_list))
            f.write('|')
            f.write(str(train_acc_list))
            f.write('|')
            f.write(str(test_acc_list))
            f.write('|')
            f.write(str(Lr_list))
        # 保存模型
        is_best = test_acc > best_acc
        best_acc = max(test_acc, best_acc)

        if len(args.gpu_id) > 1:
            save_checkpoint({
                'fold': 0,
                'epoch': epoch + 1,
                'state_dict': model.module.state_dict(),
                'train_acc': train_acc,
                'acc': test_acc,
                'best_acc': best_acc,
                'optimizer': optimizer.state_dict(),
            }, is_best, single=True, checkpoint=args.checkpoint)
        else:
            save_checkpoint({
                    'fold': 0,
                    'epoch': epoch + 1,
                    'state_dict': model.state_dict(),
                    'train_acc':train_acc,
                    'acc': test_acc,
                    'best_acc': best_acc,
                    'optimizer' : optimizer.state_dict(),
                }, is_best, single=True, checkpoint=args.checkpoint)

    # 输出最高的测试正确率
    print('Best acc:')
    print(best_acc)
    plot_list()

def train(train_loader, model, criterion, optimizer, epoch, use_cuda):
    # 转化到训练模式
    model.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    end = time.time()

    bar = Bar('Processing', max=len(train_loader))
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        # 实时加载训练时间
        data_time.update(time.time() - end)

        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda(async=True)
        inputs, targets = torch.autograd.Variable(inputs), torch.autograd.Variable(targets)

        # 计算输出
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        # 计算精度和记录损失
        prec1, prec4 = accuracy(outputs.data, targets.data, topk=(1, 3))
        losses.update(loss.item(), inputs.size(0))
        top1.update(prec1.item(), inputs.size(0))
        top5.update(prec4.item(), inputs.size(0))

        # 、、分类及其预测
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # 计算时间
        batch_time.update(time.time() - end)
        end = time.time()

        # 画图程序
        bar.suffix  = '({batch}/{size}) Data: {data:.3f}s | Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:} | Loss: {loss:.4f} | top1: {top1: .4f} | top5: {top5: .4f}'.format(
                    batch=batch_idx + 1,
                    size=len(train_loader),
                    data=data_time.val,
                    bt=batch_time.val,
                    total=bar.elapsed_td,
                    eta=bar.eta_td,
                    loss=losses.avg,
                    top1=top1.avg,
                    top5=top5.avg,
                    )
        bar.next()
    bar.finish()
    return (losses.avg, top1.avg, top5.avg)

def test(val_loader, model, criterion, epoch, use_cuda):
    global best_acc

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # 转化到测试模式
    model.eval()

    end = time.time()
    bar = Bar('Processing', max=len(val_loader))
    for batch_idx, (inputs, targets) in enumerate(val_loader):
        # 实时更新时间
        data_time.update(time.time() - end)

        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        inputs, targets = torch.autograd.Variable(inputs), torch.autograd.Variable(targets)

        # 计算输出
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        # 计算精度 记录损失
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

# 画图程序 
def plot_list():
    with open ('./data/list.txt','r') as f:
        s = f.read()
        x = range(1,len(literal_eval(s.split('|')[0]))+1)
        y1 = s.split('|')[0]
        y2 = s.split('|')[1]
        y3 = s.split('|')[2]
        y4 = s.split('|')[3]
        y5 = s.split('|')[4]
        y1 = literal_eval(y1)
        y2 = literal_eval(y2)
        y3 = literal_eval(y3)
        y4 = literal_eval(y4)
        y5 = literal_eval(y5)

    plt.title('train_loss')
    plt.plot(x,y1, 'r')
    plt.savefig('./page/train_loss.pdf')
    plt.show()

    plt.title('test_loss')
    plt.plot(x,y2, 'r')
    plt.savefig('./page/test_loss.pdf')
    plt.show()

    plt.title('train_acc')
    plt.plot(x,y3, 'r')
    plt.savefig('./page/train_acc.pdf')
    plt.show()

    plt.title('test_acc')
    plt.plot(x,y4, 'r')
    plt.savefig('./page/test_acc.pdf')
    plt.show()

    plt.title('Lr')
    plt.plot(x,y5, 'r')
    plt.savefig('./page/Lr.pdf')
    plt.show()

if __name__ == '__main__':
    time1 = datetime.datetime.now()
    main()
    time2 = datetime.datetime.now()
    time = time2 - time1
    print("训练用时：{}".format(time))
