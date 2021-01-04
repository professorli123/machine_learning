#!usr/bin/env python  
#-*- coding:utf-8 _*- 
"""
数据预处理文件，生成坐标标签
@data  :2021.1.1
@author:李雅军 李骐兆
"""

from glob import glob
import os
import codecs
import random
import numpy as np
from sklearn.model_selection import KFold, StratifiedKFold

# 人为剔除掉数据里面的一些非常离谱的数据
bad_boy = [
    'meta191.jpg','glass1.jpg','glass48.jpg','glass74.jpg','glass76.jpg',
    'glass127.jpg','glass132.jpg','glass140.jpg','glass170.jpg','glass182.jpg',
    'glass186.jpg','glass190.jpg','glass192.jpg','glass195.jpg','glass199.jpg',
    'glass213.jpg','glass217.jpg','glass223.jpg','glass228.jpg''glass339.jpg',
    'glass444.jpg','glass449.jpg','glass494.jpg','glass495.jpg','plastic1.jpg',
    'plastic2.jpg','plastic3.jpg','plastic4.jpg','plastic6.jpg','plastic7.jpg',
    'plastic8.jpg','plastic13.jpg','plastic33.jpg','plastic35.jpg','plastic78.jpg',
    'plastic94.jpg','plastic107.jpg','plastic398.jpg','plastic426.jpg','metal75.jpg',
    'paper201.jpg','paper294.jpg','paper348.jpg','paper469.jpg','paper477.jpg',
    'paper486.jpg','paper590.jpg','paper591.jpg',
]
# print('我们人为剔除了{}个极其不靠谱的数据'.format(len(bad_boy)))
print('进行数据处理中。。。。')

base_path = 'data/'
data_path = base_path + 'garbage_classify/train_data'

label_files = glob(os.path.join(data_path, '*.txt'))
img_paths = []
labels = []
result = []
label_dict = {}
data_dict = {}

for index, file_path in enumerate(label_files):
    with codecs.open(file_path, 'r', 'utf-8') as f:
        line = f.readline()
    line_split = line.strip().split(', ')
    if len(line_split) != 2:
        print('%s contain error lable' % os.path.basename(file_path))
        continue
    img_name = line_split[0]
    if img_name in bad_boy:
        continue
    label = int(line_split[1])
    img_paths.append(os.path.join(data_path, img_name))
    labels.append(label)
    result.append(os.path.join(data_path, img_name) + ',' + str(label))
    label_dict[label] = label_dict.get(label, 0) + 1
    if label not in data_dict:
        data_dict[label] = []
    data_dict[label].append(os.path.join(data_path, img_name) + ',' + str(label))
    
folds = StratifiedKFold(n_splits=10, shuffle=True, random_state=2019)
for fold_, (trn_idx, val_idx) in enumerate(folds.split(result, labels)):
    train_data = list(np.array(result)[trn_idx])
    val_data = list(np.array(result)[val_idx])

with open(base_path + 'train1.txt', 'w') as f1:
    for item in train_data:
        f1.write(item + '\n')

with open(base_path + 'val1.txt', 'w') as f2:
    for item in val_data:
        f2.write(item + '\n')

all_data = []
train = []
val = []
# 将按照顺序的文件乱序后放入new_shu_label.txt
random.shuffle(all_data)
random.shuffle(train)
random.shuffle(val)
old = []

with open(base_path + 'train1.txt', 'r') as f:
    for i in f.readlines():
        old.append(i.strip())
for i in all_data:
    img_path, label = i.strip().split(',')

all_data.extend(old)
random.shuffle(all_data)

with open(base_path + 'new_shu_label.txt', 'w') as f1:
    for item in all_data:
        f1.write(item + '\n')