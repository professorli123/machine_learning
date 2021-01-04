## 垃圾分类

### 1.代码结构

```c
 {repo_root}
  ├── data 		     //存放数据图片的文件夹
  ├── models	     //模型文件夹
  ├── page           //保存生成的图片的文件夹
  ├── pth            //保存生成的模型的文件夹
  ├── utils		     //一些函数包
  ├── args.py		 //参数配置文件
  ├── build_net.py	 //搭建模型
  ├── dataset.py	 //数据批量加载文件
  ├── preprocess.py	 //数据预处理文件，生成坐标标签
  ├── run.py         //训练测试环境搭建文件，数据预处理
  ├── train.py		 //训练运行文件
  ├── test.py        //测试运行
  ├── transform.py   //数据增强文件
```

### 2. 环境设置

安装指定的函数包，python版本为3.6，具体的函数包如下（提供了run.py 一键安装所有的包，pytorch根据自己的显卡安装合适的版本，本代码在CUDA10.1  NVIDIA 1060 6G 的环境进行测试，训练时间约为5小时，未出现过拟合的情况下测试集最高正确率95%）：

* pytorch        ==1.0.1
* torchvision  ==0.2.2
* matplotlib    ==3.1.0
* numpy          ==1.16.4
* scikit-image
* pandas
* sklearn
* adabound
* opencv-python

注：py3.7训练的话，要修改下面的代码

```python
if use_cuda:
	iputs, targets = inputs.cuda(), targets.cuda(async=True)
    inputs, targets = torch.autograd.Variable(inputs), torch.autograd.Variable(targets)

#python3.7已经移除了async关键字，而用non_blocking代替。(导致apache-airflow也出了问题)
#cuda() 本身也没有async.
```

就是把 async=True去掉

## 3.运行步骤

1.环境准备，创建虚拟环境，需要提前安装Ancoconda，在当前目录下，调出CMD命令行

```
conda env remove -n machine_learning_V_1.1.2

conda create -n machine_learning_V_1.1.2 python=3.6

conda activate machine_learning_V_1.1.2
```

2.安装环境内所需要的的包（统一写在了run.py）

3.将所有的图片放入'./data/garbage_classify/train_data/'这个路径下，不作任何处理

4.运行run.py

在当前目录下(在machine_learning_V_1.1.2虚拟环境内)执行命令

```
python run.py
```

返回complated！表明环境创建成功，数据预处理成功（90%的训练集，10%的测试集）

如果出现错误的话，可以根据以下命令手动安装上述包（一定注意版本问题）

```
conda install pytorch torchvision torchaudio cudatoolkit=10.1

pip install matplotlib

pip install scikit-image

pip install pandas

pip install numpy==1.16.4

pip install sklearn

pip install torchvision==0.2.2

pip install adabound

pip install opencv-python
```

5.多张显卡的话，修改arg.py 94行 parser.add_argument('--gpu-id', default='0'为'--gpu-id', default='0, 1, 2, 3' ，同时修改 '--train-batch'，'--test-batch'为适当的数字（因为显卡的限制，在我的电脑上只能为train-batch=10,test-batch=2，再高显卡显存不足），显卡允许的话可以同时将图像的大小调大。

6.运行train.py进行模型训练。（训练次数在args.py的第19行可调）

或者直接加入在命令中加入参数调节，默认100次

```
python train.py
```

7.运行test.py进行模型检测。

```
python test.py
```






