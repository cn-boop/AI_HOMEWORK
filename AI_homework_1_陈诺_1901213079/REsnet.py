import torch as t
from torch import nn
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
from torch import optim
import time
import visdom
import os
import cv2
from PIL import Image

# 超参数
batch_size=64
lr=1e-3
num_classes=10
epoch=100
last_batchsize=16
epochs=100
vis=visdom.Visdom()
global_step=0
# 创建保存文件
savepath=r'features_cifar10'
if not os.path.exists(savepath):
    os.mkdir(savepath)
# 对训练图像进行图像增强并进行标准化
train_transform=transforms.Compose([transforms.Resize(224),
transforms.RandomHorizontalFlip(),

transforms.ToTensor(),
transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
# 对测试图像进行中心化
test_transform = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),
     transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
#下载数据库
trainset = torchvision.datasets.CIFAR10(root='../data', train=True,
                                        download=True, transform=train_transform)
trainloader = t.utils.data.DataLoader(trainset, batch_size=batch_size,
                                          shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='../data', train=False,
                                       download=True, transform=test_transform)
testloader = t.utils.data.DataLoader(testset, batch_size=batch_size,
                                         shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
start_time=time.time()
def conv_kernel_3(in_channels,out_channels,stride=1):
    return nn.Conv2d(in_channels,out_channels,kernel_size=3,stride=stride,padding=1,bias=False)
class ResBlock(nn.Module):
    def __init__(self,in_channels,out_channels,stride=1):
        nn.Module.__init__(self)
        self.block=nn.Sequential(conv_kernel_3(in_channels,out_channels,stride),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
        conv_kernel_3(out_channels,out_channels),
        nn.BatchNorm2d(out_channels))
        self.relu=nn.ReLU(inplace=True)
        # 如果输入与输出大小维度不一致，需要将输入调整到与输出同一个纬度
        if stride != 1 or in_channels!=out_channels:
            self.modify_in_out = nn.Sequential(
            nn.Conv2d(in_channels,out_channels,kernel_size=1,stride=stride,bias=False),
            nn.BatchNorm2d(out_channels)
            )
        else:
            self.modify_in_out = nn.Sequential()
 
    def forward(self,x):
        block_out=self.block(x)
        block_out+=self.modify_in_out(x)
        block_out=self.relu(block_out)
        return block_out

class ResNet(nn.Module):
    def __init__(self,in_channels,out_channels,num_classes):
        nn.Module.__init__(self)
        self.conv1 = nn.Sequential(
            #in_channels=3,out_channels=64
            nn.Conv2d(in_channels, out_channels, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(batch_size),
            nn.ReLU()
            # nn.MaxPool2d()
        )
        self.layer1 = nn.Sequential(
            ResBlock(64, 64,stride=2),
            ResBlock(64, 64,stride=1),
            # nn.MaxPool2d(3)
        )
        
        self.layer2 = nn.Sequential(
            ResBlock(64, 128,stride=2),
            ResBlock(128, 128,stride=1),
            # nn.MaxPool2d(3)
        )
        
        self.layer3 = nn.Sequential(
           ResBlock(128, 256,stride=2),
           ResBlock(256, 256,stride=1)
        #    ,nn.MaxPool2d(3)
        )
        
        self.layer4 = nn.Sequential(
            ResBlock(256, 512,stride=2),
            ResBlock(512, 512,stride=1),
            # nn.MaxPool2d(4)
        )
        self.layer5=nn.Sequential(nn.AvgPool2d(7))
        self.classifier = nn.Linear(512, num_classes)
        
    def forward(self,x):
        
            res_out=self.conv1(x)
            # print('block 0 output: {}'.format(res_out.shape))

            res_out=self.layer1(res_out)
            # print('block 1 output: {}'.format(res_out.shape))
            res_out=self.layer2(res_out)
            # print('block 2 output: {}'.format(res_out.shape))
            res_out=self.layer3(res_out)
            # print('block 3 output: {}'.format(res_out.shape))
            res_out=self.layer4(res_out)
            # print('block 4 output: {}'.format(res_out.shape))
            res_out=self.layer5(res_out)
            # print('block 5 output: {}'.format(res_out.shape))
            res_out=res_out.view(res_out.size(0),-1)
            # print('block 6 output: {}'.format(res_out.shape))
            res_out=self.classifier(res_out)
            return res_out
#  定义模型
device = t.device("cuda" if t.cuda.is_available() else "cpu")
res_net = ResNet(3,64,num_classes).to(device)
if t.cuda.device_count() > 1:
      print('i can use gpu')
      res_net = nn.DataParallel(res_net,device_ids=[0,1,2])
criterion = nn.CrossEntropyLoss()
optimizer=optim.SGD(res_net.parameters(),lr=1e-3, momentum=0.9,weight_decay=1e-5)
#测试每个分类具体的正确率
def classes_accuracy(num_classes,batch_size,label):
    class_correct = list(0. for i in range(num_classes))  # 定义一个存储每类中测试正确的个数的 列表，初始化为0
    class_total = list(0. for i in range(num_classes))  # 定义一个存储每类中测试总数的个数的 列表，初始化为0
    for i,data in enumerate(testloader,0):  # 以一个batch为单位进行循环
        img, labels= data
      
        img = img.to(device)
        labels = labels.to(device) 
        outputs = res_net(img)
        _, predicted = t.max(outputs, 1)
        c = (predicted == labels).squeeze()
        # print(labels.size(0))
        # print(labels,i)
        if(i<156):
            for i in range(batch_size):  # 因为每个batch都有64张图片，所以需要一个循环
                label = labels[i]  # 对各个类的进行各自累加
                class_correct[label] += c[i]

                class_total[label] += 1
         # 最后一个batch只有16张图片，需要单独一个循环
        else:
            for i in range(last_batchsize):  
                label = labels[i]  # 对各个类的进行各自累加
                class_correct[label] += c[i]

                class_total[label] += 1
        # print(img)
        # print(class_correct)
    print(class_correct)
    # print(labels.size(0))
    for i in range(num_classes):
        class_10=classes[i]
        print(class_10,':', (class_correct[i] / class_total[i]).item())
    end_time=time.time()
    run_time=end_time-start_time
    print('Running Time:{} seconds '.format(run_time))

#训练模型
def train(trainloader,testloader,epoch,res_net=res_net):
    global_step=0
    acc_step=0
   
    for epoch_ in range(epochs):
        
        for i,train_data in enumerate(trainloader,0):
            img,label=train_data
            img=img.to(device)
            label=label.to(device)
            train_out=res_net(img)
            loss=criterion(train_out,label)
            # 梯度清零，更新梯度，反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch+=1
            if epoch % 100 == 0:
                print('epoch: {}, loss: {:.4}'.format(epoch, loss.data.item()))
                # 可视化LOSS
                vis.line(X=[global_step],Y=[loss.item()],win='loss',opts=dict(title = 'train loss'),update='append')
                global_step+=1
        # read_pricture() 
        # 
        # 
        # 
                eval_loss=0
                eval_acc=0
                res_net.eval()
                for i,test_data in enumerate(testloader,0):
                # 获得img(手写图片)，label标签（手写图片对应数字）
                    img,label = test_data

                # img = img.view(img.size(0), -1)
                # img = img.reshape(-1, 28, 28, 1)
                    
                    img = img.to(device)
                    label = label.to(device)
                #  向前传播，获得out结果和损失函数
                    test_out = res_net(img)
                    loss = criterion(test_out, label)
                    # total=0
                    # total+=label.size(0)
                # 损失函数乘标签大小累计
                    eval_loss += loss.data.item() * label.size(0)
                # 在10维数据中，获得最大的预测值（即预测数）
                    _, pred = t.max(test_out, 1)
                # 判断是否与真实结果相同
                    num_correct = (pred == label).sum()
                #累加正确结果
                    eval_acc += num_correct.item()
                # 可视化测试结果
                vis.line(X=[acc_step],Y=[(eval_acc / (len(testset)))],win='eval_acc / (len(testset)',opts=dict(title = 'acc'),update='append')
                acc_step+=1 
                print('Total Loss: {:.8f}, Acc: {:.8f}'.format(
                eval_loss / len(testset),
                eval_acc / len(testset)))
           
    classes_accuracy(num_classes=num_classes,batch_size=batch_size,label=label)
if __name__ == '__main__':
    train(trainloader,testloader,epoch)
    t.save(res_net, 'mres_net.pth')
    t.save(res_net.state_dict(), 'params.pth')
        


    

