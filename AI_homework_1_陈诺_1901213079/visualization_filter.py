  
# coding: utf-8
import os
import torch
import torchvision.utils as vutils
from tensorboardX import SummaryWriter
import torch.nn as nn
import torch.nn.functional as F
import REsnet
from REsnet import ResNet,ResBlock

    

model_ft=torch.load('mres_net.pth')
# 防止模型加载时出现问题
if isinstance(model_ft,torch.nn.DataParallel):
    model_ft = model_ft.module


writer = SummaryWriter(log_dir=os.path.join(  "visual_weights"))
params = model_ft.state_dict()
i=0
for k, v in params.items():
   
    i+=1
    
    # 只输出第一个卷积核
    if(i==1):
        if 'conv' in k and 'weight' in k:

            # c_int = v.size()[1]     # 输入层通道数
            # print(c_int)
            c_out = v.size()[0]     # 输出层通道数
            print(c_out)

            # 以feature map为单位，绘制一组卷积核，一张feature map对应的卷积核个数为输入通道数
            # print(k+1,(v+1).size())
            for j in range(c_out):
                print(k, v.size(), j)
                kernel_j = v[j, :, :, :].unsqueeze(1) 
                print(kernel_j)      # 压缩维度，为make_grid制作输入
                kernel_grid = vutils.make_grid(kernel_j, normalize=True, scale_each=True, nrow=3)   # 1*输入通道数, w, h
                writer.add_image(k+'_split_in_channel', kernel_grid, global_step=j)     # j 表示feature map数
           
            # 将一个卷积层的卷积核绘制在一起，每一行是一个feature map的卷积核
            k_w, k_h = v.size()[-1], v.size()[-2]
          
            kernel_all = v.view(-1, 1, k_w, k_h)
           

            kernel_grid = vutils.make_grid(kernel_all, normalize=True, scale_each=True, nrow=3)  # 1*输入通道数, w, h
          
            
            writer.add_image(k + '_all', kernel_grid, global_step=64)
           

writer.close()