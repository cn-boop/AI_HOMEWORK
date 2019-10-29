import cv2
import time
import os
import matplotlib as mpl
mpl.use('Agg')
# import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import torch
from torch import nn
import torchvision.models as models
import torchvision.transforms as transforms
import numpy as np
import REsnet
import math
import os
import errno
import shutil
from REsnet import ResNet,ResBlock
 
savepath=r'features_cifar10'
if not os.path.exists(savepath):
    os.mkdir(savepath)
 
 
def draw_features(width,height,x,savename):
    tic=time.time()
    fig = plt.figure(figsize=(16, 16))
    fig.subplots_adjust(left=0.05, right=0.95, bottom=0.05, top=0.95, wspace=0.05, hspace=0.05)
    for i in range(width*height):
        plt.subplot(height,width, i + 1)
        plt.axis('off')
        img = x[0, i, :, :]
        pmin = np.min(img)
        pmax = np.max(img)
        img = ((img - pmin) / (pmax - pmin + 0.000001))*255  #float在[0，1]之间，转换成0-255
        img=img.astype(np.uint8)  #转成unit8
        img=cv2.applyColorMap(img, cv2.COLORMAP_JET) #生成heat map
        img = img[:, :, ::-1]#注意cv2（BGR）和matplotlib(RGB)通道是相反的
        plt.imshow(img)
        print("{}/{}".format(i,width*height))
    fig.savefig(savename, dpi=100)
    fig.clf()
    plt.show()
    # plt.close()
    print("time:{}".format(time.time()-tic))





 
class ft_net(nn.Module):
 
    def __init__(self):
        super(ft_net, self).__init__()
        # model_ft=REsnet.res_net
        model_ft=torch.load('mres_net.pth')
        if isinstance(model_ft,torch.nn.DataParallel):
    	    model_ft = model_ft.module
        # model_ft.eval()
        self.model = model_ft
 
    def forward(self, x):
        if True: # draw features or not
            x = self.model.conv1(x)
            draw_features(8,8,x.cpu().numpy(),"{}/f1_conv1.png".format(savepath))
   
 
            x = self.model.layer1(x)
            draw_features(8, 8, x.cpu().numpy(), "{}/f2_layer1.png".format(savepath))
           
 
            x = self.model.layer2(x)
            draw_features(8, 16, x.cpu().numpy(), "{}/f3_layer2.png".format(savepath))
 
            x = self.model.layer3(x)
            draw_features(16, 16, x.cpu().numpy(), "{}/f4_layer3.png".format(savepath))
 
            x = self.model.layer4(x)
            draw_features(16, 32, x.cpu().numpy(), "{}/f5_layer4_12-32.png".format(savepath))
         
 
            x = self.model.layer5(x)
           
            plt.clf()
            plt.close()
 
            x = x.view(x.size(0), -1)
            print(x.size(0))
         
            plt.clf()
            plt.close()
        else :
            x = self.model.conv1(x)
            x = self.model.bn1(x)
            x = self.model.relu(x)
            x = self.model.maxpool(x)
            x = self.model.layer1(x)
            x = self.model.layer2(x)
            x = self.model.layer3(x)
            x = self.model.layer4(x)
            x = self.model.avgpool(x)
            x = x.view(x.size(0), -1)
            x = self.model.fc(x)
 
        return x
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if __name__=='__main__':
    model=ft_net()
  
    model.eval()
    img=cv2.imread('img-5-automobile.png')
    img=cv2.resize(img,(224,224))
    img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    transform = transforms.Compose(
        [transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    img=transform(img)
    img=img.unsqueeze(0)
    
    with torch.no_grad():
        start=time.time()
        out=model(img.to(device))
        print("total time:{}".format(time.time()-start))
        result=out.cpu().numpy()
        # ind=np.argmax(out.cpu().numpy())
        ind=np.argsort(result,axis=1)
        for i in range(5):
            print("predict:top {} = cls {} : score {}".format(i+1,ind[0,512-i-1],result[0,512-i-1]))
        print("done")