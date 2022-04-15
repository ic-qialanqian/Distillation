import torch
import numpy as np
import torch.nn.functional as F
from torch import nn
from torchvision import models
import torch.nn.init as init
from torch.nn import Conv2d, Parameter, Softmax


                
class RGBD_sal1(nn.Module):
    def __init__(self):
        super(RGBD_sal1, self).__init__()
        
        #
        '''
        feats = list(models.vgg16_bn(pretrained=True).features.children())
        self.conv0 = nn.Conv2d(4, 64, kernel_size=3, padding=1)
        self.conv1 = nn.Sequential(*feats[1:6])
        self.conv2 = nn.Sequential(*feats[6:13])
        self.conv3 = nn.Sequential(*feats[13:23])
        self.conv4 = nn.Sequential(*feats[23:33])
        self.conv5 = nn.Sequential(*feats[33:43])
        
        '''
        feats = list(models.vgg19_bn(pretrained=True).features.children())
        self.conv0 = nn.Conv2d(4, 64, kernel_size=3, padding=1)
        self.conv1 = nn.Sequential(*feats[1:6])
        self.conv2 = nn.Sequential(*feats[6:13])
        self.conv3 = nn.Sequential(*feats[13:26])
        self.conv4 = nn.Sequential(*feats[26:39])
        self.conv5 = nn.Sequential(*feats[39:52])
        
        
        self.merge1 = nn.Sequential(nn.Conv2d(512, 512, kernel_size=1, padding=0), nn.PReLU())
        self.merge2 = nn.Sequential(nn.Conv2d(512, 256, kernel_size=1, padding=0), nn.PReLU())
        self.merge3 = nn.Sequential(nn.Conv2d(256, 128, kernel_size=1, padding=0), nn.PReLU())
        self.merge4 = nn.Sequential(nn.Conv2d(128, 64, kernel_size=1, padding=0), nn.PReLU())
        
    
        
      
     
        
        
        self.output5 = nn.Sequential(nn.Conv2d(64, 2, kernel_size=3, padding=1), nn.PReLU())
      

        
        for m in self.modules():
            if isinstance(m, nn.ReLU) or isinstance(m, nn.Dropout):
                m.inplace = True

    def forward(self, x,depth):
        input = x
        
        c0 = self.conv0(torch.cat((x,depth),1))
        #c0 = self.conv0(x)
        c1 = self.conv1(c0)
        c2 = self.conv2(c1)
        c3 = self.conv3(c2)
        c4 = self.conv4(c3)
        c5 = self.conv5(c4)
        

        
        f0 = c5 
        
        
        f1 = c4  + F.upsample(self.merge1(f0),c4.size()[2:], mode='bilinear')

        
        f2 = c3  + F.upsample(self.merge2(f1),c3.size()[2:], mode='bilinear')

        
        f3 = c2  + F.upsample(self.merge3(f2),c2.size()[2:], mode='bilinear')

        
        f4 = c1  + F.upsample(self.merge4(f3),c1.size()[2:], mode='bilinear')
        
        
        
        output_final = F.upsample(self.output5(f4), size=x.size()[2:], mode='bilinear')
        
        
        if self.training:
            
            return output_final
        return output_final


if __name__ == "__main__":
    model = RGBD_sal1()
    model.cuda()
    input = torch.autograd.Variable(torch.zeros(4, 3, 256, 256)).cuda()
    depth = torch.autograd.Variable(torch.zeros(4, 1, 256, 256)).cuda()
    output = model(input, depth)
    print(output.size())
