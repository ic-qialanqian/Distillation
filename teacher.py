import torch
import numpy as np
import torch.nn.functional as F
from torch import nn
from torchvision import models
import torch.nn.init as init

from torch.nn import Conv2d, Parameter, Softmax


class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x

class RFB(nn.Module):
    # RFB-like multi-scale module
    def __init__(self, in_channel, out_channel):
        super(RFB, self).__init__()
        self.relu = nn.ReLU(True)
        self.branch0 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
        )
        self.branch1 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, kernel_size=(1, 3), padding=(0, 1)),
            BasicConv2d(out_channel, out_channel, kernel_size=(3, 1), padding=(1, 0)),
            BasicConv2d(out_channel, out_channel, 3, padding=3, dilation=3)
        )
        self.branch2 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, kernel_size=(1, 5), padding=(0, 2)),
            BasicConv2d(out_channel, out_channel, kernel_size=(5, 1), padding=(2, 0)),
            BasicConv2d(out_channel, out_channel, 3, padding=5, dilation=5)
        )

        self.conv_cat = BasicConv2d(3*out_channel, out_channel, 3, padding=1)
        self.conv_res = BasicConv2d(in_channel, out_channel, 1)


    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)

        x_cat = self.conv_cat(torch.cat((x0, x1, x2), 1))

        x = self.relu(x_cat + self.conv_res(x))
        return x
                
class RGBD_sal(nn.Module):
    def __init__(self):
        super(RGBD_sal, self).__init__()
        
        #
        
        feats = list(models.vgg19_bn(pretrained=True).features.children())
        self.conv0 = nn.Conv2d(4, 64, kernel_size=3, padding=1)
        self.conv1 = nn.Sequential(*feats[1:6])
        self.conv2 = nn.Sequential(*feats[6:13])
        self.conv3 = nn.Sequential(*feats[13:26])
        self.conv4 = nn.Sequential(*feats[26:39])
        self.conv5 = nn.Sequential(*feats[39:52])
        
        self.merge1 = nn.Conv2d(512, 512, kernel_size=1, padding=0)
        self.merge2 = nn.Conv2d(512, 256, kernel_size=1, padding=0)
        self.merge3 = nn.Conv2d(256, 128, kernel_size=1, padding=0)
        self.merge4 = nn.Conv2d(128, 64, kernel_size=1, padding=0)
        
        self.rfb5 = RFB(512,512)
        self.rfb4 = RFB(512,512)
        self.rfb3 = RFB(256,256)
        self.rfb2 = RFB(128,128)
        self.rfb1 = RFB(64,64)
        
        self.f1_ouput = nn.Conv2d(512, 1, kernel_size=1, padding=0)
        self.f2_ouput = nn.Conv2d(256, 1, kernel_size=1, padding=0)
        self.f3_ouput = nn.Conv2d(128, 1, kernel_size=1, padding=0)
        self.f4_ouput = nn.Conv2d(64, 1, kernel_size=1, padding=0)
        self.c5_overall_ouput=nn.Conv2d(512, 1, kernel_size=1, padding=0)
        #self.d5_overall_ouput=nn.Conv2d(512, 1, kernel_size=1, padding=0)
        
        
        
        self.output5 = nn.Sequential(nn.Conv2d(64, 2, kernel_size=3, padding=1))
        for m in self.modules():
            if isinstance(m, nn.ReLU) or isinstance(m, nn.Dropout):
                m.inplace = True

    def forward(self,x,depth):
        input = depth
        
        c0 = self.conv0(torch.cat((x,depth),1))
        c1 = self.conv1(c0)
        c2 = self.conv2(c1)
        c3 = self.conv3(c2)
        c4 = self.conv4(c3)
        c5 = self.conv5(c4)
        
        
        
        f0 = c5 
        f0 = self.rfb5(f0)
        
        f1 = c4  + F.upsample(self.merge1(f0),c4.size()[2:], mode='bilinear')
        f1 = self.rfb4(f1)
        
        f2 = c3  + F.upsample(self.merge2(f1),c3.size()[2:], mode='bilinear')
        f2 = self.rfb3(f2)
        
        f3 = c2  + F.upsample(self.merge3(f2),c2.size()[2:], mode='bilinear')
        f3 = self.rfb2(f3)
        
        f4 = c1  + F.upsample(self.merge4(f3),c1.size()[2:], mode='bilinear')
        f4 = self.rfb1(f4)
        
        f1_attention =self.f1_ouput(f1)
        f2_attention =self.f2_ouput(f2)
        f3_attention =self.f3_ouput(f3)
        f4_attention =self.f4_ouput(f4)
        c5_overall_ouput = self.c5_overall_ouput(f0)
        
        
        output_final = F.upsample(self.output5(f4), size=depth.size()[2:], mode='bilinear')
        
        if self.training:
            return output_final,c5_overall_ouput,f1_attention,f2_attention,f3_attention,f4_attention
            #return output_final
        return output_final


if __name__ == "__main__":
    model = RGBD_sal()
    model.cuda()
    input = torch.autograd.Variable(torch.zeros(4, 3, 256, 256)).cuda()
    depth = torch.autograd.Variable(torch.zeros(4, 1, 256, 256)).cuda()
    output = model(input, depth)
    print(output.size())
