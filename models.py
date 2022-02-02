import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F
from collections import OrderedDict


def getModels(cfg, device, checkpoint=None):
    net_g = Generator().to(device)
    net_d = Discriminator().to(device)
    if checkpoint != None:
        print("Start from epoch " + str(cfg.START_FROM_EPOCH))        
        net_g.load_state_dict(checkpoint['model_g'])        
        net_d.load_state_dict(checkpoint['model_d'])
        print("Model loaded")
    else:
        #net_g.apply(init_weight)
        #net_d.apply(init_weight)
        print("Start a new training from epoch 0")
    return net_g, net_d

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        # layer1输入的是一个100x1x1的随机噪声, 输出尺寸(ngf*8)x4x4
        self.ngf = 32
        """self.rot_linear_1 = nn.Linear(24*3, 64)
        self.rot_linear_2 = nn.Linear(64, 32)"""
        self.fc = nn.Linear(100, self.ngf*8*4*4)
        self.block0 = G_Block(self.ngf * 8, self.ngf * 8)
        self.block1 = G_Block(self.ngf * 8, self.ngf * 8)
        self.block2 = G_Block(self.ngf * 8, self.ngf * 8)
        self.block3 = G_Block(self.ngf * 8, self.ngf * 8)
        self.block4 = G_Block(self.ngf * 8, self.ngf * 4)
        self.block5 = G_Block(self.ngf * 4, self.ngf * 2)
        self.block6 = G_Block(self.ngf * 2, self.ngf * 1)


        self.conv_img = nn.Sequential(
            nn.LeakyReLU(0.2,inplace=True),
            nn.Conv2d(self.ngf, 3, 3, 1, 1),
            nn.Tanh(),
        )
        
    def forward(self, noise, sentence_vector, rot_vec):
        batch_s = sentence_vector.size(0)
        #out = self.fc(noise)
        #rot_vec = self.rot_linear_2(self.rot_linear_1(rot_vec.view(batch_s, 24*3)))
        c = torch.cat((sentence_vector, rot_vec.view(batch_s, 24*3)), 1)
        #c = sentence_vector
        out = self.fc(noise).view(batch_s, 8*self.ngf, 4, 4)

        out = self.block0(out,c) # 256 => 256
        out = F.interpolate(out, scale_factor=2) # 4x4 => 8x8
        out = self.block1(out,c) # 256 => 256
        out = F.interpolate(out, scale_factor=2) # 8x8 => 16x16
        out = self.block2(out,c) # 256 => 256
        out = F.interpolate(out, scale_factor=2) # 16x16 => 32x32
        out = self.block3(out,c) # 256 => 256
        out = F.interpolate(out, scale_factor=2) # 32x32 => 64x64
        out = self.block4(out,c) # 256 => 128
        out = F.interpolate(out, scale_factor=2) # 64x64 => 128x128
        out = self.block5(out,c) # 128 => 64
        out = F.interpolate(out, scale_factor=2) # 128x128 => 256x256
        out = self.block6(out,c) # 64 => 32

        out = self.conv_img(out)
        return out

class G_Block(nn.Module):

    def __init__(self, in_ch, out_ch):
        super(G_Block, self).__init__()

        self.learnable_sc = in_ch != out_ch 
        self.c1 = nn.Conv2d(in_ch, out_ch, 3, 1, 1)
        self.c2 = nn.Conv2d(out_ch, out_ch, 3, 1, 1)
        self.affine0 = affine(in_ch)
        self.affine1 = affine(in_ch)
        self.affine2 = affine(out_ch)
        self.affine3 = affine(out_ch)
        self.gamma = nn.Parameter(torch.zeros(1))
        if self.learnable_sc:
            self.c_sc = nn.Conv2d(in_ch,out_ch, 1, stride=1, padding=0)

    def forward(self, x, y=None):
        return self.shortcut(x) + self.gamma * self.residual(x, y)

    def shortcut(self, x):
        if self.learnable_sc:
            x = self.c_sc(x)
        return x

    def residual(self, x, y=None):
        h = self.affine0(x, y)
        h = nn.LeakyReLU(0.2,inplace=True)(h)
        h = self.affine1(h, y)
        h = nn.LeakyReLU(0.2,inplace=True)(h)
        h = self.c1(h)
        
        h = self.affine2(h, y)
        h = nn.LeakyReLU(0.2,inplace=True)(h)
        h = self.affine3(h, y)
        h = nn.LeakyReLU(0.2,inplace=True)(h)
        return self.c2(h)

class affine(nn.Module):

    def __init__(self, num_features):
        super(affine, self).__init__()

        self.fc_gamma = nn.Sequential(OrderedDict([
            ('linear1',nn.Linear(256+24*3, 256+24*3)),
            ('relu1',nn.ReLU(inplace=True)),
            ('linear2',nn.Linear(256+24*3, num_features)),
            ]))
        self.fc_beta = nn.Sequential(OrderedDict([
            ('linear1',nn.Linear(256+24*3, 256+24*3)),
            ('relu1',nn.ReLU(inplace=True)),
            ('linear2',nn.Linear(256+24*3, num_features)),
            ]))
        self._initialize()

    def _initialize(self):
        nn.init.zeros_(self.fc_gamma.linear2.weight.data)
        nn.init.ones_(self.fc_gamma.linear2.bias.data)
        nn.init.zeros_(self.fc_beta.linear2.weight.data)
        nn.init.zeros_(self.fc_beta.linear2.bias.data)

    def forward(self, x, y=None):

        weight = self.fc_gamma(y)
        bias = self.fc_beta(y)        

        if weight.dim() == 1:
            weight = weight.unsqueeze(0)
        if bias.dim() == 1:
            bias = bias.unsqueeze(0)

        size = x.size()
        weight = weight.unsqueeze(-1).unsqueeze(-1).expand(size)
        bias = bias.unsqueeze(-1).unsqueeze(-1).expand(size)
        return weight * x + bias	

class D_GET_LOGITS(nn.Module):
    def __init__(self, ndf):
        super(D_GET_LOGITS, self).__init__()
        self.df_dim = ndf
        """self.rot_linear_1 = nn.Linear(24*3, 64)
        self.rot_linear_2 = nn.Linear(64, 32)"""

        self.joint_conv = nn.Sequential(
            nn.Conv2d(ndf * 16+256+24*3, ndf * 2, 3, 1, 1, bias=False),
            nn.LeakyReLU(0.2,inplace=True),
            nn.Conv2d(ndf * 2, 1, 4, 1, 0, bias=False),
        )

    def forward(self, out, sentence_vector, rot_vec):
        batch_s = sentence_vector.size(0)
        #rot_vec = self.rot_linear_2(self.rot_linear_1(rot_vec.view(batch_s, 24*3)))
        tensor = torch.cat(
                            (sentence_vector.view(batch_s, 256, 1, 1).repeat(1, 1, 4, 4),rot_vec.view(batch_s, 24*3, 1, 1).repeat(1, 1, 4, 4)),
                           	1
                        )
        h_c_code = torch.cat((out, tensor), 1)
        out = self.joint_conv(h_c_code)
        return out

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.num_channels = 3
        self.ndf = 32
        self.conv_img = nn.Conv2d(3, self.ndf, 3, 1, 1)# 256,256,32
        self.netD_1 = nn.Sequential(
                resD(self.ndf, self.ndf * 2),# 128,128,64
                resD(self.ndf * 2, self.ndf * 4),# 64,64,128
                resD(self.ndf * 4, self.ndf * 8),# 32,32,256
                resD(self.ndf * 8, self.ndf * 16), # 16,16,512
                resD(self.ndf * 16, self.ndf * 16), # 8,8,512
                resD(self.ndf * 16, self.ndf * 16), # 4,4,512
            )
        self.COND_DNET = D_GET_LOGITS(self.ndf)
            
        
    def forward(self, input_img):
        output = self.conv_img(input_img)
        output = self.netD_1(output)  
        return output

class resD(nn.Module):
    def __init__(self, fin, fout, downsample=True):
        super().__init__()
        self.downsample = downsample
        self.learned_shortcut = (fin != fout)
        self.conv_r = nn.Sequential(
            nn.Conv2d(fin, fout, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(fout, fout, 3, 1, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
        )
        """self.conv_r = nn.Sequential(
            nn.Conv2d(fin, fout, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True)
        )"""

        self.conv_s = nn.Conv2d(fin,fout, 1, stride=1, padding=0)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x, c=None):
        return self.shortcut(x)+self.gamma*self.residual(x)

    def shortcut(self, x):
        if self.learned_shortcut:
            x = self.conv_s(x)
        if self.downsample:
            return F.avg_pool2d(x, 2)
        return x

    def residual(self, x):
        return self.conv_r(x)
