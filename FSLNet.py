from __future__ import absolute_import, division, print_function
import numpy as np
import torch
import torch.nn as nn
from collections import OrderedDict
#from layers import *
import torchvision.models as models
import torch.utils.model_zoo as model_zoo
#from linformer import Linformer
from einops import rearrange, repeat
from timm.models.layers import trunc_normal_
import torch.nn.functional as F
import torch.fft as fft

padding_mode = 'reflect'
#reflect, replicate,,zeros
predim = 64

def upsample(x):
    """Upsample input tensor by a factor of 2
    """
    return F.interpolate(x, scale_factor=2, mode="nearest")
          
def _init_weights(m):
    if isinstance(m, nn.Linear):
        trunc_normal_(m.weight, std=.02)
        if isinstance(m, nn.Linear) and m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.LayerNorm):
        nn.init.constant_(m.weight, 1.0)
        if isinstance(m, nn.LayerNorm) and m.bias is not None:
            nn.init.constant_(m.bias, 0)  
    elif isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)         

def RGB2MP(imgs):
    fft_imgs = fft.rfftn(imgs, dim=(2,3), norm='ortho')
    r = torch.real(fft_imgs)
    i = torch.imag(fft_imgs)
    return torch.cat([r,i], dim=1)

def MP2Disp(MP_map):
    _, c, _, _ = MP_map.shape
    r = MP_map[:,0:c//2,:,:]  
    i = MP_map[:,c//2:,:,:]
    MP_map_complex = r + (1j*i)
    rimg = fft.irfftn(MP_map_complex, dim=(2,3), norm='ortho')
    return rimg
    
class WaveTransform(nn.Module):
    """Layer to perform wave transform
    """
    def __init__(self, in_channel, out_channel, h, w):
        super(WaveTransform, self).__init__()
        #self.weights = torch.nn.Parameter(torch.rand(1, 1, h, w) * 0.02)
        self.multiheads = nn.Conv2d(in_channel, out_channel, 1, bias=False, groups=1)

    def forward(self, x):
        #x = x * self.weights #linear learning
        x = self.multiheads(x)
        return x
    
class Act(nn.Module):
    def __init__(self):
        super(Act, self).__init__()
        self.act = nn.SiLU(True)
        
    def forward(self, x):
        return self.act(x)
    
class WaveTransformBlock(nn.Module):
    """Layer to perform wave transform
    """
    def __init__(self, in_channel, out_channel, h, w):
        super(WaveTransformBlock, self).__init__()
        self.wavet = WaveTransform(in_channel, out_channel, h, w)
        self.act = Act()
        self.bn = nn.BatchNorm2d(out_channel)

    def forward(self, x):
        x = self.wavet(x)
        x = self.act(x)
        x = self.bn(x)
        return x
    
class CNNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, k=3, stride=1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=k, padding=k//2, stride=stride, bias=False, padding_mode=padding_mode),
            Act(),
            nn.BatchNorm2d(out_channels)
        )
    def forward(self, x):
        return self.net(x)
    
class FLBlock(nn.Module):
    def __init__(self, in_channels, out_channels, h, w, mode='encoder', predim=predim):
        super().__init__()
        
        self.net_global = nn.Sequential(
            WaveTransformBlock(in_channels, out_channels, h, w),
            WaveTransformBlock(out_channels, out_channels, h, w),
        ) 
        
        if in_channels//2 <= predim and out_channels//2 <= predim:
            self.net_local = nn.Sequential(
                CNNBlock(in_channels//2, out_channels//2),
                CNNBlock(out_channels//2, out_channels//2),
                CNNBlock(out_channels//2, out_channels//2),
                CNNBlock(out_channels//2, out_channels//2),
                #CNNBlock(out_channels//2, out_channels//2)
            )
            
        if in_channels//2 > predim and out_channels//2 <= predim:
            self.net_local = nn.Sequential(
                CNNBlock(in_channels//2, predim, k=1),
                CNNBlock(predim, out_channels//2),
                CNNBlock(out_channels//2, out_channels//2),
                CNNBlock(out_channels//2, out_channels//2),
                CNNBlock(out_channels//2, out_channels//2),
                #CNNBlock(out_channels//2, out_channels//2)
            )  
            
        if in_channels//2 > predim and out_channels//2 > predim:
            self.net_local = nn.Sequential(
                CNNBlock(in_channels//2, predim, k=1),
                CNNBlock(predim, predim),
                CNNBlock(predim, predim),
                CNNBlock(predim, predim),
                CNNBlock(predim, predim),
                CNNBlock(predim, out_channels//2, k=1)
            )  
            
        if in_channels//2 <= predim and out_channels//2 > predim:
            self.net_local = nn.Sequential(
                CNNBlock(in_channels//2, predim),
                CNNBlock(predim, predim),
                CNNBlock(predim, predim),
                CNNBlock(predim, predim),
                CNNBlock(predim, out_channels//2, k=1)
            )   

        self.compression = nn.Sequential(
            CNNBlock(out_channels, out_channels//2),
        )
        
        self.mode = mode
        if mode == 'encoder':
            self.resulotionchange = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
    def forward(self, x):
        b, _, h, w = x.shape
        img = x
        
        x = RGB2MP(x)
        x = self.net_global(x)
        x = MP2Disp(x)
        if x.size(2) != h or x.size(3) != w:
            x = F.interpolate(x, [h, w], mode="bilinear", align_corners=False)  
         
        img = self.net_local(img)
        
        x = self.compression(torch.cat([x, img], dim=1))
        
        if self.mode == 'encoder':
            x = self.resulotionchange(x)
        if self.mode == 'decoder':
            x = upsample(x)
        
        return x
    
class FLEncoder(nn.Module):
    def __init__(self, num_img=1, model='L'):
        super().__init__()
        if model == 'L':
            self.channels = np.array([32, 64, 128, 256]) * 2
        else:
            self.channels = np.array([32, 64, 128, 256])
        self.encoder = nn.ModuleList()
        self.encoder.append(FLBlock(3*num_img*2, self.channels[0], 128, 209, mode='encoder')) #[64, 208] *
        self.encoder.append(FLBlock(self.channels[0], self.channels[1], 64, 105, mode='encoder')) #[32, 104] *
        self.encoder.append(FLBlock(self.channels[1], self.channels[2], 32, 53, mode='encoder')) #[16, 52] *
        self.encoder.append(FLBlock(self.channels[2], self.channels[3], 16, 27, mode='encoder')) #[8, 26] *
        
        
    def forward(self, x):
        features = []
        
        for i,layer in enumerate(self.encoder):
            x = layer(x)
            features.append(x)
            
        return features
    
class FLDepthDecoder(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.channels = channels
        self.decoder = nn.ModuleList()
        self.decoder.append(FLBlock(self.channels[3], self.channels[2], 8, 14, mode='decoder')) #[16, 52]
        self.decoder.append(FLBlock(self.channels[2], self.channels[1], 16, 27, mode='decoder')) #[32, 104]
        self.decoder.append(FLBlock(self.channels[1], self.channels[0], 32, 53, mode='decoder')) #[64, 208]
        self.decoder.append(FLBlock(self.channels[0], self.channels[0], 64, 105, mode='decoder')) #[128,416]
        
        self.fusion = nn.ModuleList()
        self.fusion.append(CNNBlock(self.channels[2], self.channels[2]//2))
        self.fusion.append(CNNBlock(self.channels[1], self.channels[1]//2))
        self.fusion.append(CNNBlock(self.channels[0], self.channels[0]//2))
        
        self.predict = nn.Sequential(
            CNNBlock(self.channels[0]//2, self.channels[0]//4),
            CNNBlock(self.channels[0]//4, self.channels[0]//8),
            nn.Conv2d(self.channels[0]//8, 1, 3, padding=1, stride=1, bias=False, padding_mode=padding_mode)
        )       
        
        self.sigmoid = nn.Sigmoid() 
        
    def forward(self, features):
        x = features[-1]
        self.outputs = {}
        
        for i,layer in enumerate(self.decoder):
            x = layer(x)
            
            if i == 0:
                x = torch.cat([x, features[-2]], dim=1)
                x = self.fusion[0](x)
            if i == 1:
                x = torch.cat([x, features[-3]], dim=1)
                x = self.fusion[1](x)
            if i == 2:
                x = torch.cat([x, features[-4]], dim=1)
                x = self.fusion[2](x)

        x = self.predict(x)
        x = self.sigmoid(x)
        
        self.outputs[("disp", 0)] = x
        
        return self.outputs
    
class FLPoseDecoder(nn.Module):
    def __init__(self, channels, num_img=3):
        super().__init__() 
        self.num_img = num_img
        
        self.predict = nn.Sequential(
            CNNBlock(channels[-1]//2, channels[-1]//4),
            CNNBlock(channels[-1]//4, channels[-1]//8),
            nn.Conv2d(channels[-1]//8, (num_img-1)*6, 3, padding=1, stride=1, bias=False, padding_mode=padding_mode)
        )
    def forward(self, x):
        x = self.predict(x)
        
        x = x.mean(3).mean(2)

        x = 0.01 * x.view(-1, self.num_img - 1, 1, 6)

        axisangle = x[..., :3]
        translation = x[..., 3:]

        return axisangle, translation        
    
class JointDepthNet(nn.Module):
    def __init__(self, model='L'):
        super().__init__()
        self.encoder = FLEncoder(model=model)
        self.decoder = FLDepthDecoder(self.encoder.channels) 
                    
    def forward(self,x):               
        x = self.encoder(x)
        x = self.decoder(x)
        return x
    
class JointPoseNet(nn.Module):
    def __init__(self, model='L'):
        super().__init__()
        self.encoder = FLEncoder(num_img=3, model=model)
        self.decoder = FLPoseDecoder(self.encoder.channels)
            
    def forward(self,x):
        x = self.encoder(x)
        axisangle, translation = self.decoder(x[-1])
        return axisangle, translation 
    