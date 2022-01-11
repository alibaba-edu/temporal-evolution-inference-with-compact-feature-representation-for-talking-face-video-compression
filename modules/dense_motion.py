# -*- coding: utf-8 -*-
# + {}
from torch import nn
import torch.nn.functional as F
import torch
from modules.util import * 
import cv2
import numpy as np
from torch.autograd import Variable
import torchvision.transforms as transforms
from modules.vggloss import *
from modules.flowwarp import *

USE_CUDA = torch.cuda.is_available()
device = torch.device("cuda:0" if USE_CUDA else "cpu")


# -

class DenseMotionNetwork(nn.Module):
    """
    Module that predicting a dense motion from sparse motion representation given by kp_source and kp_driving
    """

    def __init__(self, block_expansion, num_blocks,num_down_blocks, max_features, num_kp, num_channels, num_bottleneck_blocks,
                 estimate_occlusion_map=False,scale_factor=1, kp_variance=0.01):
        
        super(DenseMotionNetwork, self).__init__()
        self.hourglass = Hourglass(block_expansion=block_expansion, in_features= num_channels + 1,
                                   max_features=max_features, num_blocks=num_blocks)

        self.flow = nn.Conv2d(self.hourglass.out_filters, 2, kernel_size=(7, 7), padding=(3, 3))
        
        if estimate_occlusion_map:
            self.occlusion = nn.Conv2d(self.hourglass.out_filters, 1, kernel_size=(7, 7), padding=(3, 3))
        else:
            self.occlusion = None

        self.num_kp = num_kp
        self.scale_factor = scale_factor
        self.kp_variance = kp_variance
        self.num_down_blocks=num_down_blocks

        # source_image down-sample
        if self.scale_factor != 1:
            self.down = AntiAliasInterpolation2d(num_channels, self.scale_factor)
        
        # keymap up-sample and obtain keymap difference    
        up_blocks = []
        for i in range(num_down_blocks):
            up_blocks.append(UpBlock2d(num_kp, num_kp, kernel_size=(3, 3), padding=(1, 1)))
        self.up_blocks = nn.ModuleList(up_blocks)
     
        # deform model        
        self.first = SameBlock2d(num_channels, block_expansion, kernel_size=(7, 7), padding=(3, 3))
        
        motiondown_blocks = []
        for i in range(num_down_blocks):
            in_features = min(max_features, block_expansion * (2 ** i))
            out_features = min(max_features, block_expansion * (2 ** (i + 1)))
            motiondown_blocks.append(DownBlock2d(in_features, out_features, kernel_size=(3, 3), padding=(1, 1)))
        self.motiondown_blocks = nn.ModuleList(motiondown_blocks)

        motionup_blocks = []
        for i in range(num_down_blocks):
            in_features = min(max_features, block_expansion * (2 ** (num_down_blocks - i)))
            out_features = min(max_features, block_expansion * (2 ** (num_down_blocks - i - 1)))
            motionup_blocks.append(UpBlock2d(in_features, out_features, kernel_size=(3, 3), padding=(1, 1)))
        self.motionup_blocks = nn.ModuleList(motionup_blocks)    

        self.bottleneck = torch.nn.Sequential()
        in_features = min(max_features, block_expansion * (2 ** num_down_blocks))
        for i in range(num_bottleneck_blocks):
            self.bottleneck.add_module('r' + str(i), ResBlock2d(in_features, kernel_size=(3, 3), padding=(1, 1)))        
        
        self.final = nn.Conv2d(block_expansion, num_channels, kernel_size=(7, 7), padding=(3, 3))

        
    def create_keymap_difference(self, source_image, heatmap_driving, heatmap_source):       
        bs, _, h, w = source_image.shape
        heatmap_d=heatmap_driving['value']
        heatmap_s=heatmap_source['value'] 
        
        for i in range(self.num_down_blocks):
            heatmap_d = self.up_blocks[i](heatmap_d)
            heatmap_s = self.up_blocks[i](heatmap_s)
            
        heatmap = heatmap_d  - heatmap_s
        return heatmap
    

       
    def create_sparse_motions(self, source_image, heatmap_driving, heatmap_source):
        ###Gunnar Farneback算法计算稠密光流      
        
        heatmap_source_lt = heatmap_source['value'] 
        heatmap_source_lt = heatmap_source_lt.cuda().data.cpu().numpy()         
        heatmap_source_lt = (heatmap_source_lt - np.min(heatmap_source_lt))/(np.max(heatmap_source_lt) - np.min(heatmap_source_lt)) *255.0  
        heatmap_source_lt=np.round(heatmap_source_lt)   
        heatmap_source_lt=heatmap_source_lt.astype(np.uint8)

        heatmap_driving_lt = heatmap_driving['value']
        heatmap_driving_lt = heatmap_driving_lt.cuda().data.cpu().numpy()         
        heatmap_driving_lt = (heatmap_driving_lt - np.min(heatmap_driving_lt))/(np.max(heatmap_driving_lt) - np.min(heatmap_driving_lt)) *255.0 
        heatmap_driving_lt=np.round(heatmap_driving_lt)   
        heatmap_driving_lt=heatmap_driving_lt.astype(np.uint8)       
        
        bs, _, h, w = source_image.shape  
        
        GFflow=[]
        for tensorchannel in range(0,bs):
        
            heatmap_source_lt11=heatmap_source_lt[tensorchannel].transpose([1,2,0])     
            heatmap_driving_lt11=heatmap_driving_lt[tensorchannel].transpose([1,2,0])      
            flow = cv2.calcOpticalFlowFarneback(heatmap_source_lt11,heatmap_driving_lt11,None, 0.5, 2, 15, 3, 5, 1.2, 0)            
            GFflow.append(flow)
            
        tmp_flow=torch.Tensor(np.array(GFflow)).to(device)         
        return tmp_flow    


    def create_deformed_source_image(self, source_image, sparse_motion):
        
        out = self.first(source_image)  
        
        for i in range(self.num_down_blocks):
            out = self.motiondown_blocks[i](out) 
        
        out=warp(out, sparse_motion)
        
        out = self.bottleneck(out)
        
        for i in range(self.num_down_blocks):
            out = self.motionup_blocks[i](out)

        out = self.final(out)  
        return out
    

    def forward(self, source_image,heatmap_source, heatmap_driving, source_image_more = None, heatmap_source_more = None):
        if self.scale_factor != 1:
            source_image = self.down(source_image)
            
        bs, c, h, w = source_image.shape   
        out_dict = dict()
        
        keymap_difference = self.create_keymap_difference(source_image, heatmap_driving, heatmap_source)  # [bs, 1, 4, 4] -> [bs, 1, 64, 64]
        sparse_motion = self.create_sparse_motions(source_image, heatmap_source, heatmap_driving)   #[bs, 4, 4, 2]
        deformed_source = self.create_deformed_source_image(source_image, sparse_motion)
        
        out_dict['sparse_motion'] = sparse_motion
        out_dict['sparse_deformed'] = deformed_source
        
        keymap_difference=keymap_difference.unsqueeze(1).view(bs,1, -1, h, w)
        deformed_source=deformed_source.unsqueeze(1).view(bs,1, -1, h, w)
        
        input =torch.cat([keymap_difference, deformed_source], dim=2)     # [bs, 4, 64, 64]
        input = input.view(bs, -1, h, w)   
        
        prediction = self.hourglass(input)          # [bs, block_expansion+4, 64, 64]

        deformation=self.flow(prediction)         
        deformation = deformation.permute(0, 2, 3, 1)     # [bs, 64, 64, 2]
        out_dict['deformation'] = deformation
        
        if self.occlusion:
            occlusion_map = torch.sigmoid(self.occlusion(prediction))   # [bs, 1, 64, 64] 
            out_dict['occlusion_map'] = occlusion_map        
        
        return out_dict
