import torch
from torch import nn
import torch.nn.functional as F
from modules.util import ResBlock2d, SameBlock2d, UpBlock2d, DownBlock2d
from modules.dense_motion import DenseMotionNetwork
from modules.util import AntiAliasInterpolation2d, make_coordinate_grid
from .GDN import GDN
import math
from modules.flowwarp import *


class OcclusionAwareGenerator(nn.Module):
    """
    Generator that given source image and and keypoints try to transform image according to movement trajectories
    induced by keypoints. Generator follows Johnson architecture.
    """

    def __init__(self, num_channels, num_kp, num_ref, block_expansion, max_features, num_down_blocks,
                 num_bottleneck_blocks, fusion_features, num_fusion_blocks, estimate_occlusion_map=False, dense_motion_params=None, estimate_jacobian=False):
        super(OcclusionAwareGenerator, self).__init__()

        self.temperature = 0.1
        self.num_ref = num_ref
        self.num_channels = num_channels
        self.estimate_occlusion_map = estimate_occlusion_map
       
        if dense_motion_params is not None:
            self.dense_motion_network = DenseMotionNetwork(num_kp=num_kp, num_channels=num_channels,
                                                           estimate_occlusion_map=estimate_occlusion_map,
                                                           **dense_motion_params)
        else:
            self.dense_motion_network = None

        self.first = SameBlock2d(num_channels, block_expansion, kernel_size=(7, 7), padding=(3, 3))

        down_blocks = []
        for i in range(num_down_blocks):
            in_features = min(max_features, block_expansion * (2 ** i))
            out_features = min(max_features, block_expansion * (2 ** (i + 1)))
            down_blocks.append(DownBlock2d(in_features, out_features, kernel_size=(3, 3), padding=(1, 1)))
        self.down_blocks = nn.ModuleList(down_blocks)

        up_blocks = []
        for i in range(num_down_blocks):
            in_features = min(max_features, block_expansion * (2 ** (num_down_blocks - i)))
            out_features = min(max_features, block_expansion * (2 ** (num_down_blocks - i - 1)))
            up_blocks.append(UpBlock2d(in_features, out_features, kernel_size=(3, 3), padding=(1, 1)))
        self.up_blocks = nn.ModuleList(up_blocks)

        self.bottleneck = torch.nn.Sequential()
        in_features = min(max_features, block_expansion * (2 ** num_down_blocks))
        for i in range(num_bottleneck_blocks):
            self.bottleneck.add_module('r' + str(i), ResBlock2d(in_features, kernel_size=(3, 3), padding=(1, 1)))
        
        self.final = nn.Conv2d(block_expansion, num_channels, kernel_size=(7, 7), padding=(3, 3))
        
        fusion_blocks = []
        fusion_blocks.append(SameBlock2d(num_channels*2, fusion_features, kernel_size=(7, 7), padding=(3, 3)))
        for i in range(num_fusion_blocks):
            fusion_blocks.append(ResBlock2d(fusion_features, kernel_size=(3, 3), padding=(1, 1)))        
        self.fusion_blocks = nn.ModuleList(fusion_blocks)
        
        self.fusion_final = nn.Conv2d(fusion_features, num_channels, kernel_size=(7, 7), padding=(3, 3))
        
        
    def deform_input(self, inp, deformation):
        _, h_old, w_old, _ = deformation.shape
        _, _, h, w = inp.shape
        if h_old != h or w_old != w:
            deformation = deformation.permute(0, 3, 1, 2)
            deformation = F.interpolate(deformation, size=(h, w), mode='bilinear')
            deformation = deformation.permute(0, 2, 3, 1)
        return warp(inp, deformation) #F.grid_sample(inp, deformation)  #########

    def forward(self, source_image, heatmap_source, heatmap_driving, source_image_more = None, heatmap_source_more = None):  
        # Downsampling
        out = self.first(source_image)
        for i in range(len(self.down_blocks)):
            out = self.down_blocks[i](out)
        
        # obtain the dense motion and occlusion map of source_image
        output_dict = {}
        dense_motion = self.dense_motion_network(source_image,heatmap_source,heatmap_driving)
        
        occlusion_map = dense_motion['occlusion_map']  #64*64*1
        output_dict['occlusion_map'] = occlusion_map  
        
        deformation = dense_motion['deformation'] #64*64*2 dense flow
        output_dict['deformation'] = deformation
        
        deformed_sparse_source = dense_motion['sparse_deformed'] #64*64*3
        output_dict['sparse_deformed'] = deformed_sparse_source
        
        sparse_motion = dense_motion['sparse_motion']  ###4*4*2
        output_dict['sparse_motion'] = sparse_motion

        out = self.deform_input(out, deformation)
        
        # obtain the dense motion and occlusion map of source_image_more
        if source_image_more is not None:
            out_more = self.first(source_image_more)
            for i in range(len(self.down_blocks)):
                out_more = self.down_blocks[i](out_more)
                
            dense_motion_more = self.dense_motion_network(source_image_more,heatmap_source_more,heatmap_driving)
            
            occlusion_map_more = dense_motion_more['occlusion_map']  #64*64*1
            output_dict['occlusion_map_more'] = occlusion_map_more  
        
            deformation_more = dense_motion_more['deformation'] #64*64*2
            output_dict['deformation_more'] = deformation_more
        
            deformed_sparse_source_more = dense_motion_more['sparse_deformed'] #64*64*3
            output_dict['sparse_deformed_more'] = deformed_sparse_source_more
        
            sparse_motion_more = dense_motion_more['sparse_motion']  ###4*4*2
            output_dict['sparse_motion_more'] = sparse_motion_more

            out_more = self.deform_input(out_more, deformation_more)
        
        # Output without occlusion map 
        out_dense = self.bottleneck(out)
        for i in range(len(self.up_blocks)):
            out_dense = self.up_blocks[i](out_dense)
        out_dense = self.final(out_dense)
        out_dense = torch.sigmoid(out_dense)        
        output_dict["deformed"] = out_dense  
        
        if source_image_more is not None:
            out_dense_more = self.bottleneck(out_more)
            for i in range(len(self.up_blocks)):
                out_dense_more = self.up_blocks[i](out_dense_more)
            out_dense_more = self.final(out_dense_more)
            out_dense_more = torch.sigmoid(out_dense_more)        
            output_dict["deformed_more"] = out_dense_more              

        # Output with occlusion map 
        if out.shape[2] != occlusion_map.shape[2] or out.shape[3] != occlusion_map.shape[3]:
            occlusion_map = F.interpolate(occlusion_map, size=out.shape[2:], mode='bilinear')
        out = out * occlusion_map

        out = self.bottleneck(out)
        for i in range(len(self.up_blocks)):
            out = self.up_blocks[i](out)
        out = self.final(out)
        out = torch.sigmoid(out)

        output_dict["prediction"] = out
        
        if source_image_more is not None:
            if out_more.shape[2] != occlusion_map_more.shape[2] or out_more.shape[3] != occlusion_map_more.shape[3]:
                occlusion_map_more = F.interpolate(occlusion_map_more, size=out_more.shape[2:], mode='bilinear')
            out_more = out_more * occlusion_map_more

            out_more = self.bottleneck(out_more)
            for i in range(len(self.up_blocks)):
                out_more = self.up_blocks[i](out_more)
            out_more = self.final(out_more)
            out_more = torch.sigmoid(out_more)

            output_dict["prediction_more"] = out_more   
            
            #### fusion module
            prediction_fusion = torch.cat([out, out_more], dim=1) 
            for i in range(len(self.fusion_blocks)):
                prediction_fusion = self.fusion_blocks[i](prediction_fusion)
            prediction_fusion = self.fusion_final(prediction_fusion) 
            prediction_fusion = torch.sigmoid(prediction_fusion)
            
            output_dict["prediction_fusion"] = prediction_fusion

        return output_dict
