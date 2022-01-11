# -*- coding: utf-8 -*-
from torch import nn
import torch
import torch.nn.functional as F
from modules.util import *
import numpy as np
from torch.autograd import grad
from .GDN import GDN
import math
from modules.vggloss import *
import numpy as np


class GeneratorFullModel(torch.nn.Module):
    """
    Merge all generator related updates into single model for better multi-gpu usage
    """

    def __init__(self, kp_extractor, generator, discriminator, train_params, common_params):
        super(GeneratorFullModel, self).__init__()
        self.kp_extractor = kp_extractor
        self.generator = generator
        self.discriminator = discriminator
        self.train_params = train_params
        self.scale_factor = train_params['scale_factor']
        self.scales = train_params['scales']
        self.temperature =train_params['temperature']
        self.out_channels =common_params['num_kp'] 
        self.num_ref = common_params['num_ref']
        self.disc_scales = self.discriminator.scales
        
        self.down = AntiAliasInterpolation2d(generator.num_channels, self.scale_factor)    
            
        self.pyramid = ImagePyramide(self.scales, generator.num_channels)
        if torch.cuda.is_available():
            self.pyramid = self.pyramid.cuda()

        self.loss_weights = train_params['loss_weights']

        self.vgg = Vgg19()
        if torch.cuda.is_available():
            self.vgg = self.vgg.cuda()
        
    def forward(self, x):
        
        heatmap_source = self.kp_extractor(x['source'])
        heatmap_driving = self.kp_extractor(x['driving'])
        if self.num_ref > 1:
            heatmap_source_more = self.kp_extractor(x['source_more'])
            generated = self.generator(x['source'],heatmap_source,heatmap_driving,x['source_more'],heatmap_source_more)
            generated.update({'heatmap_source':heatmap_source,'heatmap_driving':heatmap_driving,'heatmap_source_more':heatmap_source_more}) 
        else:  
            generated = self.generator(x['source'],heatmap_source,heatmap_driving)
            generated.update({'heatmap_source':heatmap_source,'heatmap_driving':heatmap_driving}) 
            
        loss_values = {}

        pyramide_real = self.pyramid(x['driving']) 
        pyramide_generated = self.pyramid(generated['prediction'])
        

        driving_image_downsample = self.down(x['driving'])    ### [3,64,64]   
        pyramide_real_downsample = self.pyramid(driving_image_downsample) 
        sparse_deformed_generated=generated['sparse_deformed']  ### [3,64,64]
        sparse_pyramide_generated = self.pyramid(sparse_deformed_generated)      
        
        ### Perceptual Loss---Initial
        if sum(self.loss_weights['perceptual_initial']) != 0:
            value_total = 0
            for scale in [1, 0.5, 0.25]:
                x_vgg = self.vgg(sparse_pyramide_generated['prediction_' + str(scale)])
                y_vgg = self.vgg(pyramide_real_downsample['prediction_' + str(scale)])

                for i, weight in enumerate(self.loss_weights['perceptual_initial']):
                    value = torch.abs(x_vgg[i] - y_vgg[i].detach()).mean()
                    value_total += self.loss_weights['perceptual_initial'][i] * value
            
            loss_values['initial'] = value_total   
        
            if self.num_ref > 1:
                sparse_deformed_generated_more=generated['sparse_deformed_more']  ### [3,64,64]
                sparse_pyramide_generated_more = self.pyramid(sparse_deformed_generated_more)           

                value_total = 0
                for scale in [1, 0.5, 0.25]:
                    x_vgg = self.vgg(sparse_pyramide_generated_more['prediction_' + str(scale)])
                    y_vgg = self.vgg(pyramide_real_downsample['prediction_' + str(scale)])

                    for i, weight in enumerate(self.loss_weights['perceptual_initial']):
                        value = torch.abs(x_vgg[i] - y_vgg[i].detach()).mean()
                        value_total += self.loss_weights['perceptual_initial'][i] * value
                
                loss_values['initial'] += value_total 
                
        
        ### Perceptual Loss---Final
        if sum(self.loss_weights['perceptual_final']) != 0:
            value_total = 0
            for scale in self.scales:
                x_vgg = self.vgg(pyramide_generated['prediction_' + str(scale)])
                y_vgg = self.vgg(pyramide_real['prediction_' + str(scale)])

                for i, weight in enumerate(self.loss_weights['perceptual_final']):
                    value = torch.abs(x_vgg[i] - y_vgg[i].detach()).mean()
                    value_total += self.loss_weights['perceptual_final'][i] * value
            
            loss_values['prediction'] = value_total
            
            if self.num_ref > 1:
                pyramide_generated_more = self.pyramid(generated['prediction_more'])
                pyramide_generated_fusion = self.pyramid(generated['prediction_fusion'])
                
                value_total = 0
                for scale in self.scales:
                    x_vgg = self.vgg(pyramide_generated_more['prediction_' + str(scale)])
                    y_vgg = self.vgg(pyramide_real['prediction_' + str(scale)])

                    for i, weight in enumerate(self.loss_weights['perceptual_final']):
                        value = torch.abs(x_vgg[i] - y_vgg[i].detach()).mean()
                        value_total += self.loss_weights['perceptual_final'][i] * value
            
                loss_values['prediction'] += value_total

                value_total = 0
                for scale in self.scales:
                    x_vgg = self.vgg(pyramide_generated_fusion['prediction_' + str(scale)])
                    y_vgg = self.vgg(pyramide_real['prediction_' + str(scale)])

                    for i, weight in enumerate(self.loss_weights['perceptual_final']):
                        value = torch.abs(x_vgg[i] - y_vgg[i].detach()).mean()
                        value_total += self.loss_weights['perceptual_final'][i] * value
            
                loss_values['fusion'] = value_total
        
              
        ### GAN adversial Loss
        if self.loss_weights['generator_gan'] != 0:
            
            discriminator_maps_generated = self.discriminator(pyramide_generated)
            discriminator_maps_real = self.discriminator(pyramide_real)     
            
            value_total = 0
            for scale in self.disc_scales:
                key = 'prediction_map_%s' % scale
                value = ((1 - discriminator_maps_generated[key]) ** 2).mean()
                value_total += self.loss_weights['generator_gan'] * value
            loss_values['gen_gan'] = value_total

            if sum(self.loss_weights['feature_matching']) != 0:
                value_total = 0
                for scale in self.disc_scales:
                    key = 'feature_maps_%s' % scale
                    for i, (a, b) in enumerate(zip(discriminator_maps_real[key], discriminator_maps_generated[key])):
                        if self.loss_weights['feature_matching'][i] == 0:
                            continue
                        value = torch.abs(a - b).mean()
                        value_total += self.loss_weights['feature_matching'][i] * value
                    loss_values['feature_matching'] = value_total        
            
        return loss_values, generated


class DiscriminatorFullModel(torch.nn.Module):
    """
    Merge all discriminator related updates into single model for better multi-gpu usage
    """

    def __init__(self, kp_extractor, generator, discriminator, train_params):
        super(DiscriminatorFullModel, self).__init__()
        self.kp_extractor = kp_extractor
        self.generator = generator
        self.discriminator = discriminator
        self.train_params = train_params
        self.scales = self.discriminator.scales
        self.pyramid = ImagePyramide(self.scales, generator.num_channels)
        if torch.cuda.is_available():
            self.pyramid = self.pyramid.cuda()

        self.loss_weights = train_params['loss_weights']

    def forward(self, x, generated):
        pyramide_real = self.pyramid(x['driving'])
        pyramide_generated = self.pyramid(generated['prediction'].detach())
        
        discriminator_maps_generated = self.discriminator(pyramide_generated)
        discriminator_maps_real = self.discriminator(pyramide_real)
        
        loss_values = {}
        value_total = 0
        for scale in self.scales:
            key = 'prediction_map_%s' % scale
            value = (1 - discriminator_maps_real[key]) ** 2 + discriminator_maps_generated[key] ** 2
            value_total += self.loss_weights['discriminator_gan'] * value.mean()
        loss_values['disc_gan'] = value_total

        return loss_values
