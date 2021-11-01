# -*- coding: utf-8 -*-
import matplotlib
matplotlib.use('Agg')
import os, sys
import yaml
from argparse import ArgumentParser
from tqdm import tqdm
import imageio
import numpy as np
from skimage.transform import resize
from skimage import img_as_ubyte
import torch
from sync_batchnorm import DataParallelWithCallback
from modules.generator import OcclusionAwareGenerator
from modules.keypoint_detector import KPDetector
from animate import normalize_kp
from scipy.spatial import ConvexHull
import scipy.io as io
import json
import cv2
import torch.nn.functional as F

if sys.version_info[0] < 3:
    raise Exception("You must use Python 3 or higher. Recommended version is Python 3.7")


def load_checkpoints(config_path, checkpoint_path, cpu=False):

    with open(config_path) as f:
        config = yaml.load(f)

    generator = OcclusionAwareGenerator(**config['model_params']['generator_params'],
                                        **config['model_params']['common_params'])
    if not cpu:
        generator.cuda()

    kp_detector = KPDetector(**config['model_params']['kp_detector_params'],
                             **config['model_params']['common_params'])
    if not cpu:
        kp_detector.cuda()
    
    if cpu:
        checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
    else:
        checkpoint = torch.load(checkpoint_path)
 
    generator.load_state_dict(checkpoint['generator'],strict=False)
    kp_detector.load_state_dict(checkpoint['kp_detector']) ####
    
    if not cpu:
        generator = DataParallelWithCallback(generator)
        kp_detector = DataParallelWithCallback(kp_detector)

    generator.eval()
    kp_detector.eval()
    
    return generator, kp_detector


def make_animation(source_image, driving_video_frame_kp, generator, kp_detector,frames,Qstep, relative=True, adapt_movement_scale=True, cpu=False):
    driving_list=[]
    with torch.no_grad():
        predictions = []
        source = torch.tensor(source_image[np.newaxis].astype(np.float32)).permute(0, 3, 1, 2)
        if not cpu:
            source = source.cuda()
        kp_source = kp_detector(source)   ####
        driving_list.append(kp_source)

        for frame in range(1,frames):
            txt, i= {}, 1
            txt_save=driving_video_frame_kp+'/'+str(frame)+'.txt'

            with open(txt_save,'r+',encoding='utf-8') as f:
                for line in f:
                    txt[i] = line
                    i += 1
                dict={}
                list_value=txt[1]
                list_value=json.loads(list_value)
                
                
                device = 'cuda:0'
                kp_value_list=torch.Tensor(list_value).to(device)
                kp_value_list=(kp_value_list*1)/Qstep-1    
                
                kp_value = kp_value_list.cuda().data.cpu().numpy()
                tensor_list1=kp_value.tolist()
                tensor_list1=str(tensor_list1)
                tensor_list1="".join(tensor_list1.split())  

                
                dict['value']=kp_value_list
                

                
                kp_driving_video=dict
                driving_list.append(kp_driving_video)    
    Driving=driving_list   

    for k in range(0,frames):        
        kp_norm = normalize_kp(kp_source=kp_source, kp_driving=Driving[k],
                               kp_driving_initial=kp_source, use_relative_movement=relative,
                               use_relative_jacobian=relative, adapt_movement_scale=adapt_movement_scale)
        out = generator(source, heatmap_source=kp_source,heatmap_driving=kp_norm)
            
        predictions.append(np.transpose(out['prediction'].data.cpu().numpy(), [0, 2, 3, 1])[0])
        #print(predictions[k])

    return predictions

def RawReader_planar(FileName, ImgWidth, ImgHeight, NumFramesToBeComputed):
    
    f   = open(FileName, 'rb')
    frames  = NumFramesToBeComputed
    width   = ImgWidth
    height  = ImgHeight
    data = f.read()
    f.close()
    data = [int(x) for x in data]

    data_list=[]
    n=width*height
    for i in range(0,len(data),n):
        b=data[i:i+n]
        data_list.append(b)
    x=data_list

    listR=[]
    listG=[]
    listB=[]
    for k in range(0,frames):
        
        R=np.array(x[3*k]).reshape((width, height)).astype(np.uint8)
        G=np.array(x[3*k+1]).reshape((width, height)).astype(np.uint8)
        B=np.array(x[3*k+2]).reshape((width, height)).astype(np.uint8)
        listR.append(R)
        listG.append(G)
        listB.append(B)
    return listR,listG,listB

def splitlist(list): 
    alist = []
    a = 0 
    for sublist in list:
        try: #用try来判断是列表中的元素是不是可迭代的，可以迭代的继续迭代
            for i in sublist:
                alist.append (i)
        except TypeError: #不能迭代的就是直接取出放入alist
            alist.append(sublist)
    for i in alist:
        if type(i) == type([]):#判断是否还有列表
            a =+ 1
            break
    if a==1:
        return printlist(alist) #还有列表，进行递归
    if a==0:
        return alist  

if __name__ == "__main__":
    parser = ArgumentParser()

    config_path='//video-standardization/blchen/FVC/test-0901/4-supervised-baseline/ckp/vox-256.yaml'
    checkpoint='//video-standardization/blchen/FVC/test-0901/4-supervised-baseline/ckp/00000099-checkpoint.pth.tar'
    frames=100
    width=256
    height=256
    Qstep=16
    
#     for QP in [22,27,32,37,42,47]: 
#         print(QP)
    for i in range(1,14):


        #source_rgb='//video-standardization/blchen/mixdata/vtm/'+str(QP)+'/'+str(i)+'.rgb'
        source_rgb='/video-standardization/blchen/FVC/test-0901/comparison/data/'+'/'+str(i)+'_256x256_1_8bit.rgb'      
        listR,listG,listB=RawReader_planar(source_rgb,width, height,1)
        source_image = cv2.merge([listR[0],listG[0],listB[0]])
        source_image = resize(source_image, (width, height))[..., :3]

        driving_video_frame_kp='/video-standardization/blchen/FVC/test-0901/comparison/feature/handacraft/'+str(i)+'/'     
        generator, kp_detector = load_checkpoints(config_path, checkpoint, cpu=False)
        predictions = make_animation(source_image, driving_video_frame_kp, generator, kp_detector, frames,
                                     Qstep,relative=False, adapt_movement_scale=False, cpu=False)
        #print(predictions)


        predi=[]
        rgb=[]
        for frame in range(0,frames): 
            pre=(resize(predictions[frame],(width, height))*255).astype(np.uint8)       
            pre = cv2.cvtColor(pre, cv2.COLOR_BGR2RGB)    #RGB
            predi.append(pre)

            b,g,r=cv2.split(predi[frame]) 
            r=splitlist(np.array(r).reshape((width*height,1)))
            g=splitlist(np.array(g).reshape((width*height,1)))
            b=splitlist(np.array(b).reshape((width*height,1)))
            rgb.append(r)
            rgb.append(g)
            rgb.append(b)

        rgb_new=[]
        rgb_new.extend(listR[0])  
        rgb_new.extend(listG[0]) 
        rgb_new.extend(listB[0])       
        for rgb_i in range(3,frames*3):
            rgb_new.extend(rgb[rgb_i])

        result_rgb='/video-standardization/blchen/FVC/test-0901/comparison/rgbgene/handacraft/'#+str(QP)+'/'  
        if not os.path.exists(result_rgb):
            os.makedirs(result_rgb) 

        with open(result_rgb+str(i)+'.rgb', 'wb+') as f:
             f.writelines(rgb_new)
