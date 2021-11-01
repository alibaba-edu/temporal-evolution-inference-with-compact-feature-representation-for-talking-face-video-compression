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
import scipy.io as io
import json
import numpy as np
import cv2
import math

if sys.version_info[0] < 3:
    raise Exception("You must use Python 3 or higher. Recommended version is Python 3.7")

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

def make_animation(source,kp_source, kp_driving, generator, kp_detector, relative=True, adapt_movement_scale=True, cpu=False):
    with torch.no_grad():
        predictions = []
        kp_norm = normalize_kp(kp_source=kp_source, kp_driving=kp_driving,
                               kp_driving_initial=kp_source, use_relative_movement=relative,
                               use_relative_jacobian=relative, adapt_movement_scale=adapt_movement_scale)
        out = generator(source, kp_source=kp_source, kp_driving=kp_norm)

        predictions.append(np.transpose(out['prediction'].data.cpu().numpy(), [0, 2, 3, 1])[0])
    return predictions


def load_checkpoints(config_path, checkpoint_path, cpu=False):
    with open(config_path) as f:
        config = yaml.load(f)
    kp_detector = KPDetector(**config['model_params']['kp_detector_params'],
                             **config['model_params']['common_params'])
    if not cpu:
        kp_detector.cuda()
    
    if cpu:
        checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
    else:
        checkpoint = torch.load(checkpoint_path)
 
    kp_detector.load_state_dict(checkpoint['kp_detector']) ####
    
    if not cpu:
        kp_detector = DataParallelWithCallback(kp_detector)
    kp_detector.eval()
    
    return kp_detector

def make_keypoint(source_image, kp_detector, cpu=False):
    source = torch.tensor(source_image[np.newaxis].astype(np.float32)).permute(0, 3, 1, 2)
    if not cpu:
        source = source.cuda()
    kp_source = kp_detector(source)   ####
    return kp_source

if __name__ == "__main__":
   
    parser = ArgumentParser()
    
    seqlist=[ "47" ]
    for modelkp in ['spatial']:
        config_path='/home/yixiubaixue_ex/Face/checkpoint/'+modelkp+'/vox-256.yaml'
        checkpoint_path='/home/yixiubaixue_ex/Face/checkpoint/'+modelkp+'/00000099-checkpoint.pth.tar'

        spatial_size=256
        Qstep=16
        save_path='/home/yixiubaixue_ex/Face/experiment/kp/'
        ori_rgb='/home/yixiubaixue_ex/Face/dataset/'

        width=256
        height=256
        frames =100

        for seq in seqlist:
            kp_path=save_path+str(seq)
            if not os.path.exists(kp_path):
                os.makedirs(kp_path)

            rgbname=ori_rgb+seq+'_256x256_1_8bit.rgb'          
            
            listR,listG,listB=RawReader_planar(rgbname,width, height,frames)

            for frame in range(0,frames):
                source_image = cv2.merge([listR[frame],listG[frame],listB[frame]])

                source_image = resize(source_image, (256, 256))[..., :3]

                kp_detector = load_checkpoints(config_path, checkpoint_path, cpu=False)

                kp_image = make_keypoint(source_image, kp_detector, cpu=False)

                kp_value=kp_image['value']
                #kp_value=kp_image

                #print(kp_value)
                kp_value=torch.round((kp_value+1)*Qstep/1)              

                kp_value=kp_value.int()
                #print(kp_value)
                kp_value_list=kp_value.tolist()
                kp_value_list=str(kp_value_list)
                kp_value_list="".join(kp_value_list.split())
                
                tmp=''
                if frame < 10:
                    tmp+='0'
                if frame < 100:
                    tmp+='0'
                tmp+=str(frame)

                with open(kp_path+'/frame'+tmp+'.txt','w')as f:
                   f.write(kp_value_list)    
