import os, sys
import yaml
from argparse import ArgumentParser
import numpy as np
from skimage.transform import resize
import torch
from sync_batchnorm import DataParallelWithCallback
from modules.keypoint_detector import KPDetector
import time
import random
import pandas as pd
import collections
import itertools
from arithmetic.value_encoder import *

def load_checkpoints(config_path, checkpoint_path, cpu=False):
    with open(config_path) as f:
        config = yaml.load(f)
    kp_detector = KPDetector(**config['model_params']['kp_detector_params'],
                             **config['common_params'])
        
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


if __name__ == "__main__":
   
    parser = ArgumentParser()

    frames=250
    width=256
    height=256
    Qstep=64

    modeldir = 'our2021'
    config_path='../checkpoint/'+modeldir+'/vox-256.yaml'
    checkpoint_path='../checkpoint/'+modeldir+'/0099-checkpoint.pth.tar'
    save_path='../experiment_cfte/kp/'
    
    kp_detector = load_checkpoints(config_path, checkpoint_path, cpu=False)
    
    #seqlist=['013','019']     
    seqlist=['001','002','003','004','005','006','007','008','009','010', '011','012','013','014','015','016','017','018','019','020']
    
    for seq in seqlist:
        
        start=time.time()
        
        kp_path=save_path+str(seq)
        if not os.path.exists(kp_path):
            os.makedirs(kp_path)

        original_seq='../dataset/'+str(seq)+'_'+str(height)+'x'+str(width)+'.rgb' 
        f_org=open(original_seq,'rb')
        
        kp_value_seq = []
        for frame in range(0,frames):
            img = np.fromfile(f_org,np.uint8,3*height*width).reshape((3,height,width))  #3xHxW RGB
            img = resize(img, (3, height, width))    # resize to 0-1
            img = torch.tensor(img[np.newaxis].astype(np.float32))
            img = img.cuda()    # require GPU
            
            kp_image = kp_detector(img) 
            kp_value = kp_image['value']
            kp_value = torch.round((kp_value+1)*Qstep/1)              
            kp_value = kp_value.int()
                
            kp_value_list = kp_value.tolist()
            kp_value_list = str(kp_value_list)
            kp_value_list = "".join(kp_value_list.split())
                
            frame_idx = str(frame).zfill(4)
            with open(kp_path+'/frame'+frame_idx+'.txt','w')as f:
                f.write(kp_value_list)  
                
            kp_value_frame=json.loads(kp_value_list)
            kp_value_frame= eval('[%s]'%repr(kp_value_frame).replace('[', '').replace(']', ''))
            kp_value_seq.append(kp_value_frame)
        
        
        sum_bits=0
        for frame in range(1,frames):
            kp_difference=(np.array(kp_value_seq[frame])-np.array(kp_value_seq[frame-1])).tolist()
            frame_idx = str(frame).zfill(4)
            bin_file=kp_path+'/frame'+frame_idx+'.bin'
          
            final_encoder_expgolomb(kp_difference,bin_file)     
     
            bits=os.path.getsize(bin_file)*8
            sum_bits += bits
            
        
        end=time.time()
        print("Extracting kp success. Time is %.4fs. Key points coding %d bits." %(end-start, sum_bits))
