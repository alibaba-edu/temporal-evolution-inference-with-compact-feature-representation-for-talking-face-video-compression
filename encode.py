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
import time
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


def make_animation(reference_frame, kp_reference, driving_kp, generator, kp_detector,frame_idx, Qstep, relative=False, adapt_movement_scale=False, cpu=False):
        
    txt, i= {}, 1
    frame_index=''
    if frame_idx < 10:
        frame_index+='0'
    if frame_idx < 100:
        frame_index+='0'
    frame_index+=str(frame_idx)
    txt_save=driving_kp+'/frame'+frame_index+'.txt'

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
        dict['value']=kp_value_list        
                
        kp_current=dict    

    kp_norm = normalize_kp(kp_source=kp_reference, kp_driving=kp_current,
                               kp_driving_initial=kp_reference, use_relative_movement=relative,
                               use_relative_jacobian=relative, adapt_movement_scale=adapt_movement_scale)
    out = generator(reference_frame, heatmap_source=kp_reference, heatmap_driving=kp_norm)
    
    prediction=np.transpose(out['prediction'].data.cpu().numpy(), [0, 1, 2, 3])[0]

    return prediction


def check_reference(reference_frame, current_frame, frame_idx):
    dif = (reference_frame.astype(np.int16) - current_frame.astype(np.int16))**2
    mse=np.mean(dif)
        
    if mse>400:   
        return False
    else:
        return False


# +
if __name__ == "__main__":
    parser = ArgumentParser()
        
    frames=100
    width=256
    height=256
    Qstep=16
    
    modelkp = 'spatial'  
    config_path='/home/yixiubaixue_ex/Face/checkpoint/'+'spatial'+'/vox-256.yaml'
    checkpoint='/home/yixiubaixue_ex/Face/checkpoint/'+'spatial'+'/00000099-checkpoint.pth.tar'

    os.makedirs("../experiment/dec/",exist_ok=True)     # the real decoded video
    
    seqlist=[ "48" ]
    qplist=["42"]
    
    for seq in seqlist:
        for QP in qplist:  
                
            start=time.time()
                   
            original_seq='../dataset/'+seq+'_256x256_1_8bit.rgb'
            reference_seq='../vtm/rec/'+seq+'_QP'+str(QP)+'_vtm.rgb'
            driving_kp = '../experiment/kp/'+seq            
            decode_seq="../experiment/dec/"+seq+'_QP'+str(QP)+'.rgb'
            dir_enc = "../experiment/enc/"+seq+'_QP'+str(QP)+'/'
            os.makedirs(dir_enc,exist_ok=True)     # some original frames to be compressed by vtm  
                
            generator, kp_detector = load_checkpoints(config_path, checkpoint, cpu=False) 
                
            f_org=open(original_seq,'rb')
            f_dec=open(decode_seq,'w') 
                
            for frame_idx in range(0, frames):             
                frame_idx_str=''
                if frame_idx < 10:
                    frame_idx_str+='0'
                if frame_idx < 100:
                    frame_idx_str+='0'
                frame_idx_str += str(frame_idx)
                              
                img_input=np.fromfile(f_org,np.uint8,3*height*width).reshape((3,height,width))  #RGB
                
                if frame_idx in [0]:  
                    reference_org = img_input
                    
                    f_temp=open(dir_enc+'frame'+frame_idx_str+'_org.rgb','w')
                    img_input.tofile(f_temp)
                    f_temp.close()
                                       
                    os.system("./vtm/encode.sh "+dir_enc+'frame'+frame_idx_str+" "+QP)   
                    
                    f_temp=open(dir_enc+'frame'+frame_idx_str+'_rec.rgb','rb')
                    img_rec=np.fromfile(f_temp,np.uint8,3*height*width).reshape((3,height,width))           
                    img_rec.tofile(f_dec)
                    img_rec = resize(img_rec, (3, height, width))
                                        
                    with torch.no_grad(): 
                        reference = torch.tensor(img_rec[np.newaxis].astype(np.float32))
                        reference = reference.cuda()    # require GPU
                        kp_reference = kp_detector(reference) 
                        
                elif check_reference(reference_org, img_input, frame_idx):
                    reference_org = img_input
                        
                    f_temp=open(dir_enc+'frame'+frame_idx_str+'_org.rgb','w')
                    img_input.tofile(f_temp)
                    f_temp.close()
                    
                    os.system("./vtm/encode.sh "+dir_enc+'frame'+frame_idx_str+" "+QP)   
                    
                    f_temp=open(dir_enc+'frame'+frame_idx_str+'_rec.rgb','rb')
                    img_rec=np.fromfile(f_temp,np.uint8,3*height*width).reshape((3,height,width))           
                    img_rec.tofile(f_dec)
                    img_rec = resize(img_rec, (3, height, width))
                                        
                    with torch.no_grad(): 
                        reference = torch.tensor(img_rec[np.newaxis].astype(np.float32))
                        reference = reference.cuda()    # require GPU
                        kp_reference = kp_detector(reference)                    
                    
                else:
                    prediction = make_animation(reference, kp_reference, driving_kp, generator, kp_detector, frame_idx, Qstep)
                    pre=(prediction*255).astype(np.uint8)  
                    pre.tofile(f_dec)
                               
            f_org.close()
            f_dec.close()
                
            end=time.time()
            print(seq+'_QP'+str(QP)+'.rgb',"success. Time is:%.4f"%(end-start))
                
                
                
