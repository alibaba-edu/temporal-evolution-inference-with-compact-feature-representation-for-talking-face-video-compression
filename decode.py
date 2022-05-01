import os, sys
import yaml
from argparse import ArgumentParser
import numpy as np
from skimage.transform import resize
import torch
from sync_batchnorm import DataParallelWithCallback
from modules.generator import OcclusionAwareGenerator
from modules.keypoint_detector import KPDetector
from animate import normalize_kp
import json
import time
import cv2
from arithmetic.value_decoder import *

def rgb4442yuv420(dir1, dir2):    
    os.makedirs(dir2,exist_ok=True)
    files=glob.glob(os.path.join(dir1,'*.rgb'))
    files.sort()
    for file in files:  
        f=open(file,'rb')
        file_name=file.split('/')[-1]
        file_name=os.path.splitext(file_name)[0]
        tar_path=dir2+file_name+'.yuv'
        yuvfile=open(tar_path, mode='w')

        width=256
        height=256
        framenum=250
        for idx in range(framenum):
            R=np.fromfile(f,np.uint8,width*height).reshape((height,width))
            G=np.fromfile(f,np.uint8,width*height).reshape((height,width))   
            B=np.fromfile(f,np.uint8,width*height).reshape((height,width))
            image_rgb = np.zeros((height,width,3), 'uint8')
            image_rgb[..., 0] = R
            image_rgb[..., 1] = G
            image_rgb[..., 2] = B
            image_yuv = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2YUV)
            image_yuv[:,:,0].tofile(yuvfile)
            image_yuv[:,:,1].tofile(yuvfile)
            image_yuv[:,:,2].tofile(yuvfile)
        
    print('done')


def load_checkpoints(config_path, checkpoint_path, cpu=False):

    with open(config_path) as f:
        config = yaml.load(f)

    generator = OcclusionAwareGenerator(**config['model_params']['generator_params'],
                                        **config['common_params'])
    if not cpu:
        generator.cuda()

    kp_detector = KPDetector(**config['model_params']['kp_detector_params'],
                             **config['common_params'])
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


def make_prediction(reference_frame, kp_reference, kp_current, generator, relative=False, adapt_movement_scale=False, cpu=False):
        
    kp_norm = normalize_kp(kp_source=kp_reference, kp_driving=kp_current,
                               kp_driving_initial=kp_reference, use_relative_movement=relative,
                               use_relative_jacobian=relative, adapt_movement_scale=adapt_movement_scale)
    
    out = generator(reference_frame, kp_reference, kp_norm, reference_frame, kp_reference)
    
    prediction=np.transpose(out['prediction'].data.cpu().numpy(), [0, 1, 2, 3])[0]

    return prediction


def check_reference(ref_kp_list, kp_current):
    diff_list=[]
    for idx in range(0, len(ref_kp_list)):    
        dif = (ref_kp_list[idx]['value'] - kp_current['value']).abs().mean()
        diff_list.append(dif)
    
    return diff_list

    
if __name__ == "__main__":
    parser = ArgumentParser()
            
    modeldir = 'test'  
    config_path='../checkpoint/'+modeldir+'/vox-256.yaml'
    checkpoint='../checkpoint/'+modeldir+'/0099-checkpoint.pth.tar'
    os.makedirs("../experiment/dec/",exist_ok=True)     # the real decoded video
    generator, kp_detector = load_checkpoints(config_path, checkpoint, cpu=False) 

    frames=250
    width=256
    height=256
    Qstep=128
    
    seqlist=["40", "41", "42", "43", "44", "45", "46", "47", "48", "49", "50", "51"]
    qplist=["32", "37", "42", "47", "52"]
    #seqlist=["47"]
    #qplist=["42"]
    
    totalResult=np.zeros((len(seqlist)+1,len(qplist)))
    for seqIdx, seq in enumerate(seqlist):
        for qpIdx, QP in enumerate(qplist):                              
            original_seq='../dataset/'+seq+'_256x256_1_8bit.rgb'
            driving_kp = '../experiment/kp/'+seq            
            decode_seq="../experiment/dec/"+seq+'_QP'+str(QP)+'.rgb'
            dir_enc = "../experiment/enc/"+seq+'_QP'+str(QP)+'/'
            os.makedirs(dir_enc,exist_ok=True)     # the frames to be compressed by vtm                 
            
            f_org=open(original_seq,'rb')
            f_dec=open(decode_seq,'w') 
            ref_rgb_list=[]
            ref_norm_list=[]
            ref_kp_list=[]
            seq_kp_integer=[]
            
            start=time.time() 
            gene_time = 0
            
            sum_bits = 0
            for frame_idx in range(0, frames):            
                
                frame_idx_str = str(frame_idx).zfill(4)   
                
                img_input=np.fromfile(f_org,np.uint8,3*height*width).reshape((3,height,width))  #RGB
                
                if frame_idx in [0]:      # I-frame                        
                    f_temp=open(dir_enc+'frame'+frame_idx_str+'_org.rgb','w')
                    img_input.tofile(f_temp)
                    f_temp.close()
                                       
                    os.system("./vtm/encode.sh "+dir_enc+'frame'+frame_idx_str+" "+QP)   
                    
                    bin_file=dir_enc+'frame'+frame_idx_str+'.bin'
                    bits=os.path.getsize(bin_file)*8
                    sum_bits += bits
                    
                    f_temp=open(dir_enc+'frame'+frame_idx_str+'_rec.rgb','rb')
                    img_rec=np.fromfile(f_temp,np.uint8,3*height*width).reshape((3,height,width))   # 3xHxW RGB         
                    img_rec.tofile(f_dec) 
                    
                    ref_rgb_list.append(img_rec)
                    
                    img_rec = resize(img_rec, (3, height, width))    # normlize to 0-1                                      
                    with torch.no_grad(): 
                        reference = torch.tensor(img_rec[np.newaxis].astype(np.float32))
                        reference = reference.cuda()    # require GPU
                        kp_reference = kp_detector(reference) 
                        ref_norm_list.append(reference)
                        ref_kp_list.append(kp_reference)
                        
                        ###
                        img_input = resize(img_input, (3, height, width))  
                        cur = torch.tensor(img_input[np.newaxis].astype(np.float32))
                        cur = cur.cuda()
                        kp_cur = kp_detector(cur) 
                        
                        kp_integer=torch.round((kp_cur['value']+1)*Qstep/1)
                        kp_integer=kp_integer.int()
                        kp_integer=kp_integer.tolist()
                        kp_integer=str(kp_integer)
                        kp_integer="".join(kp_integer.split())
                        seq_kp_integer.append(kp_integer)
                                                
                else:
                    
                    frame_index=str(frame_idx).zfill(4)
                    bin_save=driving_kp+'/frame'+frame_index+'.bin'            
                    kp_dec = final_decoder_expgolomb(bin_save)
                    kp_difference = data_convert_inverse_expgolomb(kp_dec)
                        
                    kp_previous=json.loads(seq_kp_integer[frame_idx-1])
                    kp_previous= eval('[%s]'%repr(kp_previous).replace('[', '').replace(']', ''))                         
                    kp_integer=listformat_ori(kp_previous, kp_difference)
                        
                    seq_kp_integer.append(kp_integer)
                        
                    kp_integer=json.loads(kp_integer)
                    kp_current_value=torch.Tensor(kp_integer).to('cuda:0')          
                    kp_current_value=(kp_current_value*1)/Qstep-1  
                    dict={}
                    dict['value']=kp_current_value  
                    kp_current=dict 
                    
                    
                    ref_idx = 0
                    reference = ref_norm_list[ref_idx]
                    kp_reference = ref_kp_list[ref_idx]
                        
                    gene_start = time.time()
                    prediction = make_prediction(reference, kp_reference, kp_current, generator)
                    gene_end = time.time()
                    gene_time += gene_end - gene_start
                    pre=(prediction*255).astype(np.uint8)  
                    pre.tofile(f_dec)                              

                    frame_index=str(frame_idx).zfill(4)
                    bin_save=driving_kp+'/frame'+frame_index+'.bin'
                    bits=os.path.getsize(bin_save)*8
                    sum_bits += bits
                               
            f_org.close()
            f_dec.close()     
            end=time.time()
            
            totalResult[seqIdx][qpIdx]=sum_bits           
            print(seq+'_QP'+str(QP)+'.rgb',"success. Total time is %.4fs. Model inference time is %.4fs. Total bits are %d" %(end-start,gene_time,sum_bits))
    
    # summary the bitrate
    for qp in range(len(qplist)):
        for seq in range(len(seqlist)):
            totalResult[-1][qp]+=totalResult[seq][qp]
        totalResult[-1][qp] /= len(seqlist)
    
    print(totalResult)
    np.set_printoptions(precision=5)
    totalResult = totalResult/1000
    seqlength = frames/25
    totalResult = totalResult/seqlength

    np.savetxt('../experiment/resultBit.txt', totalResult, fmt = '%.5f')
            
                
                
