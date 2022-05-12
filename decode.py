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
    
    prediction=np.transpose(out['prediction_fusion'].data.cpu().numpy(), [0, 1, 2, 3])[0]

    return prediction


def check_reference(ref_kp_list, kp_current):
    diff_list=[]
    for idx in range(0, len(ref_kp_list)):    
        dif = (ref_kp_list[idx]['value'] - kp_current['value']).abs().mean()
        diff_list.append(dif)
    
    return diff_list

    
if __name__ == "__main__":
    parser = ArgumentParser()
            
    modeldir = 'our2021'  
    savedir = '../experiment_cfte/'
    config_path='../checkpoint/'+modeldir+'/vox-256.yaml'
    checkpoint='../checkpoint/'+modeldir+'/0099-checkpoint.pth.tar'
    os.makedirs(savedir+'/dec/',exist_ok=True)     # the real decoded video
    generator, kp_detector = load_checkpoints(config_path, checkpoint, cpu=False) 

    frames=250
    width=256
    height=256
    Qstep=64
    max_ref_num=1
    
    with open(config_path) as f:
        config = yaml.load(f)      
    keymap_channel = config['common_params']['num_kp']  
    scale_factor = config['common_params']['scale_factor']  
    keymap_size=int(width*scale_factor/16)
    
    #seqlist=['019']
    #qplist=['39']
    seqlist=['001','002','003','004','005','006','007','008','009','010', '011','012','013','014','015','016','017','018','019','020'] 
    qplist=['32', '37', '42', '47', '52']
    
    totalResult=np.zeros((len(seqlist)+1,len(qplist)))
    for seqIdx, seq in enumerate(seqlist):
        for qpIdx, QP in enumerate(qplist):                              
            original_seq='../dataset/'+str(seq)+'_'+str(width)+'x'+str(width)+'.rgb' 
            driving_kp = savedir+'/kp/'+seq            
            decode_seq = savedir+'/dec/'+seq+'_QP'+str(QP)+'.rgb'
            dir_enc = savedir+'/enc/'+seq+'_QP'+str(QP)+'/'
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
                        
                        ###目前熵编码时是基于第一帧的原始图像编码的kp difference
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
                    # check whether refresh reference
                    frame_index=str(frame_idx).zfill(4)
                    bin_save=driving_kp+'/frame'+frame_index+'.bin'            
                    kp_dec = final_decoder_expgolomb(bin_save)
                    kp_difference = data_convert_inverse_expgolomb(kp_dec)   
                    kp_previous=json.loads(seq_kp_integer[frame_idx-1])
                    kp_previous= eval('[%s]'%repr(kp_previous).replace('[', '').replace(']', ''))                         
                    kp_integer=listformat_adptive(kp_previous, kp_difference, keymap_channel, keymap_size)
                    seq_kp_integer.append(kp_integer)                        
                    kp_integer=json.loads(kp_integer)
                    kp_current_value=torch.Tensor(kp_integer).to('cuda:0')          
                    kp_current_value=(kp_current_value*1)/Qstep-1  
                    dict={}
                    dict['value']=kp_current_value  
                    kp_current=dict 
                    
                    diff_list = check_reference(ref_kp_list, kp_current)
                                
                    # reference frame    
                    if min(diff_list) > 10000:                   #  0.1 , 0.15 , 0.2 
                        
                        f_temp=open(dir_enc+'frame'+frame_idx_str+'_org.yuv','w')
                        # wtite ref and cur (rgb444) to file (yuv420)
                        img_ref = ref_rgb_list[-1]
                        img_ref = img_ref.transpose(1, 2, 0)    # HxWx3
                        image_yuv = cv2.cvtColor(img_ref, cv2.COLOR_RGB2YUV)
                        image_yuv = image_yuv.transpose(2, 0, 1)   # 3xHxW
                        image_yuv[0,:,:].tofile(f_temp)
                        image_yuv = image_yuv[:,::2,::2]
                        image_yuv[1,:,:].tofile(f_temp)
                        image_yuv[2,:,:].tofile(f_temp)
                        
                        img_input_ = img_input.transpose(1, 2, 0)    # HxWx3
                        image_yuv = cv2.cvtColor(img_input_, cv2.COLOR_RGB2YUV)
                        image_yuv = image_yuv.transpose(2, 0, 1)   # 3xHxW
                        image_yuv[0,:,:].tofile(f_temp)
                        image_yuv = image_yuv[:,::2,::2]
                        image_yuv[1,:,:].tofile(f_temp)
                        image_yuv[2,:,:].tofile(f_temp)
                        f_temp.close()
                        
                        qp_pframe = int(QP) - 10                    
                        os.system("./vtm/encodeCLIC.sh "+dir_enc+'frame'+frame_idx_str+" "+str(qp_pframe)) 
                        
                        bin_file=dir_enc+'frame'+frame_idx_str+'.bin'
                        bits=os.path.getsize(bin_file)*8
                        sum_bits += bits
                    
                        #  read the rec frame (yuv420) and convert to rgb444
                        f_temp=open(dir_enc+'frame'+frame_idx_str+'_rec.yuv','rb')
                        img_rec=np.fromfile(f_temp,np.uint8,3*height*width//2)    # skip the refence frame
                        img_rec_Y = np.fromfile(f_temp,np.uint8,height*width).reshape((height,width))
                        img_rec_U = np.fromfile(f_temp,np.uint8,height*width//4).reshape((height//2, width//2))
                        img_rec_V = np.fromfile(f_temp,np.uint8,height*width//4).reshape((height//2, width//2))
                        img_rec_U=np.repeat(img_rec_U,2,axis=1)
                        img_rec_U=np.repeat(img_rec_U,2,axis=0)                        
                        img_rec_V=np.repeat(img_rec_V,2,axis=1)
                        img_rec_V=np.repeat(img_rec_V,2,axis=0)  
                        img_rec = np.array([img_rec_Y, img_rec_U, img_rec_V])   # 3xHxW
                        img_rec = img_rec.transpose(1, 2, 0)    # HxWx3
                        img_rec = cv2.cvtColor(img_rec, cv2.COLOR_YUV2RGB)
                        img_rec = img_rec.transpose(2, 0, 1)   # 3xHxW
                        img_rec.tofile(f_dec)
                        
                        if not len(ref_rgb_list) < max_ref_num:
                            ref_rgb_list.pop(0)
                            ref_norm_list.pop(0)
                            ref_kp_list.pop(0)
                        
                        ref_rgb_list.append(img_rec) 
                        
                        img_rec = resize(img_rec, (3, height, width))                                      
                        with torch.no_grad(): 
                            reference = torch.tensor(img_rec[np.newaxis].astype(np.float32))
                            reference = reference.cuda()    # require GPU
                            kp_reference = kp_detector(reference)                          
                            ref_norm_list.append(reference)
                            ref_kp_list.append(kp_reference)
                            
                    # generated frame
                    else: 
                        ref_idx = diff_list.index(min(diff_list))
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

    np.savetxt(savedir+'/resultBit.txt', totalResult, fmt = '%.5f')
            
                
                
