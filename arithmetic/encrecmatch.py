
import numpy as np
import random
import pandas as pd
import collections
import itertools
import json
import os


frames=50

Encpath ='C:/Users/bolinchen3/Desktop/codercompress/arith/txt/latent/enc/'

Recpath ='C:/Users/bolinchen3/Desktop/codercompress/arith/txt/latent/rec_binary/'

for sequence in range(1,38):
    
     for frame in range(1,frames):
        txt1, i= {}, 1
        txt_save_enc=Encpath+'/'+str(sequence)+'/'+str(frame)+'.txt' #####

        with open(txt_save_enc,'r+',encoding='utf-8') as f:
            for line in f:
                txt1[i] = line
                i += 1
            dict={}
            list_value_enc=txt1[1]
            list_value_enc=json.loads(list_value_enc)
            data_enc = eval('[%s]'%repr(list_value_enc).replace('[', '').replace(']', '')) 
            
            
        txt, i= {}, 1
        txt_save_rec=Recpath+'/'+str(sequence)+'/'+str(frame)+'.txt' #####

        with open(txt_save_rec,'r+',encoding='utf-8') as f:
            for line in f:
                txt[i] = line
                i += 1
            dict={}
            list_value_rec=txt[1]
            list_value_rec=json.loads(list_value_rec)
            data_rec = eval('[%s]'%repr(list_value_rec).replace('[', '').replace(']', ''))  
        
        if data_enc==data_rec:
            print ("It is matching")
        else:
            print("It is not maching")
        
