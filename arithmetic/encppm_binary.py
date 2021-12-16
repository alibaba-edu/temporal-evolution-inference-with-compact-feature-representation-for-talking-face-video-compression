import numpy as np
import random
import pandas as pd
import collections
import itertools
import json
import os
from value_encoder import *
        



if __name__ == "__main__":

    frames=5
    

    encodepath ='/mnt/workspace/code/Arithmetic/Arithmetic/latent/enc/'
    binsavepath='/mnt/workspace/code/Arithmetic/Arithmetic/latent/bin_new/'

    sum_bits=0

    for sequence in range(1,2):
        
        binsavepath_per=binsavepath+str(sequence)
        if not os.path.exists(binsavepath_per):
            os.makedirs(binsavepath_per)       
        
        #restotal=[]
        for frame in range(0,frames-1):
            txt1, i= {}, 1
            txt_save_forward=encodepath+'/'+str(sequence)+'/'+str(frame)+'.txt' #####

            with open(txt_save_forward,'r+',encoding='utf-8') as f:
                for line in f:
                    txt1[i] = line
                    i += 1
                dict={}
                list_value_forward=txt1[1]
                list_value_forward=json.loads(list_value_forward)
                data1 = eval('[%s]'%repr(list_value_forward).replace('[', '').replace(']', ''))
                #print(data1)

            txt, i= {}, 1        
            txt_save_backward=encodepath+'/'+str(sequence)+'/'+str(frame+1)+'.txt'  #####
            with open(txt_save_backward,'r+',encoding='utf-8') as f:
                for line in f:
                    txt[i] = line
                    i += 1
                dict={}
                list_value_backward=txt[1]
                list_value_backward=json.loads(list_value_backward)
                data2 = eval('[%s]'%repr(list_value_backward).replace('[', '').replace(']', ''))
                print(data2)

            ##inter-frame prediction
            datares=(np.array(data2)-np.array(data1)).tolist()
            print(datares)


    
            outputfile=binsavepath_per+'/'+str(frame+1)+'.bin'
            #final_encoder_unary(datares,outputfile)
            final_encoder_expgolomb(datares,outputfile) 
            
            
            
            bits=os.path.getsize(outputfile)*8
            sum_bits=sum_bits+bits
        print(sum_bits)
    
            

            
            

                
