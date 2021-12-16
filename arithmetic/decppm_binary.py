import numpy as np
import os
from value_decoder import *


if __name__ == "__main__":

    frames=3
   
 
    
    filepath ='C:/Users/bolinchen3/Desktop/codercompress/arith/txt/latent/bin_binary/'

    decpath ='C:/Users/bolinchen3/Desktop/codercompress/arith/txt/latent/rec_binary/'
    
    for sequence in range(1,2):

        ###Recsavepath
        Recsavepath=decpath+'/'+str(sequence) ###
        if not os.path.exists(Recsavepath):
            os.makedirs(Recsavepath)
        
        inputfile1=filepath+str(sequence)+'/'+'1.bin'
        
        #datares_rec_first = final_decoder_unary(inputfile1)
        #true_ae_number = data_convert_inverse_unary(datares_rec_first)


        datares_rec_first = final_decoder_expgolomb(inputfile1) ###############
        true_ae_number = data_convert_inverse_expgolomb(datares_rec_first) ################        

            
        ####VVC参考帧，inter-frame compensation
        refframepath ='C:/Users/bolinchen3/Desktop/codercompress/arith/txt/latent/enc/'+str(sequence)+'/'  #####
        
        latentformat=listformat(refframepath+'0.txt', true_ae_number)

        with open(Recsavepath+'/'+'1.txt','w')as f:  ####
           f.write(latentformat)

        
         #################################################
        for frame in range(2,frames):

            inputfile_inter=filepath+str(sequence)+'/'+str(frame)+'.bin' ###
            

            #datares_rec = final_decoder_unary(inputfile_inter) ###############
            #true_ae_number = data_convert_inverse_unary(datares_rec) ################

            datares_rec = final_decoder_expgolomb(inputfile_inter) ###############
            true_ae_number = data_convert_inverse_expgolomb(datares_rec) ################
            
            ### 第3-frame帧（帧间重建第2··帧 based on 重建第一帧）
            latentformat=listformat(refframepath+str(frame-1)+'.txt', true_ae_number)

            with open(Recsavepath+'/'+str(frame)+'.txt','w')as f:  ########
               f.write(latentformat)    


