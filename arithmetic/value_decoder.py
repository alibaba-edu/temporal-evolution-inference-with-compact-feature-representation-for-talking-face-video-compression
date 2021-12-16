import sys
from arithmetic.arithmeticcoding import *
from arithmetic.ppmmodel import *
import numpy as np
import random
import pandas as pd
import collections
import itertools
import json
import os
import struct
from collections import Counter
from arithmetic.expgolomb_encode_decode import *

############################# unary code#######################################
##对解码得到的数字list进行分割函数
def list_deal(list_ori,p):   
    list_new=[]				#处理后的列表，是一个二维列表
    list_short=[]			#用于存放每一段列表
    for i in list_ori:
        if i!=p:		
            list_short.append(i)
        else:
            list_new.append(list_short)
            list_short=[]
    list_new.append(list_short)   #最后一段遇不到切割标识，需要手动放入
    return list_new

##对手动切割的list进行相关的数字个数统计函数
def  count_list(std:list,tongji):
    name=Counter()
    for  num in std:
        name[num]+=1
    #print(name[tongji])
    return name[tongji]

## final decode: input: bin file; output: the 0/1 value
def final_decoder_unary(inputfile,MODEL_ORDER = 0):
    # Must be at least -1 and match ppm-compress.py. Warning: Exponential memory usage at O(257^n).
    # Perform file decompression
    with open(inputfile, "rb") as inp:
        bitin = BitInputStream(inp)  #arithmeticcoding.

        dec = ArithmeticDecoder(256, bitin) ##############arithmeticcoding.
        model = PpmModel(MODEL_ORDER, 3, 2) #######ppmmodel.
        history = []
        
        datares_rec=[]
                    
        while True:
            symbol = decode_symbol(dec, model, history)
            if symbol ==2:
                break

            model.increment_contexts(history, symbol)
            datares_rec.append(symbol)
            if model.model_order >= 1:
                # Prepend current symbol, dropping oldest symbol if necessary
                if len(history) == model.model_order:
                    history.pop()
                history.insert(0, symbol) ####
        return datares_rec

###对解码得到的0/1字符串进行以0为分割符号的list切割，并且统计每个子list中1的个数，输出原始对应的值，
###并且进行相关的数字反非负值处理，变成 inter residual

def data_convert_inverse_unary(datares_rec):


     ##按照数字0进行对list进行截断，并且统计每个sub_list里面1的个数，还原出真实的数字
    list_new=list_deal(datares_rec,0) #按照0进行切割
    #print(list_new)
    #print(len(list_new))

    true_ae_number=[]
    for subnum in range(len(list_new)-1):
        num=count_list(std=list_new[subnum],tongji=1)
        #print(num)
        true_ae_number.append(num)
    #print(true_ae_number)

    ##进行相关的恢复
    for i in range(len(true_ae_number)):
        true_ae_number[i] = true_ae_number[i]-1
    #print(true_ae_number)

    #把解码后的残差变会原先的数值 （0，1，2，3，4——》0，1，-1，2，-2)
    for ii in range(len(true_ae_number)):
        if true_ae_number[ii] ==0:
            true_ae_number[ii]=0
        elif  true_ae_number[ii] >0 and true_ae_number[ii] %2 ==0:
            true_ae_number[ii]=-(int(true_ae_number[ii]/2))
        else:
            true_ae_number[ii]=int((true_ae_number[ii]+1)/2)
    #print(true_ae_number)
    return true_ae_number

################################ 0-order exponential coding###############
## final decode: input: bin file; output: the 0/1 value
def final_decoder_expgolomb(inputfile,MODEL_ORDER = 0):
    # Must be at least -1 and match ppm-compress.py. Warning: Exponential memory usage at O(257^n).
    # Perform file decompression
    with open(inputfile, "rb") as inp:
        bitin = BitInputStream(inp)  #arithmeticcoding.

        dec = ArithmeticDecoder(256, bitin) ##############arithmeticcoding.
        model = PpmModel(MODEL_ORDER, 3, 2) #######ppmmodel.
        history = []
        
        datares_rec=[]
                    
        while True:
            symbol = decode_symbol(dec, model, history)
            if symbol ==2:
                break

            model.increment_contexts(history, symbol)
            datares_rec.append(symbol)
            if model.model_order >= 1:
                # Prepend current symbol, dropping oldest symbol if necessary
                if len(history) == model.model_order:
                    history.pop()
                history.insert(0, symbol) ####
        return datares_rec

###对0阶指数格伦布的编码方式，对这一串二进制字符串进行有效的切分
###并且进行相关的数字反非负值处理，变成 inter residual

def data_convert_inverse_expgolomb(datares_rec):


    ##按照0-order所定义的解码方式进行数字划分切割
    list_new= expgolomb_split(datares_rec)
    #print(list_new)
    #print(len(list_new))
    
    true_ae_number=[]
    for subnum in range(len(list_new)):
        num=exponential_golomb_decode(list_new[subnum])
        #print(num)
        true_ae_number.append(num)
    #print(true_ae_number)

    #把解码后的残差变会原先的数值 （0，1，2，3，4——》0，1，-1，2，-2)
    for ii in range(len(true_ae_number)):
        if true_ae_number[ii] ==0:
            true_ae_number[ii]=0
        elif  true_ae_number[ii] >0 and true_ae_number[ii] %2 ==0:
            true_ae_number[ii]=-(int(true_ae_number[ii]/2))
        else:
            true_ae_number[ii]=int((true_ae_number[ii]+1)/2)
    #print(true_ae_number)
    return true_ae_number










###对恢复出来的数字进行反inter-frame,变回真实的数字，并且按照原始的相关格式进行组合
###input:参照帧数值的路径+当前帧的数字；output: 真实数字帧数格式
def listformat(refframepath, true_ae_number ):
    
    length=16 ##############
    listnum=4  ###########
    slidelist=int(length/listnum)

    txt, i= {}, 1        
    with open(refframepath,'r+',encoding='utf-8') as f:
        for line in f:
            txt[i] = line
            i += 1
        dict={}
        refframevalue=txt[1]
        refframevalue=json.loads(refframevalue)
        refframevalue = eval('[%s]'%repr(refframevalue).replace('[', '').replace(']', ''))
        #print(refframevalue)

    ### 第二帧（帧间重建第一帧 based on VVC frame）    
    reallatentvalue=(np.array(refframevalue)+np.array(true_ae_number)).tolist()
    #print(reallatentvalue)

    #按照模型所需格式重组
    latentformat=[]
    for slideformat in range(0,slidelist):
        latentformatslide=reallatentvalue[slideformat*listnum:(slideformat+1)*listnum]
        latentformat.extend([latentformatslide])
    latentformat=[[latentformat]]
    #print(latentformat)

    ###保存恢复的真实的数据格式
    latentformat=str(latentformat)
    latentformat="".join(latentformat.split())
        
    return latentformat


def listformat_ori(refframetensor, true_ae_number ):
    
    length=16 ##############
    listnum=4  ###########
    slidelist=int(length/listnum)


    ### 第二帧（帧间重建第一帧 based on VVC frame）    
    reallatentvalue=(np.array(refframetensor)+np.array(true_ae_number)).tolist()
    #print(reallatentvalue)

    #按照模型所需格式重组
    latentformat=[]
    for slideformat in range(0,slidelist):
        latentformatslide=reallatentvalue[slideformat*listnum:(slideformat+1)*listnum]
        latentformat.extend([latentformatslide])
    latentformat=[[latentformat]]
    #print(latentformat)

    ###保存恢复的真实的数据格式
    latentformat=str(latentformat)
    latentformat="".join(latentformat.split())
        
    return latentformat