import numpy as np
import math
from collections import Counter
import itertools

###0-order
def get_digits(num):
    result = list(map(int,str(num)))
    return result

def exponential_golomb_encode(n):
    unarycode = ''
    golombCode =''
    ###Quotient and Remainder Calculation
    groupID = np.floor(np.log2(n+1))
    temp_=groupID
    #print(groupID)
    
    while temp_>0:
        unarycode = unarycode + '0'
        temp_ = temp_-1
    unarycode = unarycode#+'1'

    index_binary=bin(n+1).replace('0b','')
    golombCode = unarycode + index_binary
    return golombCode
        

### golombcode : 00100 real input[0,0,1,0,0]
def exponential_golomb_decode(golombcode):

    #golombcode=get_digits(golombcode)

    code_len=len(golombcode)

    ###Count the number of 1's followed by the first 0
    m= 0 ### 
    for i in range(code_len):
        if golombcode[i]==0:
            m=m+1
        else:
            ptr=i  ### first 0
            break

    offset=0
    for ii in range(ptr,code_len):
        #print(ii)
        num=golombcode[ii]
        #print(num)
        offset=offset+num*(2**(code_len-ii-1))
    #print(offset)
    #decodemum=2**m-1+offset
    decodemum=offset-1
    
    return decodemum


def expgolomb_split(expgolomb_bin_number):
    
    #x_list=get_digits(expgolomb_bin_number)
    #print(x_list)
    x_list=expgolomb_bin_number
    
    del(x_list[0]) ###去掉开始标识符，为了避免0无法写入
    x_len=len(x_list)
    
    sublist=[]
    while (len(x_list))>0:

        count_number=0
        i=0
        if x_list[i]==1:
            sublist.append(x_list[0:1])
            del(x_list[0])            
        else:
            num_times_zeros = [len(list(v)) for k, v in itertools.groupby(x_list)]
            count_number=count_number+num_times_zeros[0]
            sublist.append(x_list[0:(count_number*2+1)])
            del(x_list[0:(count_number*2+1)])
    return sublist

#number =220
#print(number)
#golombcode=exponential_golomb_encode(number)
#print(golombcode)

#decodemum=exponential_golomb_decode(golombcode)
#print(decodemum)

##



if __name__ == "__main__":

    x= 10100111100010110010000111101100101011 ###1 010 011 1 1 000101 1 00100
    
    x_list=get_digits(x)
    #print(x_list)

    
    del(x_list[0]) ###去掉开始标识符，为了避免0无法写入
    #print(x_list)
    x_len=len(x_list)
    #print(x_len)
    
    sublist=[]
    while (len(x_list))>0:

        count_number=0
        i=0
        if x_list[i]==1:
            sublist.append(x_list[0:1])
            del(x_list[0])            
            #print(x_list)
        else:
            num_times_zeros = [len(list(v)) for k, v in itertools.groupby(x_list)]
            #print(num_times_zeros[0])            

            count_number=count_number+num_times_zeros[0]
            #print(count_number)
            #print(x_list[0:(count_number*2+1)])

            sublist.append(x_list[0:(count_number*2+1)])
            
            del(x_list[0:(count_number*2+1)])
            #print(x_list)
            
    print(sublist)
    print(len(sublist))

    for i in range(len(sublist)):
        
        decodemum=exponential_golomb_decode(sublist[i])
        print(decodemum)


    

    




    


