import math
import numpy as np

def quantize(segment,val,orignal_interval):
    return math.floor(segment*val/orignal_interval)

def one_hot_vector(total_classes,index):
    vec= np.zeros((total_classes,1))
    vec[index]=1
    return vec 
def time_stamp_to_z(time_dict,total_length):
    x=np.zeros((total_length,1))
    mapping={"speech":1,"music":2}
    for stamp in time_dict:
        start=quantize(total_length,stamp["start"],10)
        end= quantize(total_length,stamp["end"],10)
        t=stamp["type"]
        x[start:end+1]=mapping[t]
    return x

        

if __name__=='__main__':
    time_dict=[{"start":0.69,"end":4.59,"type":"music","class":"speech_music"},{"start":7.09,"end":10,"type":"speech","class":"speech_music"}]

    print(time_stamp_to_z(time_dict,330))
    
    
    print(quantize(10,2.5514,5))