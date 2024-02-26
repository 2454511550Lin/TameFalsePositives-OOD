# utils to find quantile and percentile

from scipy.stats import norm
import numpy as np
import pandas as pd
import random


def set_seed(s):
    random.seed(s)
    np.random.seed(s)

def find_quantile(dout, q):
    #return dout[int(len(dout)*q)]
    return np.quantile(dout,q)

def find_percentile(percentile, mean,std):
    return norm.ppf(percentile, loc=mean, scale=std)

def find_CDF(x, mean,std):
    # x is the random variable value
    return norm.cdf(x, loc=mean, scale=std)


''' not used

def stationary(gamma, din, dout, sample_size, seed = 0):
    # gamma: the ratio of din/dout
    # din: the distribution of ID
    # dout: the distribution of OD
    
    # randomly sample from list din and dout
    # the probability of sampling from din is gamma
    # the probability of sampling from dout is 1-gamma

    # set seed
    random.seed(seed)

    id,ood = list(din),list(dout)
    
    stream = []
    for i in range(sample_size):
        if random.random() < gamma:
            stream.append([id.pop(0),1])
        else:
            stream.append([ood.pop(0),0])
    
    return np.array(stream)


def shifted_sudden(gamma_0,gamma_1, din, dout, sample_size, shift_size, seed = 0):
    # gamma_0: the ratio of din/dout before shift
    # gamma_1: the ratio of din/dout after shift
    # din: the distribution of ID
    # dout: the distribution of OD
    # shift_size: the data point to start shifting
    
    random.seed(seed)

    stream = []
    for i in range(sample_size):
        if i < shift_size:
            if random.random() < gamma_0:
               stream.append([din[i],1])
            else:
                stream.append([dout[i],0])
        else:
            if random.random() < gamma_1:
                stream.append([din[i],1])
            else:
               stream.append([dout[i],0])
    return np.array(stream)

def shifted_grad(gamma_0,gamma_1, din, dout, sample_size, shift_size, shift_period,seed = 0):

    
    # gamma_0: the ratio of din/dout before shift
    # gamma_1: the ratio of din/dout after shift
    # din: the distribution of ID
    # dout: the distribution of OD
    # shift_size: the data point to start shifting
    # shift_period: the period of shifting

    random.seed(seed)

    stream = []
    gamma_lst = np.linspace(gamma_0,gamma_1,shift_period)
    for i in range(sample_size):
        if i < shift_size:
            if random.random() < gamma_0:
                stream.append([din[i],1])
            else:
                stream.append([dout[i],0])
        elif  i >= shift_size and i < shift_size + shift_period:
            if random.random() < gamma_lst[i-shift_size]:
                stream.append([din[i],1])
            else:
                stream.append([dout[i],0])
        else:
            if random.random() < gamma_1:
                stream.append([din[i],1])
            else:
                stream.append([dout[i],0])
    return np.array(stream)


    '''