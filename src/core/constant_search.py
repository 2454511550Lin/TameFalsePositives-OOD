import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from math import log, sqrt
import math
import random

import logging

import sys
import os

import multiprocessing
from multiprocessing import Process

import pandas as pd

# append parent directory to path
sys.path.append('../')

dict = {}

# estimate a coin flip using lil-ucb bound
# compute the failure times for different constant

def lil_theory(t,delta,beta_t,p,lambda_max,lambda_min,eta):
    ct = 1-beta_t+beta_t/(p**2)
    try:
        result = sqrt( (3*ct/t) * (2*log(log(3*ct*t/2)) + 
                                   log((2/delta)*log((lambda_max-lambda_min)/eta)) ) )
        return result
    except ValueError:
        
        exit(-27)

def lil_heuristic(t, delta, c1, c2 ,beta_t=0.01,p=0.2,c3 = 1):
    ct = 1-beta_t+beta_t/(p**2)
    try:
        result = c1* sqrt( (ct/t) * (log(log(c2*ct*t)) + log(c3/delta) ) )
        return result
    except ValueError:
        print("t:{} delta:{} c1:{} c2:{} ct:{}".format(t, delta, c1, c2 ,ct))
        exit(-27)


# beta_t = (# of samples that are importantly sampled)/(all the ood samples)
# sampled from a bernoulli distribution with p
def get_failure_times(return_dict,p,T,n, delta, c1, c2):
    '''
    p: true mean
    T: number of samples for each trial
    n: number of trials
    delta: falure probability
    '''
    failures_list = []
    for i in range(n):
        data = np.random.binomial(1, p,size = T)
        # compute the empirical mean for each t in T
        empirical_means = np.cumsum(data) / np.arange(1, T+1)
        # compute the lil-ucb bound for each t in T
        lil_ucb = [lil_heuristic(t, delta,c1,c2) for t in range(1, T+1)]
        # calculate how many t that empirical_mean + lil_ucb > p (failure)
        failures = len(np.where(empirical_means + lil_ucb < p)[0]) + len(np.where(empirical_means - lil_ucb > p)[0])
        failures_list.append(failures)
    # compute the average failure times

    return_dict[(delta,c1,c2)] = np.mean(failures_list)
    return 

def par_run(overwrite=True):

    # set the parameters
    # delta from 0.01 to 0.2 with step 0.01
    #delta_lst = np.arange(0.01,0.21,0.01)
    delta_lst = [0.01, 0.05, 0.1, 0.2, 0.3, 0.4]
    c1_lst = np.arange(0.1,1, 0.1)
    c2_lst = np.arange(1.5,5, 0.25)
    seed = 2023

    trails = 50
    T = 1000
    pro= 0.5 # for coin mean
    dir = 'simulation/constant_search'

    batch_size = 100

    # set seed
    np.random.seed(seed)
    random.seed(seed)
    
    lst_p = []
    manager = multiprocessing.Manager()
    return_dict = manager.dict()
    for delta in delta_lst:
        for c1 in c1_lst:
            for c2 in c2_lst:
                p = Process(target=get_failure_times, args=(return_dict,pro,T,trails,delta,c1,c2))
                lst_p.append(p)
    
    # start with a batch size
    for i in range(0,len(lst_p),batch_size):
        batch = lst_p[i:i+batch_size]
        for p in batch:
            p.start()
        for p in batch:
            p.join()
        print('finished batch %s' % str(i))

    # save the dic to a .csv file 
    # each line is a specific delta and c1 followed by the averaged failure times 
    #print(return_dict)
    with open(os.path.join(dir, 'constant_search.csv'), 'w') as f:
        # write the header
        f.write('delta,c1,c2,failure_rate\n')
        for key, value in return_dict.items():
            delta,c1,c2,failure_rate = str(np.round(key[0],4)),\
                                       str(np.round(key[1],4)),\
                                       str(np.round(key[2],4)),\
                                       str(np.round(value/T,4))
            f.write('%s,%s,%s,%s\n' % (delta,c1,c2,failure_rate))
    
    # for each delta, plot the heatmap of failure times, with c1, c2 as x,y axis
    for delta in delta_lst:
        # read the csv file
        df = pd.read_csv(os.path.join(dir, 'constant_search.csv'))
        df = df[df['delta'] == delta]
        df = df.pivot(index='c1', columns='c2', values='failure_rate')
        # plot the heatmap
        sns.set(font_scale=0.6)
        plt.figure()
        sns.heatmap(df, annot=True, fmt='.2f', cmap = 'crest', cbar_kws={'label': 'failure rate'})
        # cmap = Blues
        plt.title('delta = %s' % delta)
        plt.savefig(os.path.join(dir, 'constant_search_delta_%s.png' % delta),bbox_inches='tight')
        plt.close()
if __name__ == '__main__':
    
    par_run()
   