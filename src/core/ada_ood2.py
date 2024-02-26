

import sys
import numpy as np
from scipy.stats import bernoulli

import logging
import bisect

# append parent directory to path
sys.path.append('../')

from .bounds import * 

class AdaOOD2:

    def __init__(self,conf):

        
        self.conf = conf
        self.num = conf.num
        
        self.method    = conf.method
        
        self.alpha = conf.alpha
        self.log_freq = 500

        self.tpr_k = conf.tpr_k

        logging.info(self.conf)
        
        
    
    # run one method (method means which confidence interval to use)
    def run(self, stream):

        '''
        stream: the data stream, could be merged stream as well.
        '''

        logging.info('Running Ada OOD with following params: ')
        logging.info(f' ucb : {self.method} \t  num : {self.num}  \t tpr_k : {self.tpr_k}')

        # get all the ood samples from data
        lst_ood_all = stream.lst_ood
        lst_id_all =  stream.lst_id

        self.r_min = np.quantile(lst_ood_all,0.5)
        self.r_max = np.max(lst_id_all)

        r_max = self.r_max 

        logging.info('r_min: {}, r_max: {}'.format(self.r_min, self.r_max))
        
        self.thr_lst = [] # threshold list
        self.ucb_lst = [] # phi(t)
        self.emfpr_lst = [] # empirical FPR list

        self.id_lst = [] # empirical id list
        
        num = self.conf.num

        # empirical ood list: [(s,i,t)], s is the score, i is 1 for imp samp 0 for not, t is the threshold
        cur_ood_lst = [] 
        cur_id_lst = []
        
        cur_ood_lst_window = []
        cur_id_lst_window  = []
        
        #prev_threshold = r_max # the last threshold

        self.beta_t_list = []
        self.i_lst = []
        self.l_lst = []

        self.n_id_lst = []
        self.n_ood_lst = []

        s0= stream.get_stream(0)

        if(self.tpr_k==0):
            lamda_k_tpr = s0.lamda_0tpr
        
        if(self.tpr_k==5):
            lamda_k_tpr = s0.lamda_5tpr

        if(self.tpr_k==10):
            lamda_k_tpr = s0.lamda_10tpr

        if(self.tpr_k==95):
            lamda_k_tpr = s0.lamda_95tpr
        
        if(self.tpr_k==90):
            lamda_k_tpr = s0.lamda_90tpr

        if(self.tpr_k==85):
            lamda_k_tpr = s0.lamda_85tpr

        for j,(s,y) in enumerate(stream):
                        
            log = j % self.log_freq == 0
            
            cur_n_id  = len(cur_id_lst)
            
            if(y==0):
                cur_ood_lst_window.append(s)
                cur_ood_lst.append(s)
            
            cur_n_ood_window = len(cur_ood_lst_window)

            if(cur_n_ood_window>num):
                cur_ood_lst_window.pop(0)

            if(y==1):
                cur_id_lst_window.append(s)
                cur_id_lst.append(s)

            cur_n_id_window  = len(cur_id_lst_window)
            if(cur_n_id_window>num):
                cur_id_lst_window.pop(0)
            
            cur_n_id_window  = len(cur_id_lst_window)
            cur_n_ood_window = len(cur_ood_lst_window)

            if(cur_n_ood_window==cur_n_id_window and cur_n_ood_window == num):

                cur_id_lst_window.sort()
                
                r = 1 - self.tpr_k/100
                lamda_k_tpr = cur_id_lst_window[int(cur_n_id_window*r)] 

                cur_id_lst_window = []
                cur_ood_lst_window = []
                logging.info(f'Updated threshold : {lamda_k_tpr}')
                


            if(log):
                logging.info(f'current threshold : {lamda_k_tpr}')
            
            self.thr_lst.append(lamda_k_tpr)
        
        result = {}

        result['thr'] = self.thr_lst

        return result
