

import sys
import numpy as np
from scipy.stats import bernoulli

import logging
import bisect

# append parent directory to path
sys.path.append('../')

from .bounds import * 

class AdaOOD:

    def __init__(self,conf):

        
        self.conf = conf
        self.window = conf.window
        self.prob   = conf.prob
        self.method    = conf.method
        self.delta  = conf.delta
        self.alpha = conf.alpha
        self.log_freq = 500

        self.detect_change     = conf.detect_change if 'detect_change' in conf else False 
        self.restart_on_change = conf.restart_on_change if 'restart_on_change' in conf else False 
        logging.info(self.conf)
        self.change_detected_at = []
        
    
    # run one method (method means which confidence interval to use)
    def run(self, stream):

        '''
        stream: the data stream, could be merged stream as well.
        '''

        logging.info('Running Ada OOD with following params: ')
        logging.info(f' ucb : {self.method} \t  window : {self.window} \t prob : {self.prob} \t delta : {self.delta} \t alpha : {self.alpha}')

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
        
        window = self.conf.window

        # empirical ood list: [(s,i,t)], s is the score, i is 1 for imp samp 0 for not, t is the threshold
        cur_ood_lst = [] 
        if window is not None:
            cur_ood_lst_window = []
        
        prev_threshold = r_max # the last threshold

        self.beta_t_list = []
        self.i_lst = []
        self.l_lst = []

        self.n_ood_lst = []

        cur_ood_lst_len = []
        cur_ood_lst_window_len = []

        for j,(s,y) in enumerate(stream):
            
            log = j % self.log_freq == 0

            l = 0 # 0 for not query, 1 for query
            i = 0 # 0 for sample w.p. 1, 1 for sample w.p. pro

            # first determine query or not
            # if score s is less than the previous threshold, then queried it w.p. 1
            if s <= prev_threshold:
                l = 1 
                i = 0
            else: # if score s is larger than the last threshold, then queried it w.p. pro
                # compute a bernoulli random variable with probability pro
                l = bernoulli.rvs(self.prob, size=1)
                i = 1

            self.l_lst.append(l)
            
            if y==0 and l==1: 
                # if the queried sample is out-of-distribution
                cur_n_ood = len(cur_ood_lst) # number of ood samples
                bisect.insort(cur_ood_lst,(s,i,cur_n_ood))
                if window is not None:
                    bisect.insort(cur_ood_lst_window,(s,i,cur_n_ood))

                    if len(cur_ood_lst_window) > window:
                        for k in range(len(cur_ood_lst_window)):
                            if cur_ood_lst_window[k][2] < len(cur_ood_lst) - window:
                                del cur_ood_lst_window[k]
                                break
                        try:
                            assert(len(cur_ood_lst_window) == window)
                        except:
                            logging.error(f't : {j}\t len(ood_lst_window) != window: {len(cur_ood_lst_window)} != {window}')
                            exit()
                
                self.i_lst.append(i)
            
            if y==1 and l==1:
                # if the queried sample is in-distribution
                bisect.insort( self.id_lst,s)
            
            if y==1 or l==0:
                # if not queried label or the sample is ID.
                # copy the last threshold and continue
                self.emfpr_lst.append(self.emfpr_lst[-1] if len(self.emfpr_lst)!=0 else 0)
                self.thr_lst.append(prev_threshold)
                self.ucb_lst.append(self.ucb_lst[-1] if len(self.ucb_lst)!=0 else 0)

                continue 
            
            # second compute the emprical fpr using dp
            assert(y == 0)# need to be ood sample
            assert(l == 1)# need to be a queried sample
            
            # given r_min and r_max, do a binary search
            lst_ood_avlbl = cur_ood_lst if window == None else cur_ood_lst_window
            n_ood_avlbl  = len(lst_ood_avlbl)

            ucb_val = self.get_ucb_val(n_ood_avlbl)


            feasible, cur_threshold,thr_fpr =  self.solve_for_lambda(lst_ood_avlbl, ucb_val, prev_threshold, self.r_min, r_max, steps=100)

            if(log):
                logging.info(f't : {j}\t Current n_ood_avlbl : {n_ood_avlbl} , ucb_val :{ucb_val}, FPR : {thr_fpr} ')
                if(feasible):
                    logging.info(f't : {j}\t Found feasible lambda : {cur_threshold} , and FPR :{thr_fpr}')
                else:
                    logging.info(f't : {j}\t Current lambda was not feasible.')
            
            if(feasible):
                prev_threshold = cur_threshold
            else:
                cur_threshold = prev_threshold
            
            index = bisect.bisect_left(lst_ood_avlbl, (cur_threshold, 0, 0))
            thr_fpr = sum([1 if i==0 else 1/self.prob for s,i,t in lst_ood_avlbl[index:]])/len(lst_ood_avlbl)
            
            if(log):
                logging.info(f't : {j}\t Current n_ood_avlbl : {n_ood_avlbl} , ucb_val :{ucb_val}, FPR : {thr_fpr} ')

            # if the previous threshold is not feasible in the currect ucb, then log the distribution shift
            # calculate the empirical FPR using the last threshold
            #if ucb_val < self.alpha and self.method != 'no' and window is not None:
             
            change_detected = False 
            if self.detect_change:
                index = bisect.bisect_left(lst_ood_avlbl, (prev_threshold, 0, 0))
            
                emfpr_tm1 = sum([1 if i==0 else 1/self.prob for s,i,t in lst_ood_avlbl[index:]])/len(lst_ood_avlbl)
                if(log):
                    logging.info('Inside Change Detection')
                    logging.info(f't : {j}\t Current n_ood_avlbl : {n_ood_avlbl} , ucb_val :{ucb_val}, FPR Prev Threshold : {emfpr_tm1} ')
                
                if emfpr_tm1 - ucb_val > self.alpha :
                    change_detected = True 
                    self.change_detected_at.append(j)

                    if(log):
                        logging.info(f"t : {j}\t **** Detected the distribution shift, number of oods = {cur_n_ood}")

                        logging.info(f't : {j}\t emfpr_tm1 : {emfpr_tm1}')
                        logging.info(f't : {j}\t sum : {emfpr_tm1 - ucb_val}')
                    
                        logging.info(f't : {j}\t Current n_ood_avlbl : {n_ood_avlbl} , and ucb_val :{ucb_val}')
            
            if(self.method in ["lil-heuristic", "lil-theory", "hoeffding"]):
                r_max = self.r_max if change_detected else prev_threshold

            
            if(change_detected and self.restart_on_change):
                cur_ood_lst = [] 
                if window is not None:
                    cur_ood_lst_window = []
                
                prev_threshold = self.r_max # the last threshold

                self.beta_t_list = []
                self.i_lst = []
                self.l_lst = []

                self.n_ood_lst = []

                cur_ood_lst_len = []
                cur_ood_lst_window_len = []
            

            self.n_ood_lst.append(cur_n_ood)
            self.emfpr_lst.append(thr_fpr)
            self.thr_lst.append(cur_threshold)
            self.ucb_lst.append(ucb_val)
        
        logging.info('total number of queries: {} out of {}'.format(sum(self.l_lst),len(self.l_lst)))
        logging.info('total number queried ood samples: {}'.format(len(cur_ood_lst)))
        logging.info('total number of queries w.p. {}: {} out of {}'.format(self.prob,sum(self.i_lst),len(self.l_lst)))

        result = {}
        result['ood_lst_len'] = cur_ood_lst_len
        result['ood_lst_window_len'] = cur_ood_lst_window_len
        result['n_ood'] = self.n_ood_lst
        result['thr'] = self.thr_lst
        result['ucb'] = self.ucb_lst
        result['emfpr'] = self.emfpr_lst
        result['id'] = self.id_lst
        result['ood'] = [s for s,i,t in cur_ood_lst]
        result['beta_t'] = self.beta_t_list
        result['change_detected_at'] = self.change_detected_at

        return result


    def solve_for_lambda(self,lst_ood, ucb_val,prev_threshold, r_min, r_max, steps=100):

        # find the smallest feasible threshold that emfpr + ucb <= alpha
        # use binary search
        
        index = bisect.bisect_left(lst_ood, (prev_threshold, 0, 0))
        emfpr_tm1 = sum([1 if i==0 else 1/self.prob for s,i,t in lst_ood[index:]])/len(lst_ood)
        if emfpr_tm1 + ucb_val <= self.alpha:
            r_max = prev_threshold

        thr_fpr = 1.0
        thr_last   = r_max
        
        temp_r_min = r_min
        temp_r_max = r_max

        #print(r_min, r_max)

        feasible = False 
        # an indicator for whether there is a feasible threshold

        for j in range(steps):
            
            thr = (temp_r_min + temp_r_max) /2
            
            # compute the emfpr
            index = bisect.bisect_left(lst_ood, (thr, 0, 0))
            emfpr = sum([1 if i==0 else 1/self.prob for s,i,t in lst_ood[index:]])/len(lst_ood)
            
            #print(temp_r_min,temp_r_max,thr, emfpr , emfpr + ucb_val)

            #logging.info('r_min: {} r_max: {} thr: {} emfpr: {} ucb: {}'.format(r_min,r_max,thr,emfpr,ucb))

            if emfpr + ucb_val <= self.alpha :
                temp_r_max = thr
                thr_last = thr
                thr_fpr = emfpr
                feasible = True
            else:
                temp_r_min = thr

        return feasible, thr_last, thr_fpr 

    def get_ucb_val(self,n_ood):

        # third compute the confidence interval
        ucb_val = 0.0
        if self.method == 'lil-theory':
            beta_t = sum(self.i_lst)/n_ood
            self.beta_t_list.append(beta_t)
            ucb_val = lil_theory(n_ood, self.delta, beta_t, self.prob, self.r_max, self.r_min)

        elif self.method == 'lil-heuristic':
            beta_t = sum(self.i_lst)/n_ood
            self.beta_t_list.append(beta_t)
            ucb_val = lil_heuristic(n_ood, self.delta, beta_t,self.prob)

        elif self.method == 'hoeffding':
            ucb_val = hoeffding(n_ood, self.delta)

        elif self.method == 'no-ucb':
            ucb_val = 0

        else:
            logging.error(f'ucb not found: {self.method}')
            sys.exit()

        return ucb_val 
        