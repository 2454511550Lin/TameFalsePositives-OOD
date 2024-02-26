from sklearn.metrics import accuracy_score

import os 
import pickle 
import pandas as pd 
import numpy as np 

from collections import defaultdict 
import copy 

from collections import defaultdict 
from datetime import datetime 
import itertools

from .conf_utils import * 

def load_pkl_file(fpath):
    with open(fpath, 'rb') as handle:
        o = pickle.load(handle)
    return o 

def get_all_outs_for_exp(root_pfx):

    lst_files = [os.path.join(dp, f) for dp, dn, fn in os.walk(root_pfx) for f in fn]
    lst_out_files = [f  for f in lst_files if f[-3:]=='pkl'] 
    print(f'Total output pkl files read : {len(lst_out_files)}')

    lst_outs = []
    for fpath in lst_out_files:
        #print(fpath)
        out = load_pkl_file(fpath)
        params =fpath[len(root_pfx):]
        #print(params)
        p = {}
        #print("Number of the / in the key : ", params.split("/")[1:])
        for x in params.split("/")[1:-1]:
            #print("The key to be split is : ", x)
            if(x.startswith('stream')): 
                s_conf = streams_key_to_conf(x[8:])
                u = 'streams'
                v = x[8:]
            else:
                u,v = x.split('__')
            p[u]  = v

        out.update(p)
        lst_outs.append(out)
        #lst_outs.append(out_)
    
    print(f'total outs read : {len(lst_outs)}')
    return lst_outs 

def filter_outputs(lst_outs,param_f):
    filtered_outs = []
    for out in lst_outs:
        flag = True 
        for k in param_f.keys():
            flag = flag and (k in out) and (out[k]==str(param_f[k]))
        if(flag):
            filtered_outs.append(out)
    return filtered_outs

def filter_outputs_2(df,param_f):
    query = ' & '.join([ str(param)+ '==' + "'"+str(param_f[param])+"'" for param in param_f.keys()])
    return df.query(query)

def agg_on_seed(lst_outs,lst_eta = []):

    def get_agg(sub_lst_outs):
        metrics = {'fpr': [], 'tpr':[], 'thr':[],'change_detected_at':[], 
                   'feasible_at':[],
                   'eta_opt':{}}
        
        for eta in lst_eta: 
            metrics['eta_opt'][f'{eta}'] = [] 

        for o in sub_lst_outs:
            for k in o['metrics'].keys():
                metrics[k].append(o['metrics'][k])

            if('change_detected_at' in o['results']):
                metrics['change_detected_at'].append(o['results']['change_detected_at'])
            
            t_f = np.argmax(np.array(o['metrics']['fpr'])>1e-5)
            metrics['feasible_at'].append(t_f)

            for eta in lst_eta: 
                t_eta_opt = np.argmax(np.array(o['metrics']['fpr'])> 0.05-eta)
                if(t_eta_opt>0):
                    #metrics['feasible_at'].append(t_f)
                    metrics['eta_opt'][f'{eta}'].append(t_eta_opt)

        
        o['mean_metrics'] = {}
        o['std_metrics'] = {}
        o['change_detected_at'] = {}
        o['feasible_at'] = {}

        keys = ['fpr','tpr','thr']
        for k in keys:
            o['mean_metrics'][k] = np.mean(metrics[k], axis = 0 )
            o['std_metrics'][k]  = np.std(metrics[k], axis = 0 )
            
        o['change_detected_at'] = metrics['change_detected_at']
        
        o['feasible_at']        = metrics['feasible_at']
        o['feasible_at_mean']   = np.mean(metrics['feasible_at'])
        o['feasible_at_std']    = np.std(metrics['feasible_at'])

        o['eta_opt_at']   = metrics['eta_opt']
        o['eta_opt_mean'] =  {}
        o['eta_opt_std']  =  {}
        for eta in lst_eta:
            o['eta_opt_mean'][f'{eta}'] = np.mean(metrics['eta_opt'][f'{eta}'])
            o['eta_opt_std'][f'{eta}'] = np.std(metrics['eta_opt'][f'{eta}'])

        o['num_runs']               = len(sub_lst_outs)
        
        return o 

    D = defaultdict(list)
    D_ = {}
    
    keys = ['streams','method','prob','window','delta','alpha','num','tpr_k']

    for o in lst_outs:
        
        out_key = "##".join([f"{k}__{o[k]}" for k in keys if k in o ])
        print(out_key)
        D[out_key].append(o)
        
        D_[out_key] = dict([(k,o[k]) for k in keys if k in o ])

    #print(D.keys())

    D2 = {}
    for k in D.keys():
        D2[k] = get_agg(D[k])
    
    return D2 


def plot_results(root_pfx,outputs_path,keys):

    lst_outs = get_all_outs_for_exp(outputs_path)
    