
from multiprocessing import Process
import time
import sys
import os
import pickle
import logging

# append parent directory to path
sys.path.append('../')

from src.data_layer.data_utils import * 
from src.utils.eval_utils import * 
from src.core.ada_ood import * 
from src.core.ada_ood2 import AdaOOD2

def run_fixed_thr_method(merged_stream, conf):
    result = {'thr':[],'ucb':[], 'n_ood':[]}
    metric = {'fpr':[], 'tpr':[], 'thr':[]}
    T = len(merged_stream)
    method = conf.method 
    for t in range(T):
        s = merged_stream.get_stream(t)
        if(method == 'tpr_95'):
            result['thr'].append(s.lamda_95tpr)
        elif(method == 'tpr_90'):
            result['thr'].append(s.lamda_90tpr)
        elif(method == 'tpr_85'):
            result['thr'].append(s.lamda_85tpr)
        elif(method == 'tpr_80'):
            result['thr'].append(s.lamda_80tpr)
        elif(method == 'fpr_5'):
            result['thr'].append(s.lambda_star)
        else:
            logging.error(f'Undefined fixed threshold method : {method}')
            logging.error('Exiting...')
            sys.exit()
    
    metrics = get_metrics_one(result,merged_stream,simluation = conf.data_conf.simulation)
    o = {'results': result, 'metrics' : metrics}
    return o 

def run_non_fixed_thr_method(merged_stream, conf):
    method = conf.method 
    assert method in ['no-ucb','hoeffding','lil-theory','lil-heuristic', 'adaood2']
    
    metrics = {'fpr':[], 'tpr':[], 'thr':[]}
    if(method=='adaood2'):
        adaood = AdaOOD2(conf)
        result = adaood.run(merged_stream)
    else:
        adaood = AdaOOD(conf)
        result = adaood.run(merged_stream)

    metrics = get_metrics_one(result,merged_stream,simluation = conf.data_conf.simulation)
    o = {'results': result, 'metrics' : metrics}
    return o         

def run_method(merged_stream, conf):
    method = conf.method 
    if(method in ['no-ucb','hoeffding','lil-theory','lil-heuristic','adaood2']):
        o = run_non_fixed_thr_method(merged_stream, conf)
    elif(method in ['tpr_95','tpr_90','tpr_85','tpr_80','fpr_5']):
        o = run_fixed_thr_method(merged_stream, conf)

    else:
        logging.error(f'Undefined method : {method}')
        logging.error('Exiting...')
        sys.exit()
    
    return o 

def run_conf(conf,overwrite=True,stdout=False):

    if(not overwrite):
        if(os.path.exists(conf['out_file_path'])):
            print(f"path exists {conf['out_file_path']}")
            return 
    try:
        os.makedirs(conf['run_dir'])
    except OSError:
        pass
    
    # set up the logging
    if(stdout):
        logging.basicConfig(stream=sys.stdout,
                        filemode = 'w+',
                        level=logging.INFO, 
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    else:         
        logging.basicConfig(filename=conf['log_file_path'],
                            filemode = 'w+',
                            level=logging.INFO, 
                            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    logging.info(f"setting seed {conf['seed']}")

    set_seed(conf['seed'])

    # get start time
    start_time = time.time()

    streams  = get_data_streams(conf)
    merged_stream = MergedStream(streams)
    T = len(merged_stream)
    logging.info('total number of samples: {}'.format(T))
    
    output = run_method(merged_stream,conf)


    end_time = time.time()
    logging.info('Time: {}'.format(end_time-start_time))
    
    output['conf']= conf 

    with open(conf['out_file_path'], 'wb') as out_file:
        pickle.dump(output, out_file, protocol=pickle.HIGHEST_PROTOCOL) 


def par_run(lst_confs,overwrite=True):
    lstP = []
    #print(len(lst_confs))
    for conf in lst_confs:
        #print(conf)
        #conf = copy.deepcopy(conf) # ensure no shit happens
        p = Process(target = run_conf, args=(conf,overwrite))
        p.start()
        
        lstP.append(p)
    for p in lstP:
        p.join()
    
def exclude_existing_confs(lst_confs):
    lst_out_confs = []
    for conf in lst_confs:
        path = conf["out_file_path"]
        if os.path.exists(path):
            print(f"path exists {conf['out_file_path']}")
        else:
            lst_out_confs.append(conf)
    return lst_out_confs


def batched_par_run(lst_confs,batch_size=2, lst_devices=['cpu'],overwrite=True):
    
    if(not overwrite):
        lst_confs = exclude_existing_confs(lst_confs)
        n = len(lst_confs)
        print(f'NUM confs to run : {n}')

    
    i=0
    n = len(lst_confs)
    total_time = 0
    big_bang_time = time.time()
    while(i<n):
        start = time.time()
        j = min(i+batch_size, n)
        print(f'running confs from {i} to {j} ')
        #for conf in lst_confs[i:i+batch_size]:
        #    print(conf['device'])
        par_run(lst_confs[i:j],overwrite)
        
        i = j 
        end = time.time()
        u = round((end-start)/60,2) # in minutes
        print( f"Time taken to run these confs : {u} minutes, ")

        total_time = round((end-big_bang_time)/60, 2) # in minutes
        avg_time   = round((total_time/i), 2)  # already in minutes
        r = round( max(0,(n - i))*avg_time, 2)

        print(f"Total confs run so far : {i}")
        print(f"Total time taken so far : {total_time//60} hours and {total_time%60 : .2f} minutes")
        print( f"Avg. Time taken to run confs so far : {avg_time//60} hours and {avg_time%60 :.2f} minutes, ")
        print(f"Remaining confs: {n-i} and estimated time to be taken : { r//60} hours and {r%60 :.2f} minutes")

def seq_run(lst_confs):
    for conf in lst_confs:
        run_conf(conf)

    


