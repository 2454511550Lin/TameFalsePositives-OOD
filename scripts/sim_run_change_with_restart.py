
root_dir = './'

import sys
sys.path.append(root_dir)

from omegaconf import OmegaConf
from src.utils.run_lib import *
#from src.utils.counting_utils import * 
from src.utils.conf_utils import *
import math


root_pfx  = "synthetic-exp-change-restart"

conf_dir  =  os.path.join(root_dir , "configs" ) 

base_conf = OmegaConf.load(os.path.join( conf_dir, f"base_config_synthetic_change_restart.yaml")) 

base_conf['output_root'] = os.path.join(root_dir, "outputs", root_pfx )
base_conf['root_dir']    = root_dir
base_conf['root_pfx']    = root_pfx


run_batch_size = 20
overwrite_flag = False  # False ==> don't overwrite, True ==> overwrite existing results 

run_confs      = False 

#dump_results   = True 
dump_results   = False 


# Root level config parameters

T                     = 10 # number of random seeds ( tirals)
lst_methods           = ['lil-heuristic','fpr_5','tpr_95']

lst_alpha            = [0.05]
lst_probs            = [0.2]
lst_delta            = [0.2]

lst_window           = [None,5000,10000,15000]#,5000,10000]

#lst_window           = [None,10000,15000,20000]


lst_seeds             = [i for i in range(T)] 

lst_gamma = [0.2]

lil_heuristic_params = {  
                          'seed'     : lst_seeds,
                          'alpha'    : lst_alpha,
                          'delta'    : lst_delta,
                          'method'   : ['lil-heuristic'],
                          'prob'     : lst_probs,
                          'window'   : lst_window,
                          'gamma'    : lst_gamma
                        }

no_ucb_params = {
                    "seed": lst_seeds, 
                    "alpha" : lst_alpha,
                    "method":['no-ucb'],
                    "window":lst_window,
                    "gamma": lst_gamma
                }

heoffding_params = {
                    "seed": lst_seeds, 
                    "alpha" : lst_alpha,
                    "method":['hoeffding'],
                    "window":lst_window,
                    "gamma": lst_gamma,
                    'delta'    : lst_delta
                }

tpr_95_params = {
                    "seed": lst_seeds, 
                    "method":['tpr_95']
                }

fpr_5_params = {
                    "seed": lst_seeds, 
                    "method":['fpr_5']
                }

d_method_params = {'no-ucb': no_ucb_params,'hoeffding':heoffding_params,
                   'lil-heuristic':lil_heuristic_params, 'tpr_95':tpr_95_params,'fpr_5':fpr_5_params}

if __name__ == "__main__":
    
    if(len(sys.argv)>1):
        mode = sys.argv[1]
        if(mode=="make_conf"):
            make_confs = True 
            run_confs  = False
            overwrite_flag= False
            dump_results = False

        elif(mode=='force_run'):
            make_confs = True 
            run_confs  = True 
            overwrite_flag= True
            dump_results = True 

        elif(mode=='run'):
            make_confs = True 
            run_confs  = True 
            overwrite_flag= False 
            dump_results = True 

        elif(mode=='save'):
            make_confs = False 
            run_confs  = False 
            dump_results = True 
        else:
            print('Specify mode: make_conf | force_run | run | save')
            exit()
    else:
        print('Specify mode: make_conf | force_run | run | save')
        exit()  
    
    
    if(make_confs or run_confs):
        lst_confs = []
        for m in lst_methods:
            lst_confs_1          = create_confs(base_conf, d_method_params[m])
            
            lst_confs.extend(lst_confs_1)

        print(f'Total Confs to run {len(lst_confs)}')

    if(run_confs):
        batched_par_run(lst_confs,batch_size=run_batch_size, overwrite=overwrite_flag) 
    