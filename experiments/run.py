
root_dir = './'

import sys
sys.path.append(root_dir)

from omegaconf import OmegaConf
from src.utils.run_lib import *
#from src.utils.counting_utils import * 
from src.utils.conf_utils import *
import math


if __name__ == "__main__":

    conf_dir  =  os.path.join(root_dir , "configs" ) 

    #base_conf = OmegaConf.load(os.path.join( conf_dir, f"cifar10_nochange.yaml")) 
    
    # load in the base config file using the cmd line arg
    
    config_file = sys.argv[2]
    
    print(f'Loading config file {config_file}')
    
    # check if the config file exists
    if(not os.path.exists(os.path.join( conf_dir, config_file))):
        print(f'Config file {config_file} does not exist')
        exit()
    base_conf = OmegaConf.load(os.path.join( conf_dir, config_file))

    #root_pfx  = "real"

    base_conf['output_root'] = os.path.join(root_dir, "outputs", base_conf['root_pfx'] )
    base_conf['root_dir']    = root_dir
    #base_conf['root_pfx']    = root_pfx

    overwrite_flag = False  # False ==> don't overwrite, True ==> overwrite existing results 

    run_confs      = False 

    #dump_results   = True 
    dump_results   = False 


    # Root level config parameters


    T = base_conf['T']
    lst_seeds  = [i for i in range(T)] 
    lst_methods = base_conf['methods']
    lst_scores = base_conf['scores']
    lst_id_ds = base_conf['id_ds']
    lst_alpha = base_conf['alphas']
    lst_probs = base_conf['probs']
    lst_window = base_conf['window_sizes']
    lst_delta = base_conf['deltas']
    lst_gamma = base_conf['gamma']
    run_batch_size = base_conf['run_batch_size']
    detect_change = base_conf['detect_change']
    restart = base_conf['restart']

    lil_heuristic_params = {  
                              'seed'     : lst_seeds,
                              'alpha'    : lst_alpha,
                              'delta'    : lst_delta,
                              'method'   : ['lil-heuristic'],
                              'prob'     : lst_probs,
                              'window'   : lst_window,
                              'gamma'    : lst_gamma,
                              'score' : lst_scores,
                              'id_ds': lst_id_ds
                            }

    no_ucb_params = {
                        "seed": lst_seeds, 
                        "alpha" : lst_alpha,
                        "method":['no-ucb'],
                        "window":lst_window,
                        "gamma": lst_gamma,
                        'score' : lst_scores,
                        'id_ds': lst_id_ds
                    }

    heoffding_params = {
                        "seed": lst_seeds, 
                        "alpha" : lst_alpha,
                        "method":['hoeffding'],
                        "window":lst_window,
                        "gamma": lst_gamma,
                        'delta'    : lst_delta,
                        'score' : lst_scores,
                        'id_ds': lst_id_ds
                    }

    tpr_95_params = {
                        "seed": lst_seeds, 
                        "method":['tpr_95'],
                        'score' : lst_scores,
                        'id_ds': lst_id_ds
                    }

    fpr_5_params = {
                        "seed": lst_seeds, 
                        "method":['fpr_5'],
                        'score' : lst_scores,
                        'id_ds': lst_id_ds
                    }

    d_method_params = {'no-ucb': no_ucb_params,'hoeffding':heoffding_params,
                       'lil-heuristic':lil_heuristic_params, 'tpr_95':tpr_95_params,'fpr_5':fpr_5_params}


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