import matplotlib.pyplot as plt 
import pickle 

# import src from the parent directory

root_dir = './'

import sys
sys.path.append(root_dir)


from src.utils.conf_utils import * 
from src.plotting.ybreak_plot import * 

from omegaconf import OmegaConf

from  src.utils.counting_utils import *

def get_metrics(D,methods,window,gamma):

    mean_metrics = []
    std_metrics  = [] 
    for m in methods:
        for k in D.keys():
            if( f'method__{m}' in k and m in ['tpr_95', 'fpr_5']):
                mean_metrics.append(D[k]['mean_metrics']) 
                std_metrics.append(D[k]['std_metrics'])
                #print(k)
            elif(f'method__{m}' in k and f'window__{window}' in k and f'gamma__{gamma}-' in k):
                mean_metrics.append(D[k]['mean_metrics']) 
                std_metrics.append(D[k]['std_metrics'])
                #print(k)
    
    return mean_metrics,std_metrics
if __name__ == '__main__':

    conf_dir  =  os.path.join(root_dir , "configs" ) 
    
    config_file = sys.argv[1]
    
    print(f'Loading config file {config_file}')
    
    # check if the config file exists
    if(not os.path.exists(os.path.join( conf_dir, config_file))):
        print(f'Config file {config_file} does not exist')
        exit()
    base_conf = OmegaConf.load(os.path.join( conf_dir, config_file))

    lst_id_ds = base_conf['id_ds']

    lst_scores = base_conf['scores']

    lst_windows = base_conf['window_sizes']

    gamma = base_conf['gamma'][0]
    
    output_root = os.path.join(root_dir, "outputs", base_conf['root_pfx'] )
    

    for id_ds in lst_id_ds:
        for score in lst_scores: 

            #root_pfx = f'../../outputs/real-exp-supp-change/{id_ds}/{score}/'
            root_pfx = os.path.join(output_root, id_ds, score)
            key = f"{id_ds}_{score}"
            #print(root_pfx)
            lst_outs = get_all_outs_for_exp(root_pfx)
            D = agg_on_seed(lst_outs)

            for window in lst_windows:
                methods = ['no-ucb','fpr_5', 'tpr_95', 'lil-heuristic','hoeffding']
                mean_metrics, std_metrics = get_metrics(D,methods, window, gamma)

                for hoeffding in [True, False]:
                    if(not hoeffding):
                        methods = ['no-ucb','fpr_5', 'tpr_95', 'lil-heuristic']

                    z_up = np.array(mean_metrics[0]['tpr'])*100 + np.array(std_metrics[0]['tpr'])*100
                    z_low = np.array(mean_metrics[3]['tpr'])*100 - np.array(std_metrics[3]['tpr'])*100
                    ylims= {}
                    n = len(z_up)
                    #z = z[1000:]
                    u =  np.sort(z_up)[ int(n*0.99)]
                    l = np.sort(z_low)[int(n*0.02)] #max(5, u-50)
                    
                    if base_conf['change']:
                        l = np.sort(z_low)[int(n*0.02)]
                    
                    #l = max(10,u-50) #max(5,min(l,u-30))
                    l = max(0, l- l*0.05)

                    if(score=='MDS'):
                        l = 0
                        
                    ylims['11'] = [l,u+5]
                    ylims['01'] = [u+5,105]

                    z_up = np.array(mean_metrics[3]['fpr'])*100 + np.array(std_metrics[3]['fpr'])*100
                    u =  np.sort(z_up)[ int(n*0.99)]
                    #
                    #fpr95_up = np.array(mean_metrics[2]['fpr'])*100 + np.array(std_metrics[2]['fpr'])*100
                    #u2 = np.sort(fpr95_up)[ int(n*0.99)]
                    #
                    ylims['10'] = [0,u]
                    #ylims['00'] = [u+2,u2+5]
                    #print(ylims)

                    sns.set_theme(style='white')
                    
                    if base_conf['change']:
                        plot_wo_y_break(methods, mean_metrics, std_metrics,legend=False,ylims=ylims)      
                    else: 
                        plot_wo_y_break(methods, mean_metrics, std_metrics,legend=False,ylims=ylims)
                        #plot_with_y_break(methods, mean_metrics, std_metrics,legend=False,ylims=ylims)
                    
                    change_str = 'change-' if base_conf['change'] else 'no-change-'
                    
                    temp_dir = os.path.join('./plots', f'{change_str}{id_ds}')
                    if(not os.path.exists( temp_dir)):
                        os.makedirs( temp_dir)

                    if(hoeffding):
                        temp_dir = os.path.join(temp_dir, 'with-hoeffding')        
                    else:
                        temp_dir = os.path.join(temp_dir, 'without-hoeffding')        
                    if(not os.path.exists( temp_dir)):
                        os.makedirs( temp_dir)

                    if(hoeffding):
                        temp_dir = os.path.join(temp_dir, f'{key}_plot_window_{window}_gamma_{gamma}_hfding.png')
                    else:
                        temp_dir = os.path.join(temp_dir, f'{key}_plot_window_{window}_gamma_{gamma}_no-hfding.png')
                    plt.savefig(temp_dir,dpi=150,bbox_inches='tight')

