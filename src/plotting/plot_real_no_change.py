import matplotlib.pyplot as plt 
import pickle 
import sys 
sys.path.append('../')
sys.path.append('../../')
from src.utils.conf_utils import * 
from src.plotting.ybreak_plot import * 

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
                print(k)
    
    return mean_metrics,std_metrics

score = 'EBO'
id_ds = 'cifar10'

root_pfx = f'../../outputs/real-exp-supp/{id_ds}/{score}/'
key = f"{id_ds}_{score}"
lst_outs = get_all_outs_for_exp(root_pfx)
D = agg_on_seed(lst_outs)





lst_windows = [None,5000,10000,15000]

gamma = 0.2


for window in lst_windows:
    for hoeffding in [True, False]:
        if(hoeffding):
            methods = ['no-ucb','fpr_5', 'tpr_95', 'lil-heuristic','hoeffding']
        else:
            methods = ['no-ucb','fpr_5', 'tpr_95', 'lil-heuristic']#,'hoeffding']
        mean_metrics, std_metrics = get_metrics(D,methods, window, gamma)
        z = np.array(mean_metrics[0]['tpr'])*100 + np.array(std_metrics[0]['tpr'])*100
        ylims= {}
        n = len(z)
        #z = z[1000:]
        u =  np.sort(z)[ int(n*0.99)]
        l = 5
        ylims['11'] = [l,u+5]
        ylims['01'] = [u+5,105]
        print(ylims)

        sns.set_theme(style='white')
        plot_with_y_break(methods, mean_metrics, std_metrics,legend=False,ylims=ylims)
        #plt.tight_layout()
        if(hoeffding):
            plt.savefig(f'../../plots-supp/real_{key}_plot_window_{window}_gamma_{gamma}_hfding.png',dpi=150,bbox_inches='tight')
        else:
            plt.savefig(f'../../plots-supp/real_{key}_plot_window_{window}_gamma_{gamma}.png',dpi=150,bbox_inches='tight')

