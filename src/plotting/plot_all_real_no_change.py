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
                print(k)
            elif(f'method__{m}' in k and f'window__{window}' in k and f'gamma__{gamma}-' in k):
                mean_metrics.append(D[k]['mean_metrics']) 
                std_metrics.append(D[k]['std_metrics'])
                print(k)
    
    return mean_metrics,std_metrics



if __name__ == '__main__':

    #lst_id_ds =['cifar10']
    #lst_id_ds =['cifar100']
    lst_id_ds =['cifar10','cifar100']
    
    # if there is a command line argument, use it as the dataset
    if(len(sys.argv) > 1):
        lst_id_ds = [sys.argv[1]]
        
    print(" check the list of ID:  ",lst_id_ds)
         
    lst_scores = ['KNN','EBO', 'SSD','ODIN','MDS','VIM']
    #lst_id_ds = ['imagenet1k']
    #lst_scores = ['ASH','GRADNORM','REACT']

    lst_windows = [None]

    gamma = 0.2

    for id_ds in lst_id_ds:
        for score in lst_scores: 
            root_pfx = f'../../outputs/real-exp-supp/{id_ds}/{score}/'
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

                    print(id_ds, score, window, gamma, hoeffding)
                    print(len(mean_metrics),len(std_metrics))
                    
                    z_up = np.array(mean_metrics[0]['tpr'])*100 + np.array(std_metrics[0]['tpr'])*100
                    z_low = np.array(mean_metrics[3]['tpr'])*100 - np.array(std_metrics[3]['tpr'])*100
                    ylims= {}
                    n = len(z_up)
                    #z = z[1000:]
                    u =  np.sort(z_up)[ int(n*0.99)]
                    l = np.sort(z_low)[int(n*0.015)] #max(5, u-50)
                    l = max(0, l- l*0.05)
                    
                    if(score=='MDS'):
                        l = 0

                    ylims['11'] = [l,u+5]
                    ylims['01'] = [u+5,105]

                    z_up = np.array(mean_metrics[0]['fpr'])*100 + np.array(std_metrics[0]['fpr'])*100
                    u =  np.sort(z_up)[ int(n*0.99)]
                    ylims['10'] = [0,u]
                    #print(ylims)

                    sns.set_theme(style='white')
                    plot_with_y_break(methods, mean_metrics, std_metrics,legend=False,ylims=ylims)
                    #plt.tight_layout()

                    # check if the directory exists, if not create it
                    if(not os.path.exists(f'../../plots-supp/no-change-{id_ds}/')):
                        os.mkdir(f'../../plots-supp/no-change-{id_ds}/')

                    if(hoeffding):
                        if(not os.path.exists(f'../../plots-supp/no-change-{id_ds}/with-hoeffding/')):
                            os.mkdir(f'../../plots-supp/no-change-{id_ds}/with-hoeffding/')
                    else:
                        if(not os.path.exists(f'../../plots-supp/no-change-{id_ds}/without-hoeffding/')):
                            os.mkdir(f'../../plots-supp/no-change-{id_ds}/without-hoeffding/')

                    if(hoeffding):
                        plt.savefig(f'../../plots-supp/no-change-{id_ds}/with-hoeffding/real_{key}_plot_window_{window}_gamma_{gamma}_hfding.png',dpi=150,bbox_inches='tight')
                    else:
                        plt.savefig(f'../../plots-supp/no-change-{id_ds}/without-hoeffding/real_{key}_plot_window_{window}_gamma_{gamma}.png',dpi=150,bbox_inches='tight')


