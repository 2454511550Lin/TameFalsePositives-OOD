import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

import sys
import os
import pickle

import matplotlib as mpl
import logging
xlim_min = 1000
# plot the tpr, fpr, thr for one method 
def plot_one(metric,ax,method,smetric_std = None):

    fpr,tpr,thr = np.array(metric['fpr']),np.array(metric['tpr']),np.array(metric['thr'])
    assert(len(fpr) == len(tpr) and len(tpr )== len(thr))
    T = len(fpr)

    # color = https://www.color-hex.com/color-palette/45362
    markers_on = None
    a = 0.7
    if method == 'no':
        method = 'No-UCB'
        marker = 'X'
        markers_on = [20000,40000,60000,80000]
        mfc = color = '#5cb85c'

    elif method == 'lil-heuristic':
        method = 'Lil-heuristic (Our)'
        marker = "d"
        markers_on = [10000,30000,50000,70000,90000]
        mfc = color =  '#f37735'

    elif method == 'hoeffding':
        method = 'Hoeffding'
        marker = 'p' 
        markers_on = [20000,40000,60000,80000]
        mfc = color = '#428bca'

    elif method == 'lil-theory':
        return
    elif method == '5\% FPR':
        marker = '*'
        mfc = color = 'r'
        markers_on = [10000,30000,50000,70000,90000]
        a = 1

    elif method == '$\lambda$ for 95\% TPR':
        marker = None
        mfc = color = '#e2b93c'
    else:
        raise ValueError('Unknown method: {}'.format(method))
    
    mask = (np.arange(T) >= xlim_min)

    if smetric_std != None and method != '$\lambda$ for 95\% TPR':
        # fill between the plus and minus std and mean
        ax[0].fill_between(np.arange(T)[mask], (tpr - smetric_std['tpr'])[mask], (tpr + smetric_std['tpr'])[mask],color=color, alpha=0.3)
        ax[1].fill_between(np.arange(T)[mask], (fpr - smetric_std['fpr'])[mask], (fpr + smetric_std['fpr'])[mask],color=color, alpha=0.3)
        ax[2].fill_between(np.arange(T)[mask], (thr - smetric_std['thr'])[mask], (thr + smetric_std['thr'])[mask],color=color, alpha=0.3)

    sns.lineplot(x = np.arange(T)[mask], y = tpr[mask], ax = ax[0], label = method,linewidth = 3,marker = marker,markersize=20,color = color,markevery=markers_on,alpha=a,linestyle='-')
    sns.lineplot(x = np.arange(T)[mask], y = fpr[mask], ax = ax[1], label = method,linewidth = 3,marker = marker,markersize=20,color = color,markevery=markers_on,alpha=a,linestyle='-')
    sns.lineplot(x = np.arange(T)[mask], y = thr[mask], ax = ax[2], label = method,linewidth = 3,marker = marker,markersize=20,color = color,markevery=markers_on,alpha=a,linestyle='-')
    
# plot the tpr, fpr, thr for all methods (by calling plot_one)
def plot_all(metrics,ax,methods,smetrics_std = None):

    sns.set_theme()
    sns.set_context("notebook")
    sns.set(font_scale=1.5)

    #font = {
    #    'weight' : 'bold',
    #    'size'   : 22}

    #matplotlib.rc('font', **font)

    num = len(methods)
    for i in range(num):
        if smetrics_std == None:
            plot_one(metrics[i],ax,methods[i])
        else:
            plot_one(metrics[i],ax,methods[i],smetrics_std[i])

def plot_one_run_shifted(dic,dic_std=None):
    
    # get things from dic
    metrics = dic['metrics']
    metrics_std = dic_std['metrics'] if dic_std != None else None

    methods = dic['methods']
    in_sizes = dic['in_sizes']
    out_sizes = dic['out_sizes']
    
    #sns.set_theme()
    #sns.set_context("notebook")
    mpl.rcParams['axes.prop_cycle'] = mpl.cycler(color=['#d62728','#1f77b4', '#2ca02c', '#9467bd',  
                                                        '#ff7f0e','#8c564b', '#e377c2', '#7f7f7f', 
                                                        '#bcbd22', '#17becf']) 

    fig,ax = plt.subplots(1,3,figsize=(20,5))

    plot_all(metrics,ax,methods,metrics_std)

    fs=18

    # set xticks for each ax
    for i in range(3):
        ax[i].set_xticks([20000,40000,60000,80000])
        ax[i].set_xticklabels(["20K", "40K", "60K", "80K"],fontsize=fs)
        ax[i].set_xlim([xlim_min,100000])
        # set y ticks label font size
        ax[i].tick_params(labelsize=fs)

        
    
    ax[0].set_xlabel('time($t$)',fontsize=fs)
    ax[0].set_ylabel('TPR',fontsize=fs)
    ax[0].set_title('TPR',fontsize=fs)

    ax[1].set_xlabel('time($t$)',fontsize=fs)
    ax[1].set_ylabel('FPR',fontsize=fs)
    ax[1].set_title('FPR',fontsize=fs)

    #sns.lineplot(x=np.arange(T)[mask],y=[lambda_star]*(T-xlim_min),label='5\% FPR',ax=ax[2], linestyle='--')
    ax[2].set_xlabel('time($t$)',fontsize=fs)
    ax[2].set_ylabel('$\lambda$',fontsize=fs)
    ax[2].set_title('Threshold $\lambda$',fontsize=fs)

    handles, labels = ax[0].get_legend_handles_labels()

    # find the label == '5\% FPR' and make it the first of the list
    idx = labels.index('5\% FPR')
    labels.insert(0, labels.pop(idx))
    handles.insert(0, handles.pop(idx))
    
    # remove legend from all subpots
    # get the total number of sample size:
    T =sum(in_sizes) + sum(out_sizes)
    for i in range(3):
        ax[i].legend([],[], frameon=False)
        ax[i].set_xlim(xlim_min,T-10)

    #fig.legend(handles, labels, loc='upper center', ncol=6,bbox_to_anchor=(0.5,1.1))
    
    return fig

def plot_one_run_average(path,seeds):
    # load the result for each seeds
    lst_dic = []
    for s in seeds:
        with open(os.path.join(path,'seed_{}/sim.pkl'.format(s)), 'rb') as f:
            dic = pickle.load(f)
        lst_dic.append(dic)

    num_methods = len(lst_dic[0]['metrics'])
    keys = lst_dic[0]['metrics'][0].keys()

    # create an empty list for each method
    metrics_lst = [{} for i in range(num_methods)]
    for i in range(num_methods):
        for k in keys:
            metrics_lst[i][k] =  []

    # add the data to the list
    for d in lst_dic:
        for i in range(num_methods):
            for k in keys:
                #print("key_{}-len_{}".format(k,len(d['metrics'][i][k])))
                #metrics_lst[i][k].append(np.array(d['metrics'][i][k])[:min_len])
                metrics_lst[i][k].append(np.array(d['metrics'][i][k]))

    # compute the average, and std of the lst_dic
    dic_mean,dic_std = {},{}
    dic_mean['metrics'] = [{} for i in range(num_methods)]
    dic_std['metrics'] = [{} for i in range(num_methods)]

    # compute the mean and std
    for i in range(num_methods):
        for k in keys:
            dic_mean['metrics'][i][k] = np.mean(np.array(metrics_lst[i][k]),axis=0)
            dic_std['metrics'][i][k] = np.std(np.array(metrics_lst[i][k]),axis=0)

    # copy the first dic and replace the metrics
    new_lst_dic = lst_dic[0].copy()
    new_lst_dic['metrics'] = dic_mean['metrics']
    # remove data and results from the new dic
    new_lst_dic.pop('data', None)
    new_lst_dic.pop('results', None)
    dic_mean = new_lst_dic
    
    fig = plot_one_run_shifted(dic_mean,dic_std)
    fig.savefig(os.path.join(path,'average.png'), dpi=200,bbox_inches='tight')
    plt.close(fig)

# the __main__ function
if __name__ == "__main__":
    #path = 'simulation/sim_result/delta_0.2_alpha_0.05_pro_0.2_#in_[5000, 5000]_#out_[50000, 50000]_indis_[[5.5, 4], [5.5, 4]]_outdis_[[-6, 4], [-5, 4]]_win_10000'
    #path = 'simulation/sim_result/delta_0.2_alpha_0.05_pro_0.2_#in_[5000, 5000]_#out_[50000, 50000]_indis_[[5.5, 4], [5.5, 4]]_outdis_[[-6, 4], [-5, 4]]_win_None'
    path = 'simulation/sim_result/delta_0.2_alpha_0.05_pro_0.2_#in_[10000]_#out_[100000]_indis_[[5.5, 4]]_outdis_[[-6, 4]]_win_10000'
    #path = 'simulation/sim_result/delta_0.2_alpha_0.05_pro_0.2_#in_[10000]_#out_[100000]_indis_[[5.5, 4]]_outdis_[[-6, 4]]_win_None'
    seeds = range(10)
    plot_one_run_average(path,seeds)