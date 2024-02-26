import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

import sys
import os
import pickle

import matplotlib as mpl
import matplotlib.patches as mpatches
import logging
xlim_min = 1000
# plot the tpr, fpr, thr for one method 
def plot_one(mean_metric,ax,method,std_metric = None, figure_index = None, merge_win = False,up = False,m_name='fpr'):
    
    T = len(mean_metric)

    # color = https://www.color-hex.com/color-palette/45362
    markers_on = None
    #ls = '-.'
    ls = '-' #(0, (5, 5))
    alpha_line = 0.8
    alpha_shade = 0.25
    
    if method == 'no-ucb':
        method_label = 'No-UCB'
        marker = 'X'
        markers_begin = 20000
        mark_every = 30000
        
        mfc = color = '#5cb85c'
        ms = 15
        
    elif method == 'lil-heuristic':
        #method = 'Lil-heuristic (Our)'
        #method = 'LIL (Our)'
        marker = "d"
        markers_begin = 10000

        mark_every = 30000

        mfc = color =  '#f37735' #lightcoral
        #method_label = 'Lil-heuristic (Our)'
        method_label = 'LIL (Our)'
        ms = 15
        if merge_win:
            marker = None
            ls = '--'
            method_label = 'Lil-heuristic-window'
        

    elif method == 'hoeffding':
        method_label = 'Hoeffding'
        marker = 's' 
        
        markers_begin = 20000
        mark_every = 30000

        mfc = color = '#428bca'
        method_label = "Hoeffding"
        ms = 13
        alpha_line = 0.6
        if merge_win:
            marker = None
            ls = '-.-'
            method_label = 'Hoeffding-window'
        

    elif method == 'lil-theory':
        return
    
    elif method == 'fpr_5':
        ls = '-.'
        marker = '*'
        mfc = color = 'red'

        markers_begin = 10000
        mark_every = 30000

        a = 1
        ms = 15
        #method_label = '5% FPR'
        method_label = 'FPR-5%'

    elif method == 'tpr_95':
        marker = None #'o'
        ls = '-.'
        #method_label = '95%  TPR'
        method_label = 'TPR-95%'
        mfc = color =  'darkviolet' #'mediumvioletred' #'slategray' #'steelblue' #'#e2b93c'
        ms = 15 
        a=1 
        
    else:
        raise ValueError('Unknown method: {}'.format(method))
    
    if(marker):
        u = (T-markers_begin)//mark_every
        markers_on = [markers_begin + i* mark_every for i in range(u)]
    
    mask = (np.arange(T) >= xlim_min)
        
    sns.lineplot(x = np.arange(T)[mask], y = mean_metric[mask]*100, ax = ax, label = method_label,linewidth = 2,marker = marker,markersize=ms,color = color,markevery=markers_on,alpha=alpha_line,linestyle=ls)
    
    if method != 'tpr_95':
        # fill between the plus and minus std and mean
        ax.fill_between(np.arange(T)[mask], 100*(mean_metric - std_metric)[mask], 100*(mean_metric + std_metric)[mask],color=color, alpha=alpha_shade)

    if(method == 'lil-heuristic' and m_name=='fpr'):
        z = (mean_metric + std_metric)*100
        #print(z[50000:50200])
        
        z = z[50000:]
        t = np.argmax(z<=5)
        if(t>0):
            t = t+50000
            print(t)
            c = 'royalblue'
            arrow = mpatches.FancyArrowPatch((t, 8), (t, 5),
                                 mutation_scale=20, linewidth=1, facecolor=c, edgecolor=c)
            ax.add_patch(arrow)
            #ax.annotate("", xy=(t, 5), xytext=(t, 7), arrowprops=dict(arrowstyle="->"))

def plot_with_no_y_break(methods, metrics_fig, mean_metrics,std_metrics,legend=False,legend_fs=12):

    mpl.rcParams['axes.prop_cycle'] = mpl.cycler(color=['#d62728','#1f77b4', '#2ca02c', '#9467bd',  
                                                        '#ff7f0e','#8c564b', '#e377c2', '#7f7f7f', 
                                                        '#bcbd22', '#17becf']) 
    k = len(metrics_fig)
    fig, axes = plt.subplots(1,k,figsize=(6*k,5))
    fs=18
    
    def plot_all_methods(ax, m, mean_metrics, std_metrics):
        for i in range(len(methods)):
            plot_one(mean_metrics[i][m], ax=ax, method=methods[i], std_metric=std_metrics[i][m],m_name=m)
    
    for i,m in enumerate(metrics_fig):
        
        plot_all_methods(axes[i], m['name'], mean_metrics, std_metrics)
        x_tick_begin = 20000

        T = len(mean_metrics[i][m['name']])

        u = 5 #(T-x_tick_begin)//x_tick_every
        x_tick_every = (T)//u

        xticks  = [x_tick_begin + i* x_tick_every for i in range(u)]

        axes[i].set_xticks(xticks)

        xtick_labels = [ f"{x//1000}k" for x in xticks]

        axes[i].set_xticklabels(xtick_labels,fontsize=fs)

        axes[i].set_xlim([xlim_min,T])
        # set y ticks label font size
        axes[i].tick_params(labelsize=fs)

        axes[i].set_xlabel('time ($t$)',fontsize=fs)
        #ax[0].set_ylabel('TPR',fontsize=fs)
        axes[i].set_title(m['label'],fontsize=fs)
        if(m['ylim']):
            axes[i].set_ylim(m['ylim'])

        handles, labels = axes[i].get_legend_handles_labels()

        # find the label == '5\% FPR' and make it the first of the list
        idx = labels.index('FPR-5%')
        labels.insert(0, labels.pop(idx))
        handles.insert(0, handles.pop(idx))
    
        # remove legend from all subpots
        # get the total number of sample size:
        T = len(mean_metrics[0]) #sum(in_sizes) + sum(out_sizes)
    
        axes[i].legend([],[], frameon=False)


    if(legend):
        fig.legend(handles, labels, loc='upper center', ncol=6,bbox_to_anchor=(0.5,1.1), fontsize=legend_fs)
