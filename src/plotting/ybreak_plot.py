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
xlim_min = 2000
# plot the tpr, fpr, thr for one method 
def plot_one(metric,ax,method,smetric_std = None, figure_index = None, merge_win = False,up = False,ylims=None):

    fpr,tpr,thr = np.array(metric['fpr']),np.array(metric['tpr']),np.array(metric['thr'])
    assert(len(fpr) == len(tpr) and len(tpr )== len(thr))
    T = len(fpr)

    if figure_index == 1:
        if method == '$\lambda$ for 95\% TPR':
            return
        
    if merge_win:
        # only plot elif 'lil-heuristic'and 'hoeffding'
        if method not in ['lil-heuristic','hoeffding']:
            return

    # color = https://www.color-hex.com/color-palette/45362
    markers_on = None
    ls = '-'
    a = 0.7
    label = 'No-UCB'

    alpha_line = 0.8
    alpha_shade = 0.25

    if method == 'adaood2_5_200':
        label = 'K=5, N=200'

        marker = "P"
        markers_on = [30000,70000,110000,145000]
        ms = 15
        alpha_shade = 0.2

        mfc = color = 'darkviolet'

    elif method == 'adaood2_5_2000':
        label = 'K=5, N=2000'

        marker = "X"
        markers_on = [10000,50000,90000,130000]
        ms = 15
        alpha_shade = 0.2

        mfc = color = 'slategray'

    elif method == 'adaood2_95_200':
        label = 'K=95,N=200'

        marker = "s"
        markers_on = [10000,50000,90000,130000]
        ms = 15
        alpha_shade = 0.2

        mfc = color = 'violet'

    elif method == 'adaood2_95_2000':
        label = 'K=95,N=2000'

        marker = "o"
        markers_on = [30000,70000,110000,145000]
        ms = 15
        alpha_shade = 0.2

        mfc = color = 'gray'

    elif method == 'no-ucb':
        label = 'No-UCB'
        marker = "4"
        markers_on = [30000,70000,110000,145000]
        ms = 15
        alpha_shade = 0.4

        mfc = color = '#5cb85c'
        #if up == True:
        #    return

    elif method == 'lil-heuristic':
        label = 'LIL (Our)'
        marker = "d"
        markers_on = [10000,50000,90000,130000]
        mfc = color =  '#f37735'
        ms = 15

        if merge_win:
            marker = None
            ls = '--'
            method = 'Lil-heuristic-window'
        #if up == True:
        #    return

    elif method == 'hoeffding':
        label = 'Hoeffding'
        marker = 's' 
        markers_on = [20000,60000,100000,140000]
        mfc = color = '#428bca'
        ms = 13

        if merge_win:
            marker = None
            ls = '--'
            method = 'Hoeffding-window'
        #if up == True:
        #    return

    elif method == 'lil-theory':
        return
    
    elif method == 'fpr_5':
        ls = "-."
        label = "FPR-5%"
        marker = '*'
        mfc = color = 'r'
        markers_on = [40000,80000,120000]
        a = 1
        ms = 15

        #if up == True:
        #    return

    elif method == 'tpr_95':
        ls = "-."
        label = "TPR-95%"
        marker = '>'
        #mfc = color = '#e2b93c'
        mfc = color = "darkviolet"
        ms = 13
        markers_on = [20000,60000,100000,140000]
        #if up == False:
        #    return
        
    elif method == 'tpr_90':
        ls = "-."
        label = "TPR-90%"
        marker = '<'
        #mfc = color = '#e2b93c'
        mfc = color = "purple"
        ms = 13
        markers_on = [20000,60000,100000,140000]
        #if up == False:
        #    return
        
    elif method == 'tpr_85':
        ls = "-."
        label = "TPR-85%"
        marker = '^'
        #mfc = color = '#e2b93c'
        mfc = color = "mediumorchid"
        markers_on = [20000,60000,100000,140000]
        ms = 13

        #if up == False:
        #    return

    elif method == 'tpr_80':
        ls = "-."
        label = "TPR-80%"
        marker = 'v'
        #mfc = color = '#e2b93c'
        mfc = color = "mediumpurple"
        markers_on = [20000,60000,100000,140000]
        ms = 13

        #if up == False:
        #    return
    else:
        raise ValueError('Unknown method: {}'.format(method))
    
    mask = (np.arange(T) >= xlim_min)

    if figure_index == 0:
        #print(method)
        #print(tpr[:10])
        #print(fpr[:10])
        sns.lineplot(x = np.arange(T)[mask], y = tpr[mask]*100, ax = ax[1], label = label,linewidth = 2,
                     marker = marker,markersize=ms,color = color,markevery=markers_on,alpha=alpha_line,linestyle=ls)
        
        sns.lineplot(x = np.arange(T)[mask], y = fpr[mask]*100, ax = ax[0], label = label,linewidth = 2,
                     marker = marker,markersize=ms,color = color,markevery=markers_on,alpha=alpha_line,linestyle=ls)
       
        if smetric_std != None and method not in ['tpr_95','tpr_90','tpr_85','tpr_80']:
            # fill between the plus and minus std and mean
            ax[1].fill_between(np.arange(T)[mask], (tpr - smetric_std['tpr'])[mask]*100, (tpr + smetric_std['tpr'])[mask]*100,color=color, alpha=alpha_shade)
            ax[0].fill_between(np.arange(T)[mask], (fpr - smetric_std['fpr'])[mask]*100, (fpr + smetric_std['fpr'])[mask]*100,color=color, alpha=alpha_shade)
        

        if(method == 'lil-heuristic'):
            z = (fpr + smetric_std['fpr'])*100
            #print(z[50000:50200])
            
            z = z[50000:]
            slack = 5e-2
            t = np.argmax(z<=5 + slack)
            if(t>0):
                t = t+50000
                #print(t)
                c = 'royalblue'
                h = ylims['10']
                if(h[1]<10):
                    d = 2.5 #int((h[1]-h[0])*0.3)
                elif(h[1]<20):
                    d = int((h[1]-h[0])*0.3)
                else:
                    d = int((h[1]-h[0])*0.35)

                #print(d)
                arrow = mpatches.FancyArrowPatch((t, d+5), (t, 5),
                                    mutation_scale=20, linewidth=1, facecolor=c, edgecolor=c)
                ax[0].add_patch(arrow)
                #ax.annotate("", xy=(t, 5), xytext=(t, 7), arrowprops=dict(arrowstyle="->"))
    
    if figure_index == 1:
        sns.lineplot(x = np.arange(T)[mask], y = thr[mask], ax = ax, label = label,linewidth = 3,marker = marker,markersize=15,color = color,markevery=markers_on,alpha=a,linestyle=ls)
        if smetric_std != None and method != 'tpr_95':
            ax.fill_between(np.arange(T)[mask], (thr - smetric_std['thr'])[mask], (thr + smetric_std['thr'])[mask],color=color, alpha=0.3)
        
# plot the tpr, fpr, thr for all methods (by calling plot_one)
def plot_all(metrics,ax,methods,metrics_std = None, figure_index = None, merge_win = False,up = False,ylims=None):

    sns.set_theme()
    sns.set_context("notebook")
    #sns.set(font_scale=1.5)

    num = len(methods)
    for i in range(num):
        if metrics_std is None:
            plot_one(metrics[i],ax,methods[i],figure_index, merge_win = merge_win,up = up,ylims=ylims)
        else:
            #print('i_{}_mean_{}_std_{}_methods_{}'.format(i,len(metrics),len(metrics_std),len(methods)))
            plot_one(metrics[i],ax,methods[i],metrics_std[i],figure_index, merge_win = merge_win,up = up,ylims=ylims)

def plot_with_y_break(methods, mean_metrics,std_metrics,legend=False,legend_fs=12,ylims=None):
    
    #sns.set_theme()
    #sns.set_context("notebook")
    mpl.rcParams['axes.prop_cycle'] = mpl.cycler(color=['#d62728','#1f77b4', '#2ca02c', '#9467bd',  
                                                        '#ff7f0e','#8c564b', '#e377c2', '#7f7f7f', 
                                                        '#bcbd22', '#17becf']) 
    fig1,ax1 = plt.subplots(2,2,figsize=(12,5),gridspec_kw={'height_ratios': [1, 4]})

    # add six subplots

    fs=18

    # set xticks for each ax
    for i in range(2):
        # set y ticks label font size
        ax1[1][i].tick_params(labelsize=fs)
        ax1[0][i].tick_params(labelsize=fs)
        ax1[0][i].spines.bottom.set_visible(False)
        ax1[1][i].spines.top.set_visible(False)
        ax1[0][i].xaxis.tick_top()
        ax1[0][i].tick_params(labeltop=False)  # don't put tick labels at the top
        ax1[1][i].xaxis.tick_bottom()
        ax1[1][i].set_xticks([20000,50000,80000,110000,140000])
        ax1[1][i].set_xticklabels(["20k","50k", "80k","110k", "140k"],fontsize=fs)

        ax1[1][i].set_xlim([xlim_min,150000])
        ax1[0][i].set_xlim([xlim_min,150000])

        # set the y lim to be start from 0.001
        # but dont set the upper limit

                


    # 1. for fpr
    # # make the y ticks sparse
    

    #ax1[0][0].set_ylim([0.9,1])
    #ax1[1][0].set_ylim([0.59,0.85])
    ## set y ticks 
    #ax1[0][0].set_yticks([0.95])
    #ax1[1][0].set_yticks([0.6,0.7,0.8])


    ## 2. for tpr
    #print(ylims)

    if(ylims):
        ax1[1][1].set_ylim(ylims['11'])
        ax1[0][1].set_ylim(ylims['01'])
        
        #ax1[0][0].set_ylim(ylims['00'])
        #ax1[1][0].set_ylim(ylims['10'])
        

    else:
        ax1[1][1].set_ylim([60.0,88])

        ax1[0][1].set_ylim([88,100])
        
        ax1[1][1].set_yticks(np.arange(60,88,5))

        ax1[0][0].set_ylim([12,50])
        ax1[1][0].set_ylim([0,10])
    
    two = True 
    if(two):
        ax1[1][1].set_ylim([-1,85])

        ax1[0][1].set_ylim([90,100])
        
        ax1[1][1].set_yticks(np.arange(0,85,20))

        ax1[0][0].set_ylim([12,60])
        ax1[1][0].set_ylim([-1,12])
    ## set y ticks 
    #ax1[0][1].set_yticks([45])
    #ax1[1][1].set_yticks([1,5,9,13])
#
    ## 3. for thr
    #ax2[0].set_ylim([-0.07,-0.0])
    #ax2[1].set_ylim([-0.17,-0.2])
    ## set y ticks 
    #ax2[0].set_yticks([-0.05,-0.03])
    #ax2[1].set_yticks([-0.18]) 

    def plot_diagnol(ax1,ax2,transAxes):
    
        d = .02  # how big to make the diagonal lines in axes coordinates
        # arguments to pass to plot, just so we don't keep repeating them
        kwargs = dict(transform=transAxes[0], color='k', clip_on=False)
        ax1.plot((-d, +d), (0, 0), **kwargs)    # top-left diagonal
        ax1.plot((1 - d, 1 + d), (0, 0), **kwargs)  # top-right diagonal

        kwargs.update(transform=transAxes[1])  # switch to the bottom axes
        ax2.plot((-d, +d), (1, 1), **kwargs)  # bottom-left diagonal
        ax2.plot((1 - d, 1 + d), (1, 1), **kwargs)  # bottom-right diagonal
    
    # plot diagnol lines
    plot_diagnol(ax1[0][0],ax1[1][0],transAxes=[ax1[0][0].transAxes,ax1[1][0].transAxes])
    plot_diagnol(ax1[0][1],ax1[1][1],transAxes=[ax1[0][1].transAxes,ax1[1][1].transAxes])

    fig1.subplots_adjust(hspace=0.05)
    plot_all(mean_metrics,ax1[0],methods,std_metrics,figure_index=0,up=True,ylims=ylims)
    plot_all(mean_metrics,ax1[1],methods,std_metrics,figure_index=0,up=True,ylims=ylims)

    # set the fpr, tpr, and threshold
    ax1[1][0].set_xlabel('time($t$)',fontsize=fs)
    #ax1[1][0].set_ylabel('TPR',fontsize=fs,loc=loc)
    ax1[0][0].set_title('FPR(%)',fontsize=fs)
    ax1[1][1].set_xlabel('time($t$)',fontsize=fs)
    #ax1[1][1].set_ylabel('FPR',fontsize=fs,loc=loc)
    ax1[0][1].set_title('TPR(%)',fontsize=fs)


    # TODO: we may want to add a legend for 95% TPR 
    handles, labels = ax1[1][0].get_legend_handles_labels()    

    ## find the label == '5\% FPR' and make it the first of the list
    #idx = labels.index('5\% FPR')
    #labels.insert(0, labels.pop(idx))
    #handles.insert(0, handles.pop(idx))
    
    # remove legend from all subpots
    
    T = len(mean_metrics[0])

    for i in range(2):
        for j in range(2):
            ax1[j][i].legend([],[], frameon=False)


    '''
    # for fpr y axis
    ax1[0][0].yaxis.set_major_locator(plt.MaxNLocator(1))
    ax1[1][0].yaxis.set_major_locator(plt.MaxNLocator(4))

    # change the y axis to percentage, but without the percentage sign
    ax1[0][0].set_yticklabels(['{:.0f}'.format(x*100) for x in ax1[0][0].get_yticks()])
    ax1[1][0].set_yticklabels(['{:.0f}'.format(x*100) for x in ax1[1][0].get_yticks()])
    ax1[0][0].tick_params(labelsize=fs)
    ax1[1][0].tick_params(labelsize=fs)

    # for tpr y axis
    ax1[0][1].yaxis.set_major_locator(plt.MaxNLocator(1))
    ax1[1][1].yaxis.set_major_locator(plt.MaxNLocator(4))
    
    ax1[0][1].set_yticklabels(['{:.0f}'.format(x*100) for x in ax1[0][1].get_yticks()])
    ax1[1][1].set_yticklabels(['{:.0f}'.format(x*100) for x in ax1[1][1].get_yticks()])

    ax1[0][1].tick_params(labelsize=fs)
    ax1[1][1].tick_params(labelsize=fs)
    '''

    return fig1,ax1


def plot_wo_y_break(methods, mean_metrics,std_metrics,legend=False,legend_fs=12,ylims=None):
    
    #sns.set_theme()
    #sns.set_context("notebook")
    mpl.rcParams['axes.prop_cycle'] = mpl.cycler(color=['#d62728','#1f77b4', '#2ca02c', '#9467bd',  
                                                        '#ff7f0e','#8c564b', '#e377c2', '#7f7f7f', 
                                                        '#bcbd22', '#17becf']) 
    fig1,ax1 = plt.subplots(1,2,figsize=(12,5))

    # add six subplots

    fs=18

    # set y ticks label font size
    ax1[1].tick_params(labelsize=fs)
    ax1[0].tick_params(labelsize=fs)
    ax1[1].xaxis.tick_bottom()
    ax1[1].set_xticks([20000,50000,80000,110000,140000])
    ax1[1].set_xticklabels(["20k","50k", "80k","110k", "140k"],fontsize=fs)
    ax1[1].set_xlim([xlim_min,150000])
    ax1[0].set_xlim([xlim_min,150000])
    # set the y lim to be start from 0.001
    # but dont set the upper limit

    plot_all(mean_metrics,[ax1[0],ax1[1]],methods,std_metrics,figure_index=0,up=True,ylims=ylims)
    #plot_all(mean_metrics,[ax1[1],ax1[1]],methods,std_metrics,figure_index=0,up=True,ylims=ylims)

    # set the fpr, tpr, and threshold
    ax1[1].set_xlabel('time($t$)',fontsize=fs)
    #ax1[1][0].set_ylabel('TPR',fontsize=fs,loc=loc)
    ax1[0].set_title('FPR(%)',fontsize=fs)
    ax1[1].set_xlabel('time($t$)',fontsize=fs)
    #ax1[1][1].set_ylabel('FPR',fontsize=fs,loc=loc)
    ax1[1].set_title('TPR(%)',fontsize=fs)


    
    # remove legend from all subpots
    

    ax1[0].legend([],[], frameon=False)
    ax1[1].legend([],[], frameon=False)

    return fig1,ax1