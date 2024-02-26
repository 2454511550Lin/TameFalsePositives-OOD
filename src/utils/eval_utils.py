

from .common_utils import * 

# compute the tpr and fpr for run_one result
def get_metrics_one(result,merged_stream,simluation = True):
    
    thr_lst = result['thr']
    T = len(thr_lst)
    fpr = []
    tpr = [] 
    for t in range(T):
        stream = merged_stream.get_stream(t)

        if simluation == True: 
            fpr.append(1 - find_CDF(thr_lst[t], stream.mu_ood, stream.sigma_ood) )
            tpr.append(1 - find_CDF(thr_lst[t], stream.mu_id, stream.sigma_id))

        else: 
            j1 = np.searchsorted(stream.scores_ood_test_sorted , thr_lst[t])

            fpr.append( 1 - (j1/len(stream.scores_ood_test_sorted)) )

            j2 = np.searchsorted(stream.scores_id_test_sorted , thr_lst[t])

            tpr.append(1 - (j2)/len(stream.scores_id_test_sorted))
            
            #fpr.append(sum(stream.scores_ood > thr_lst[t])/len(stream.scores_ood))
            #tpr.append(sum(stream.scores_id  > thr_lst[t])/len(stream.scores_id))

        
    metrics = dict({'fpr':fpr, 'tpr':tpr, 'thr':thr_lst})
    return metrics