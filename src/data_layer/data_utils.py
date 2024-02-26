import os
import numpy as np
import logging
from ..utils.common_utils import *

from collections.abc import Sequence

class SyntheticDataStream(Sequence):

    def __init__(self,stream_conf,root_dir=None ):

        self.stream_conf = stream_conf 

        self.mu_id    = stream_conf.mu_id 
        self.sigma_id = stream_conf.sigma_id 
        

        self.mu_ood   = stream_conf.mu_ood 

        self.sigma_ood = stream_conf.sigma_ood 
        
        self.gamma    = stream_conf.gamma
        self.num_samples   = stream_conf.num_samples

        self.num_id   = int( (1-self.gamma)*self.num_samples)
        self.num_ood  = int(self.gamma * self.num_samples) 

        self.stream = []

    def __len__(self):
        return len(self.stream)
    
    def __getitem__(self,i):
        return self.stream[i]
    

    def get_stream(self):

        lst_id = list(np.random.normal(self.mu_id, self.sigma_id, self.num_id))
        lst_ood = list(np.random.normal(self.mu_ood, self.sigma_ood, self.num_ood))

        ds_id  = [[s,1] for s in lst_id ]
        ds_ood = [[s,0] for s in lst_ood]
        self.stream = np.array( ds_id + ds_ood )

        np.random.shuffle(self.stream)

        self.scores_id = lst_id
        self.scores_ood = lst_ood 

        return self.stream 

    def get_stats(self,alpha,k=500):

        self.lambda_star = find_percentile(1-alpha, self.mu_ood, self.sigma_ood) # the 5% FPR threshold of dout
        self.true_TPR    = 1-find_CDF(self.lambda_star, self.mu_id, self.sigma_id)    # the true TPR of din

        #lamda_95tpr = find_percentile(alpha, mu_in, sigma_in) # the 95% TPR threshold of din
        #fpr_95tpr = 1-find_CDF(lamda_95tpr, mu_out, sigma_out) # the FPR of dout when TPR of din is 95%

        # collect k data points and find the 95% TPR percentile
        id_sample = list(np.random.normal(self.mu_id, self.sigma_id, k))
        id_sample.sort()

        self.lamda_95tpr = id_sample[int(k*alpha)] # the 95% TPR threshold of din

        self.lamda_90tpr = id_sample[int(k*0.1)] # the 90% TPR threshold of din

        self.lamda_85tpr = id_sample[int(k*0.15)] # the 85% TPR threshold of din

        self.lamda_80tpr = id_sample[int(k*0.2)] # the 80% TPR threshold of din
        
        self.lamda_0tpr = id_sample[k-1] # the 0% TPR threshold of din
        self.lamda_5tpr = id_sample[int(k*0.95)] # the 5% TPR threshold of din
        self.lamda_10tpr = id_sample[int(k*0.90)] # the 10% TPR threshold of din
        self.lamda_15tpr = id_sample[int(k*0.85)] # the 15% TPR threshold of din
        self.lamda_20tpr = id_sample[int(k*0.80)] # the 20% TPR threshold of din

        self.fpr_95tpr  = 1-find_CDF(self.lamda_95tpr, self.mu_ood, self.sigma_ood) # the FPR of dout when TPR of din is 95%

        self.fpr_90tpr  = 1-find_CDF(self.lamda_90tpr, self.mu_ood, self.sigma_ood) # the FPR of dout when TPR of din is 90%

        self.fpr_85tpr  = 1-find_CDF(self.lamda_85tpr, self.mu_ood, self.sigma_ood) # the FPR of dout when TPR of din is 85%

        self.fpr_80tpr  = 1-find_CDF(self.lamda_80tpr, self.mu_ood, self.sigma_ood) # the FPR of dout when TPR of din is 80%

        logging.info(f"lambda_star: {self.lambda_star}, true_TPR: {self.true_TPR},lamda_95tpr: {self.lamda_95tpr}, fpr_95tpr: {self.fpr_95tpr}")
        
        logging.info(f"lambda_90tpr:{self.lamda_90tpr},  fpr_90tpr : {self.fpr_90tpr}")      
        logging.info(f"lambda_85tpr:{self.lamda_85tpr},  fpr_85tpr : {self.fpr_85tpr}")    
        logging.info(f"lambda_80tpr:{self.lamda_80tpr},  fpr_80tpr : {self.fpr_80tpr}")       
        
        out = {"lambda_star": self.lambda_star, 'true_TPR':self.true_TPR, 
               'lambda_95tpr': self.lamda_95tpr, 'fpr_95tpr':self.fpr_95tpr,
               'lambda_90tpr' : self.lamda_90tpr, 'fpr_90tpr' : self.fpr_90tpr,
               'lambda_85tpr' : self.lamda_85tpr, 'fpr_85tpr' : self.fpr_85tpr,
               'lambda_80tpr' : self.lamda_80tpr, 'fpr_80tpr' : self.fpr_80tpr
               }
        
        

        return out 
    
class RealDataStream(Sequence):
    
    def __init__(self,stream_conf, root_dir =None ):

        # sets that are used for metrics evaluation
        self.id_frozen = None
        self.ood_frozen = None 

        # sets that are used for sampling
        self.id_sampling = None
        self.ood_sampling = None 

        self.gamma = stream_conf.gamma 

        self.id_name = stream_conf.name_id
        self.ood_name = stream_conf.name_ood

        self.ood_method = stream_conf.ood_method

        self.num_samples   = stream_conf.num_samples

        self.num_id   = int( (1-self.gamma)*self.num_samples)
        self.num_ood  = int(self.gamma * self.num_samples) 

        self.root_dir = root_dir

        self.stream = [] 

    
    def get_stream(self):

        path = os.path.join(self.root_dir,'score',self.ood_method)

        if self.ood_method in ['SSD','GRADNORM',"ASH","ASH_AUGMIX",'REACT']:
            model = 'resnet50'
        elif self.ood_method in ['REACT_SWINT']:
            model = 'swint'
        elif self.ood_method in ['IDECODE']:
            model = 'resnet34'
        else:
            model = 'resnet18'

        path = os.path.join(path,'{}_{}'.format(self.id_name,model))
        path = os.path.join(path,'scores')
        
        # load id
        self.id_frozen = np.sort(np.load(os.path.join(path,self.id_name)+'.npz')['conf'])
        self.id_sampling = self.id_frozen.copy()
        
        logging.info('loaded {} id points, name: {}'.format(len(self.id_frozen),self.id_name))
        
        # load ood
        
        self.ood_frozen   = np.sort(np.load(os.path.join(path,self.ood_name)+'.npz')['conf'])
        self.ood_sampling = self.ood_frozen.copy()
        
        logging.info('loaded {} ood points, name: {}'.format(len(self.ood_frozen),self.ood_name))

        # sample with replacement, and allows repeated samples
        
        
        lst_id = random.choices(self.id_sampling, k=self.num_id)
        lst_ood = random.choices(self.ood_sampling,k=self.num_ood)


        ds_id  = [[s,1] for s in lst_id ]
        ds_ood = [[s,0] for s in lst_ood]


        self.scores_id = lst_id

        self.scores_ood = lst_ood 

        self.scores_id_test_sorted   = self.id_frozen
        
        self.scores_ood_test_sorted  = self.ood_frozen


        self.stream = np.array( ds_id + ds_ood )

        np.random.shuffle(self.stream)

        return self.stream 
    
    def __len__(self):
        return len(self.stream)


    def __getitem__(self,i):
        return self.stream[i]
    
    
    def get_stats(self,alpha=0.05, k= 1000):
        
        self.lambda_star = np.quantile(self.ood_frozen,0.95) # the 5% FPR threshold of dout
        self.true_TPR = sum(self.id_frozen > self.lambda_star)/len(self.id_frozen) # the true TPR of din when FPR of dout is 5%

        id_sample = random.sample(list(self.id_frozen),k)
        id_sample.sort()

        self.lamda_95tpr = id_sample[int(k*alpha)] # the 95% TPR threshold of din
        self.fpr_95tpr = sum(self.ood_frozen > self.lamda_95tpr)/len(self.ood_frozen) # the FPR of dout when TPR of din is 95%

        self.lamda_90tpr = id_sample[int(k*0.1)] # the 90% TPR threshold of din

        self.fpr_90tpr = sum(self.ood_frozen > self.lamda_90tpr)/len(self.ood_frozen) # the FPR of dout when TPR of din is 90%

        self.lamda_80tpr = id_sample[int(k*0.2)] # the 80% TPR threshold of din

        self.fpr_80tpr = sum(self.ood_frozen > self.lamda_80tpr)/len(self.ood_frozen) # the FPR of dout when TPR of din is 90%

        self.lamda_85tpr = id_sample[int(k*0.15)] # the 80% TPR threshold of din

        self.fpr_85tpr = sum(self.ood_frozen > self.lamda_85tpr)/len(self.ood_frozen) # the FPR of dout when TPR of din is 90%
        

        self.lamda_0tpr = id_sample[k-1] # the 0% TPR threshold of din
        self.lamda_5tpr = id_sample[int(k*0.95)] # the 5% TPR threshold of din
        self.lamda_10tpr = id_sample[int(k*0.90)] # the 10% TPR threshold of din
        self.lamda_15tpr = id_sample[int(k*0.85)] # the 15% TPR threshold of din
        self.lamda_20tpr = id_sample[int(k*0.80)] # the 20% TPR threshold of din

        logging.info(f"lambda_star: {self.lambda_star}, true_TPR: {self.true_TPR},lamda_95tpr: {self.lamda_95tpr}, fpr_95tpr: {self.fpr_95tpr}") 

        logging.info(f"lambda_90tpr:{self.lamda_90tpr},  fpr_90tpr : {self.fpr_90tpr}")      
        logging.info(f"lambda_85tpr:{self.lamda_85tpr},  fpr_85tpr : {self.fpr_85tpr}")    
        logging.info(f"lambda_80tpr:{self.lamda_80tpr},  fpr_80tpr : {self.fpr_80tpr}")       
        
        out = {"lambda_star": self.lambda_star, 'true_TPR':self.true_TPR, 
               'lambda_95tpr': self.lamda_95tpr, 'fpr_95tpr':self.fpr_95tpr,
               'lambda_90tpr' : self.lamda_90tpr, 'fpr_90tpr' : self.fpr_90tpr,
               'lambda_85tpr' : self.lamda_85tpr, 'fpr_85tpr' : self.fpr_85tpr,
               'lambda_80tpr' : self.lamda_80tpr, 'fpr_80tpr' : self.fpr_80tpr
               }
        
        return out 

def get_data_streams(conf):

    data_conf = conf.data_conf  
    streams = []

    if(data_conf.simulation):
        for stream_conf in data_conf.streams:
            stream = SyntheticDataStream(stream_conf,conf.root_dir)
            streams.append(stream)

    else:
        for stream_conf in data_conf.streams:
            stream = RealDataStream(stream_conf,conf.root_dir)
            streams.append(stream)
    
    for stream in streams:
        stream.get_stream()
        stream.get_stats(conf.alpha)
    
    return streams 

class MergedStream(Sequence) :
    def __init__(self,streams):
        self.all_streams = streams 
        self.stream = np.vstack([s.stream for s in streams ])
        T = len(self.stream)
        self.lst_id = [self.stream[i][0] for i in range(T) if self.stream[i][1]==1]
        self.lst_ood = [self.stream[i][0] for i in range(T) if self.stream[i][1]==0]

        self.stream_ids = []
        for i in range(len(streams)):
            self.stream_ids.extend([i]*len(streams[i]))

    def get_stream(self,t):
        return self.all_streams[self.stream_ids[t]]
    
    def __len__(self):
        return len(self.stream)
    
    def __getitem__(self,i):
        return self.stream[i]
    
    
        
        