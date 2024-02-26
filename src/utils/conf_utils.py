import copy
import sys 
import os 

def streams_conf_to_key(streams_conf):
    st_key = []
    for i,stream in enumerate(streams_conf):
        ks = list(stream.keys())
        ks.sort()
        st_key.append("-x-".join([f"{k}__{stream[k]}" for k in ks]))
    st_key = "--xx--".join(st_key)
    return st_key 

def streams_key_to_conf(streams_key):
    lst_streams_key = []
    if('--xx--' in streams_key ):
        lst_streams_key  = streams_key.split('--xx--')
    else:
        lst_streams_key = [streams_key]
    
    streams_conf = []
    for stream_key in lst_streams_key:
        lst_kv = [ z.split('__') for z in stream_key.split('-x-')]
        streams_conf.append(dict(lst_kv))
    return streams_conf 

def create_confs(conf,params):
    from itertools import product
    keys, values = zip(*params.items())
    lst_confs = []

    for bundle in product(*values):
        d = dict(zip(keys, bundle))
        #print(d)
        conf_cp = copy.deepcopy(conf)
        data_conf = conf_cp.data_conf

        path_d = {}
        path_keys = []


        out_pfx = ""
        if('id_ds' in d and 'score' in d ):   
            for st in conf_cp['data_conf']['streams']:
                st['name_id']= d['id_ds']
                st['ood_method'] = d['score']
            
            out_pfx  = os.path.join(f"{d['id_ds']}", f"{d['score']}")

        
          #if(data_conf.simulation):
        for i,stream in enumerate(data_conf.streams):
            if('gamma' in d):
                stream.gamma = d['gamma']
        
        path_d['stream'] = streams_conf_to_key(data_conf.streams) 

        path_keys.append('stream')

        conf_cp['method'] = d['method'] 
        path_d['method']  = d['method']
        
        path_keys.append('method')

        path_d.update(d)

        
        if('prob' in d):
            conf_cp['prob']   = d['prob']
            path_keys.append('prob')

            
        if('window' in d):
            conf_cp['window'] = d['window']
            path_keys.append('window')

        if('delta' in d):
            conf_cp['delta']  = d['delta']
            path_keys.append('delta')
        
        if('alpha' in d):
            conf_cp['alpha']  = d['alpha']
            path_keys.append('alpha')

        if('tpr_k' in d):
            conf_cp['tpr_k']  = d['tpr_k']
            path_keys.append('tpr_k')
        
        if('num' in d):
            conf_cp['num']  = d['num']
            path_keys.append('num')

        conf_cp["seed"]   = d['seed']
        path_keys.append('seed')

        lst = [f"{k}__{path_d[k]}" for k in path_keys]
        
        z = os.path.join(*lst)

        conf_cp['run_dir'] = os.path.join(conf['output_root'], out_pfx, z)

        conf_cp['log_file_path']  = os.path.join(conf_cp['run_dir'], conf_cp['method']+".log" )
        conf_cp['out_file_path']  = os.path.join(conf_cp['run_dir'], conf_cp['method']+".pkl" )
        conf_cp['conf_file_path'] = os.path.join(conf_cp['run_dir'],  "run_config.yaml" )


        lst_confs.append(conf_cp)

    return lst_confs 

