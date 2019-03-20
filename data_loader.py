#coding=utf-8
import pickle
import random
import numpy as np
import pandas as pd
import os
import pickle

class CacheFile(object) : #同时存在才算可以,否则调用op
    def __init__(self , args) : 
        if args.get('cache_use' , 'y') == 'n' :
            self.use_cache = 0
        else  : 
            self.use_cache = 1

        self.dirpath = args['cache_path']
        self.args = args

    # add all the file in return_file_list data in the args
    def get(self , args) : 
        fs = self.override_get_file_list_()
        if not fs : return 
        if not isinstance(fs , list) : fs = [fs]
        if self._judge_file_cached_(fs) : 
            self._load_from_cache_(fs , args)
        else  :
            self.override_generate_data(args)
            self._save_to_cache_(fs , args)
                
    def _judge_file_cached_ (self , l_fs) :      
        if not self.use_cache :  return False # mask the cache
        for f in l_fs : 
            if not os.path.exists(os.path.join(self.dirpath , f)) :
                print '[INFO] ' + os.path.join(self.dirpath , f) +  ' not found'
                return False
        return True
        
    def _load_from_cache_ (self , l_fs , args) : 
        print '[INFO] cache found and loading data , may take little time'
        for f in l_fs : 
            if f in args : 
                print '[WARN] Exist data in the args'
            args[f] = pickle.load(open(os.path.join(self.dirpath , f)))
        return True

    def _save_to_cache_ ( self , l_fs , args ) : 
        if not self.use_cache : pass 
        for f in l_fs : 
            if f not in args : 
                assert(f in args)  #必须在里面，否则出错,没有生成
            pickle.dump(args[f] , open(os.path.join(self.dirpath , f ) , 'w'))

    def return_file_list_ (self) : 
       return self.override_get_file_list_()

    #def override_get_file_list_ (self) : 
    #    return ''

    #def override_generate_data (self , dic_output) : 
    #    return ''
        
        
##define your own CacheFile SubClass here
class MyCacheFile(CacheFile) : 
    def override_get_file_list_(self) : 
        return ['a_uu' , 'a_um' , 'a_uAm' , 'n_trainset' , 'd_u_stestm' ,
            'n_testset' , 'a_testset' , 'd_u_strainm']
        
    def override_generate_data(self , dic_output) : 
        print '[INFO] generate data , may take a lot of time'
        args = self.args
        i_mu = int(args['max_user'])
        i_mm = int(args['max_item'])
            
        M = int(args['m'])

        fp_r = open('%s/ratings.dat'%args['base_path'])

        dic_m_lu = dict()
        a_uAm = np.zeros([i_mu , 1] , dtype='int64')

        l_dataset = []
        a_um = np.zeros([i_mu , i_mm] , dtype='int64')

        while True : 
            line = fp_r.readline()
            if not line : break 
            t_u = int(line.split('::')[0]) - 1
            t_m = int(line.split('::')[1]) - 1
            assert(t_u >= 0)
            assert(t_m >= 0)
            l_dataset.append([t_u , t_m])
        
        a_dataset = (np.array(l_dataset))
        np.random.shuffle(a_dataset)
        n_dataset = len(a_dataset) 
        n_trainset= n_dataset / M * (M-1)
        n_testset = n_dataset - n_trainset
        d_u_stestm = dict()
        d_u_strainm = dict()

        a_trainset = a_dataset[:n_trainset]
        a_testset = a_dataset[n_trainset:]
    
        for t_u , t_m in a_trainset : 
            a_uAm[t_u] += 1 
            if t_u not in d_u_strainm : 
                d_u_strainm[t_u] = set()
            if t_m not in dic_m_lu : 
                dic_m_lu[t_m] = []
            dic_m_lu[t_m].append(t_u)
            d_u_strainm[t_u].add(t_m)
            a_um[t_u][t_m] = 1

        for t_u , t_m in a_testset : 
            if t_u not in d_u_stestm : 
                d_u_stestm[t_u] = set()
            d_u_stestm[t_u].add(t_m)

        a_uu = np.zeros([i_mu , i_mu] , dtype='int64')

        for m , lu in dic_m_lu.items() : 
            for u in lu : 
                for t in lu : 
                    if u != t  : 
                        a_uu[u][t] += 1     
                        a_uu[t][u] += 1     

        for i in range(i_mu) : 
            for j in range(i_mu) : 
                assert(not (a_uu[i][j] & 1))
                a_uu[i][j] /= 2 

        dic_output['a_uu'] = a_uu  
        dic_output['a_um'] = a_um  
        dic_output['a_uAm'] = a_uAm 
        dic_output['n_testset'] = n_testset
        dic_output['n_trainset'] = n_trainset
        dic_output['d_u_stestm'] = d_u_stestm
        dic_output['a_testset'] = a_testset
        dic_output['d_u_strainm'] = d_u_strainm

class DataLoader(): # cache file manager
                    # singleton , shouldn't init mutiply
    def __init__(self, args):
        # init the CacheFileManager
        self.args = args 
        self.dic_data = {}
        self.dic_dataname_cache = {}
        self.override_register_all()

    def get(self , data_name): # add the file_name in the args['file_name']
        if data_name in self.dic_data : 
            return self.dic_data[data_name]
        if data_name in self.dic_dataname_cache : 
            c = self.dic_dataname_cache[data_name]
            c.get(self.dic_data)
            assert(data_name in self.dic_data)
            return self.dic_data[data_name]
    
        print 'Fatal Error : Can not Get the Data named : %s' % data_name
        assert(False)

    # get all the filename in the registerd data , and add it to dic_caller
    def get_all(self , dic_caller) : 
        print '[INFO] loading or generate data'
        for k in self.dic_dataname_cache.keys() : 
            dic_caller[k] = self.get(k)
        print '[INFO] data loaded'
        

    def register(self , cacheclass) : 
        c = cacheclass(self.args)
        for i in c.return_file_list_() : 
            self.dic_dataname_cache[i] = c

    # modify by different class
    def override_register_all(self) : 
        self.register(MyCacheFile)
