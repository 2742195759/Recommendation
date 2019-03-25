#coding=utf-8
import pickle
import random
import numpy as np
import pandas as pd
import os
import pickle

class CacheFile(object) : #同时存在才算可以,否则调用op
    def __init__(self , data_loader , args) : 
        if args.get('cache_use' , 'y') == 'n' :
            self.use_cache = 0
        else  : 
            self.use_cache = 1

        self.dirpath = args['cache_path']
        self.data_loader = data_loader
        self.args = args

    # add all the file in return_file_list data in the args
    def get(self , args) : 
        fs = self.override_get_file_list_()
        if not fs : return 
        if not isinstance(fs , list) : fs = [fs]
        if self._judge_file_cached_(fs) : 
            print '[DATA] cache found and loading data , may take little time'
            self._load_from_cache_(fs , args)
        else  :
            print '[DATA] generate data , may take a lot of time'
            self.override_generate_data(args)
            self._save_to_cache_(fs , args)
                
    def _judge_file_cached_ (self , l_fs) :      
        if not self.use_cache :  return False # mask the cache
        for f in l_fs : 
            if not os.path.exists(os.path.join(self.dirpath , f)) :
                print '[DATA] ' + os.path.join(self.dirpath , f) +  ' not found'
                return False
        return True
        
    def _load_from_cache_ (self , l_fs , args) : 
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

    def get_data_fromloader(self, data_name) :
        return self.data_loader.get(data_name)

    #def override_get_file_list_ (self) : 
    #    return ''

    #def override_generate_data (self , dic_output) : 
    #    return ''
    ##define your own CacheFile SubClass here
        
class DataLoader(): # cache file manager
                    # singleton , shouldn't init mutiply
    def __init__(self, args):
        # init the CacheFileManager
        self.args = args 
        self.dic_data = {} #restore all the data in the dic , and return caller 
        self.dic_dataname_cache = {}
        self.override_option_args_process()
        self.override_register_all()
    def get(self , data_name): # add the file_name in the DataLoader.dic_data
        if data_name in self.dic_data : 
            return self.dic_data[data_name]
        if data_name in self.dic_dataname_cache : 
            c = self.dic_dataname_cache[data_name]
            c.get(self.dic_data)
            assert(data_name in self.dic_data)
            return self.dic_data[data_name]
    
        print '[Fatal Error] Can not Get the Data named : %s' % data_name
        assert(False)

    def get_list_return_dict(self , data_names) : 
        if not isinstance(data_names , list) : 
            data_names = [data_names]
        ans = {}
        for data_name in data_names : 
            ans[data_name] = self.get(data_name)
        return ans

    # get all the filename in the registerd data , and add it to dic_caller
    def get_alldata_putin_dic(self , dic_caller) : 
        print '[INFO] loading or generate data'
        for k in self.dic_dataname_cache.keys() : 
            dic_caller[k] = self.get(k)
            print '[DATA] data %s loaded' % k
        print '[INFO] data loaded finished'
        

    def register(self , cacheclass) : 
        c = cacheclass(self , self.args)
        for i in c.return_file_list_() : 
            self.dic_dataname_cache[i] = c

    # modify by different class
    def override_register_all(self) : 
        self.register(MyCacheFile)
    def override_option_args_process(self) :  # deal with some useful args convert
        pass
