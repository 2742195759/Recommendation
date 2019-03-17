import pickle
import random
import numpy as np
import pandas as pd

class DataLoader():

    def __init__(self, args):
        i_mu = args['max_user']
        i_mm = args['max_item']
        M = args['M']

        fp_r = open('%s/rating.dat'%args['base_path'])

        dic_m_lu = dict()
        a_uAm = np.zeros([i_mu , 1] , dtype='int64')

        l_dataset = []

        while True : 
            line = fp_r.readline()
            if not line : break 
            t_u = int(line.split('::')[0]) - 1
            t_m = int(line.split('::')[1]) - 1
            assert(t_u >= 0)
            assert(t_i >= 0)
            l_dataset.append([t_u , t_m])
        
        a_dataset = array(l_dataset).randow_shuffle()
        n_dataset = len(a_dataset) 
        n_trainset= n_dataset / M 
        n_testset = n_dataset - n_trainset
        d_u_stestm = dict()
        d_u_strainm = dict()

        a_trainset = a_dataset[:n_trainset]
        a_testset = a_dataset[n_trainset:]
    
        for t_u , t_m in a_testset : 
            a_uAm[t_u] += 1 
            if t_u not in d_u_strainm : 
                d_u_strainm[t_u] = set()
            if t_m not in dic_m_lu : 
                dic_m_lu[t_m] = []
            dic_m_lu[t_m].append(t_u)
            d_u_strainm[t_u].add(t_m)

        for t_u , t_m in a_testset : 
            if t_u not in d_u_stestm : 
                d_u_stestm[t_u] = set()
            d_u_stestm[t_u].add(t_m)

        a_uu = np.zeros([i_mu , i_mu] , dtype='int64')
        a_um = np.ones(i_mu , i_mm , dtype='int64')

        for m , lu in dic_m_lu.items() : 
            for u in lu : 
                for t in lu : 
                    a_uu[u][t] += 1     

        for i in range(i_mu) : 
            for j in range(i_mu) : 
                assert(not (a_uu[i][j] & 1))
                a_uu[i][j] /= 2 

        args['a_uu'] = a_uu  
        args['a_um'] = a_um  
        args['a_uAm'] = a_uAm 
        args['n_testset'] = n_testset
        args['n_trainset'] = n_trainset
        args['d_u_stestm'] = d_u_stestm
        args['a_testset'] = a_testset
        args['d_u_strainm'] = d_u_strainm

        pass

    def make_data(self):
        pass
