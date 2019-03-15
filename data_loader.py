import pickle
import random
import numpy as np
import pandas as pd

class DataLoader():

    def __init__(self, args):
        i_mu = args['max_user']
        i_mm = args['max_item']

        fp_r = open('%s/rating.dat')

        dic_m_lu = dict()
        a_uAm = np.zeros([i_mu , 1] , dtype='int64')

        while True : 
            line = fp_r.readline()
            if not line : break 
            t_u = int(line.split('::')[0]) - 1
            t_m = int(line.split('::')[1]) - 1
            assert(t_u >= 0)
            assert(t_i >= 0)
            a_uAm[t_u] += 1 
            if t_m not in dic_m_lu : 
                dic_m_lu[t_m] = []
            dic_m_lu[t_m].append(t_u)

        a_uu = np.zeros([i_mu , i_mu] , dtype='int64')
        a_um = np.ones(i_mu , i_mm , dtype='int64')

        for m , lu in dic_m_lu.items() : 
            for u in lu : 
                for t in lu : 
                    a_uu[u][t] += 1     

        for i in range(i_mu) : 
            for j in range(i_mu) : 
                a_uu[i][j] /= 2 

        args['a_uu'] = a_uu  
        args['a_um'] = a_um  
        args['a_uAm'] = a_uAm 

        pass

    def make_data(self):
        pass
