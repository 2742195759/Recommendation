from model.efm import *
from model.lrppm import *
import tensorflow as tf
from data_loader import DataLoader
import numpy as np
import pandas as pd

'''
Solve a model: 
1. train the model
2. evaluate the model
'''


class Solver:
    def __init__(self, args):
        self.args = args
        self.data_loader = DataLoader(args)
        self.build_model()
        pass

    def loss(self) : 
        # return [ precision , recall ]
        d_u_stestm = self.args['d_u_stestm']
        prec = 0.0
        recall = 0.0 
        for t_u , t_sm in d_u_stestm : 
            s_predict = self.predict(t_u)   
            assert(isinstance(s_predict , set))
            if len(s_predict) : 
                prec += (len (s_predict.intersection(t_sm) ) * 1.0 / len(s_predict))
            recall += (len (s_predict.intersection(t_sm) ) * 1.0 / len(t_sm))
        prec /= len(d_u_stestm)
        recall /= len(d_u_stestm)


    def predict(self , usr) : 
        return (self.d_u_spredictm[usr]) - self.args['d_u_stestm'][user]

    def build_model(self ) : 
        R_uu = tf.placeholder('float64' , args['a_uu'].shape)
        R_um = tf.placeholder('float64' , args['a_um'].shape)
        R_uAm = tf.placeholder('float64' , args['a_uAm'].shape)

        R_similar = tf.cos(R_uu / R_uAm / R_uAm.reshape(1,-1))
        R_umrate  = tf.matmul(R_similar , R_um)

        R_ans = tf.nn.top_k(R_umrate , int(args['top_k']))
        self.R_ans = R_ans  
    
    def run(self):
        with tf.Session() as ss : 
            a_u_rankm = ss.run(self.R_ans).indices
        for t_u , t_m in a_u_rankm : 
            if t_u not in self.d_u_spredictm : 
                self.d_u_spredictm[t_u] = set()
            self.d_u_spredictm.add(t_m)
