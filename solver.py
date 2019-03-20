from model.efm import *
from model.lrppm import *
import tensorflow as tf
from data_loader import DataLoader
import numpy as np
import pandas as pd
import pdb
import xktensorflow as xktf
import time

'''
Solve a model: 
1. train the model
2. evaluate the model
'''


class Solver:
    def __init__(self, args):
        self.args = args
        self.data_loader = DataLoader(args)
        self.data_loader.get_all(args)
        start_time = time.time()
        self.build_model()
        self.run()
        print '[STATISTIC] time usage : %s ms' % ((time.time() - start_time) * 1000)
        print self.loss()
        print self.predict(0) #the 0 user 's recommendation
        pass

    def loss(self) : 
        # return [ precision , recall ]
        print '[INFO] Start Loss Calculate'
        d_u_stestm = self.args['d_u_stestm']
        prec = 0.0
        prec_tot = 0.0
        recall = 0.0 
        recall_tot = 0.0
        for t_u , t_sm in d_u_stestm.items() : 
            s_predict = self.predict(t_u)   
            assert(isinstance(s_predict , set))
            prec += len (s_predict.intersection(t_sm) )
            prec_tot += len(s_predict)
            recall += len (s_predict.intersection(t_sm) )
            recall_tot +=  len(t_sm)
        prec /= prec_tot
        recall /= recall_tot
        return [prec , recall]

    def predict(self , usr) : 
        return (self.d_u_spredictm[usr]) - self.args['d_u_strainm'][usr]

    def build_model(self ) : 
        with tf.device('/gpu:1') :
            print '[INFO] Start Model Build'
            R_uu = self.R_uu = tf.placeholder('float64' , self.args['a_uu'].shape)
            R_um = self.R_um = tf.placeholder('float64' , self.args['a_um'].shape)
            R_uAm = self.R_uAm = tf.placeholder('float64' , self.args['a_uAm'].shape)
            R_uAm = tf.sqrt(R_uAm)

            R_similar = R_uu / R_uAm / tf.reshape(R_uAm , (1,-1))

            R_rank = tf.nn.top_k(R_similar , int(self.args['top_k'])).indices
            R_mask = tf.cast(xktf.multi_hots(R_rank , R_similar.shape[0] , device='/cpu:0') , 'float64')
            R_umrate = tf.matmul(tf.multiply(R_similar , R_mask ), R_um)
            R_ans = tf.nn.top_k(R_umrate , int(self.args['top_k_item'])).indices

            self.R_ans = R_ans  
            print '[INFO] Start Model Run'
            data = self.args
            feed_data = {self.R_uu : data['a_uu'] , self.R_um : data['a_um'] , self.R_uAm : data['a_uAm']}
            with tf.Session() as ss : 
                #pdb.set_trace()
                #print ss.run(R_um , feed_data )
                #print ss.run(R_umrate , feed_data )
                a_u_rankm = ss.run(self.R_ans , feed_data )
            print a_u_rankm
            self.d_u_spredictm = {}
            for t_u , t_lm in enumerate(a_u_rankm) : 
                self.d_u_spredictm[t_u] = set(t_lm)
    
    def run(self):
        pass
