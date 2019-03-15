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

    def build_model(self ) : 
        R_uu = tf.placeholder('float64' , args['a_uu'].shape)
        R_um = tf.placeholder('float64' , args['a_um'].shape)
        R_uAm = tf.placeholder('float64' , args['a_uAm'].shape)

        R_similar = tf.cos(R_uu / R_uAm / R_uAm.reshape(1,-1))
        R_umrate  = tf.matmul(R_similar , R_um)

        R_ans = tf.nn.top_k(R_umrate , int(args['top_k']))

    def run(self):
        pass





