from model.efm import *
from model.lrppm import *
import tensorflow as tf
from data_loader import DataLoader

'''
Solve a model: 
1. train the model
2. evaluate the model
'''


class Solver:
    def __init__(self, args):

        self.data_loader = DataLoader(args)
        self.data_loader.make_data()

        if args.model == 'efm':
            m = EFM(self.data_loader.params)
            m.loss()


    def evaluate(self):
        pass

    def run(self):
        pass


