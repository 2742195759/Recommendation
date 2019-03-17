import argparse
import configparser
from solver import *

'''
Control the experiment schedule
'''

class ExcuteExperiments:
    def __init__(self, s):
        self.solver = s

    def excute(self):
        self.solver.run()



if __name__ == '__main__':
    cf = configparser.ConfigParser()
    cf.read("./config/default_setting.conf")
    args = dict()
    for k , v in cf['path'].items() : 
        args[k] = v
    for k , v in cf['parameters'].items() : 
        args[k] = v

    s = Solver(args)
