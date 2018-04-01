import argparse
import configparser
from solver import *

'''
Control the experiment schedule
'''

class ExcuteExperiments:

    def __init__(self, s):
        pass

    def excute(self):
        pass





if __name__ == '__main__':
    cf = configparser.ConfigParser()
    cf.read("./config/default_setting.conf")
    parser = argparse.ArgumentParser()
    # 11 parameters
    parser.add_argument('--category', type=str, default=cf.get("path", "category"), required=False,
                        help='category')
    parser.add_argument('--base_path', type=str, default=cf.get("path", "base_path"), required=False,
                        help='base_path')

    parser.add_argument('--model', type=str, default=cf.get("parameters", "model"), required=False,
                        help='model')
    parser.add_argument('--batch_size', type=int, default=cf.get("parameters", "batch_size"), required=False,
                        help='batch_size')
    parser.add_argument('--embed_dim_r', type=int, default=cf.get("parameters", "embed_dim_r"), required=False,
                        help='embed_dim_r')
    parser.add_argument('--embed_dim_r_', type=int, default=cf.get("parameters", "embed_dim_r_"), required=False,
                        help='embed_dim_r_')
    parser.add_argument('--reg_u', type=float, default=cf.get("parameters", "reg_u"), required=False,
                        help='reg_u')
    parser.add_argument('--reg_h', type=float, default=cf.get("parameters", "reg_h"), required=False,
                        help='reg_h')
    parser.add_argument('--learning_rate', type=float, default=cf.get("parameters", "learning_rate"), required=False,
                        help='learning_rate')
    parser.add_argument('--optimizer', type=str, default=cf.get("parameters", "optimizer"), required=False,
                        help='optimizer')
    parser.add_argument('--model_dir', type=str, default=cf.get("parameters", "model_dir"), required=False,
                        help='model_dir')
    parser.add_argument('--min_eval_frequency', type=int, default=cf.get("parameters", "min_eval_frequency"),
                        required=False, help='min_eval_frequency')
    parser.add_argument('--epoch_number', type=int, default=cf.get("parameters", "epoch_number"), required=False,
                        help='epoch_number')
    args = parser.parse_args()

    s = Solver(args)
    m = ExcuteExperiments(s)
    m.excute()