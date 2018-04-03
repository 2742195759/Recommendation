from model.efm import *
from model.lrppm import *
import tensorflow as tf
from data_loader import DataLoader
import numpy as np

'''
Solve a model: 
1. train the model
2. evaluate the model
'''


class Solver:
    def __init__(self, args):
        self.args = args
        self.data_loader = DataLoader(args)
        self.data_loader.make_data()

        if args.model == 'efm':
            self.m = EFM(self.data_loader.params)
            self.m.build_loss()
            self.m.build_train_op()
            self.m.build_prediction()

    def evaluate(self):
        pass

    def run(self):
        with tf.Session() as self.sess:
            init = tf.initialize_all_variables()
            self.sess.run(init)
            best_rmse = 100
            for epoch in range(self.args.epoch_number):
                for step in range(int(self.data_loader.train_sample_num/self.args.batch_size)):
                    print('epoch: %s/%s, step %s/%s' % (epoch, self.args.epoch_number, step, int(self.data_loader.train_sample_num/self.args.batch_size)))
                    train_input_fn = self.data_loader.get_train_batch_data(self.args.batch_size)

                    self.sess.run(self.m.train_op, feed_dict={self.m.user_id: train_input_fn[0],
                                                              self.m.item_id: train_input_fn[1],
                                                              self.m.feature_id: train_input_fn[2],
                                                              self.m.a_ui: train_input_fn[3],
                                                              self.m.x_uf: train_input_fn[4],
                                                              self.m.y_if: train_input_fn[5]})

                    if step % 50 == 0:
                        result = []
                        if self.args.evaluate == 'rmse':
                            for _ in range(self.data_loader.test_sample_num/2):
                                test_input_fn = self.data_loader.get_test_batch_data(2)
                                error_square = self.sess.run(self.m.error_square,
                                                    feed_dict={self.m.user_id: test_input_fn[0],
                                                               self.m.item_id: test_input_fn[1],
                                                               self.m.a_ui:    test_input_fn[2]})
                                result.extend(error_square)

                            rmse = np.sqrt(np.array(result).mean())
                            print('current rmse = %s'%(rmse))
                            if rmse < best_rmse:
                                best_rmse = rmse
                            print('current best rmse = %s' % (best_rmse))
                        else:
                            pass





