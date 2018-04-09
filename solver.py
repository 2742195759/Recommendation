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
        self.data_loader.make_data()
        print('all user number:', self.data_loader.user_number)
        print('all item number:', self.data_loader.item_number)
        print('all feature number:', self.data_loader.feature_number)
        print('train user number:', len(self.data_loader.user_purchased_items.keys()))
        print('train instance number:', self.data_loader.train_sample_num)
        print('test user number:', len(self.data_loader.ground_truth.keys()))
        print('test item number:', len(self.data_loader.item_candidates))
        print('test instance number:', self.data_loader.test_sample_num)

        if args.model == 'efm':
            self.m = EFM(self.data_loader.params)
            self.m.build_loss()
            self.m.build_train_op()
            self.m.build_prediction()
        else:
            self.m = LRPPM(self.data_loader.params)
            self.m.build_loss()
            self.m.build_train_op()
            self.m.build_prediction()

    def MAP(self, ground_truth, pred):
        result = []
        for k,v in ground_truth.items():
            fit = [i[0] for i in pred[k]][:self.args.Top_K]
            tmp = 0
            hit = 0
            for j in range(len(fit)):
                if fit[j] in v:
                    hit += 1
                    tmp += hit / (j+1)
            result.append(tmp)
        return np.array(result).mean()

    def MRR(self, ground_truth, pred):
        result = []
        for k, v in ground_truth.items():
            fit = [i[0] for i in pred[k]][:self.args.Top_K]
            tmp = 0
            for j in range(len(fit)):
                if fit[j] in v:
                    tmp = 1 / (j + 1)
                    break
            result.append(tmp)
        return np.array(result).mean()

    def NDCG(self, ground_truth, pred):
        result = []
        for k, v in ground_truth.items():
            fit = [i[0] for i in pred[k]][:self.args.Top_K]
            temp = 0
            Z_u = 0
            for j in range(len(fit)):
                Z_u = Z_u + 1 / np.log2(j + 2)
                if fit[j] in v:
                    temp = temp + 1 / np.log2(j + 2)
            if Z_u == 0:
                temp = 0
            else:
                temp = temp / Z_u
            result.append(temp)
        return np.array(result).mean()

    def top_k(self, ground_truth, pred):
        correct = []
        co_length = []
        re_length = []
        pu_length = []
        p = []
        r = []
        f = []
        hit = []
        for k, v in ground_truth.items():
            temp = []
            fit = [i[0] for i in pred[k]][:self.args.Top_K]
            for j in fit:
                if j in v:
                    temp.append(j)
            if len(temp):
                hit.append(1)
            else:
                hit.append(0)
            co_length.append(len(temp))
            re_length.append(len(fit))
            pu_length.append(len(v))
            correct.append(temp)

        for i in range(len(ground_truth.keys())):
            if re_length[i] == 0:
                p_t = 0.0
            else:
                p_t = co_length[i] / float(re_length[i])
            if pu_length[i] == 0:
                r_t = 0.0
            else:
                r_t = co_length[i] / float(pu_length[i])
            p.append(p_t)
            r.append(r_t)
            if p_t != 0 or r_t != 0:
                f.append(2.0 * p_t * r_t / (p_t + r_t))
            else:
                f.append(0.0)
        return np.array(p).mean(), np.array(r).mean(), np.array(f).mean(), np.array(hit).mean()

    def evaluate(self):
        # map, mrr, p, r, f1, hit, ndcg = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
        index_path = self.args.base_path + self.args.category+'output_top_n_result_index'
        index = pd.read_csv(index_path, header=None)
        predictions_path = self.args.base_path + self.args.category+'output_top_n_result'
        predictions = pd.read_csv(predictions_path, header=None)

        ground_truth = {}
        pred = {}

        l = len(predictions.values)
        for i in range(l):
            ind = index.values[i]
            pre = predictions.values[i][0]
            user = ind[0]
            item = ind[1]
            pur_or_not = ind[2]

            if pur_or_not == 1:
                if user not in ground_truth.keys():
                    ground_truth[user] = [item]
                else:
                    ground_truth[user].append(item)

            if user not in pred.keys():
                pred[user] = {item: pre}
            else:
                pred[user][item] = pre

        for k, v in pred.items():
            pred[k] = sorted(v.items(), key=lambda item: item[1])[::-1]

        p, r, f1, hit = self.top_k(ground_truth, pred)
        map = self.MAP(ground_truth, pred)
        mrr = self.MRR(ground_truth, pred)
        ndcg = self.NDCG(ground_truth, pred)
        return map, mrr, p, r, f1, hit, ndcg

    def run(self):
        with tf.Session() as self.sess:
            init = tf.initialize_all_variables()
            self.sess.run(init)
            best_rmse = 100.0
            best_top_n = []
            best_value = 0.0
            for epoch in range(self.args.epoch_number):
                for step in range(int(self.data_loader.train_sample_num/self.args.batch_size)):
                    print('epoch: %s/%s, step %s/%s' % (epoch, self.args.epoch_number, step, int(self.data_loader.train_sample_num/self.args.batch_size)))
                    train_input_fn = self.data_loader.get_train_batch_data(self.args.batch_size)
                    if self.args.evaluate == 'rmse':
                        self.sess.run(self.m.train_op, feed_dict={self.m.user_id: train_input_fn[0],
                                                              self.m.item_id: train_input_fn[1],
                                                              self.m.pos_feature_id: train_input_fn[2],
                                                              self.m.neg_feature_id: train_input_fn[3],
                                                              self.m.a_ui: train_input_fn[4],
                                                              })
                    else:
                        self.sess.run(self.m.train_op, feed_dict={self.m.user_id: train_input_fn[0],
                                                                  self.m.item_id: train_input_fn[1],
                                                                  self.m.neg_item_id: train_input_fn[2],
                                                                  self.m.pos_feature_id: train_input_fn[3],
                                                                  self.m.neg_feature_id: train_input_fn[4],
                                                                  })

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
                            for _ in range(self.data_loader.test_sample_num/2):
                                test_input_fn = self.data_loader.get_test_batch_data(2)
                                score = self.sess.run(self.m.final_score,
                                                    feed_dict={self.m.user_id: test_input_fn[0],
                                                               self.m.item_id: test_input_fn[1]})
                                result.extend(score)

                            t = pd.DataFrame(result)
                            t.to_csv(self.args.base_path + self.args.category + 'output_top_n_result', index=False,
                                     header=None)

                            map, mrr, p, r, f1, hit, ndcg = self.evaluate()
                            if f1 > best_value:
                                best_top_n = [map, mrr, p, r, f1, hit, ndcg]
                                best_value = f1
                            print('map = %s, mrr = %s, p = %s, r = %s, f1 = %s, hit = %s, ndcg = %s' % (map, mrr, p, r, f1, hit, ndcg))
                            print('current best:%s' % (str(best_top_n)))

            if self.args.evaluate == 'rmse':
                print('overall best rmse = %s' % (best_rmse))
            else:
                print('overall best top_n = %s' % (str(best_top_n)))





