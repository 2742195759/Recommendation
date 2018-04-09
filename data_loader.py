import pickle
import random
import numpy as np
import pandas as pd

class DataLoader():

    def __init__(self, args):
        self.args = args
        self.params = {}
        self.train_batch_id = 0
        self.test_batch_id = 0
        for k, v in vars(self.args).items():
            self.params[k] = v

        self.base_path = self.args.base_path
        self.category = self.args.category
        path = self.base_path + self.category
        print('loading data ...')
        train_data_path = path + 'train_dict'
        self.train_data = pickle.load(open(train_data_path, 'rb'))
        test_data_path = path + 'test_dict'
        self.test_data = pickle.load(open(test_data_path, 'rb'))
        user_purchased_items_path = path + 'user_purchased_items_dict'
        self.user_purchased_items = pickle.load(open(user_purchased_items_path, 'rb'))
        A_path = path + 'A'
        self.A = pickle.load(open(A_path, 'rb'))
        if self.args.model == 'efm':
            X_path = path + 'X'
            self.X = pickle.load(open(X_path, 'rb'))
            Y_path = path + 'Y'
            self.Y = pickle.load(open(Y_path, 'rb'))



        user_id_path = path + 'user_id_dict'
        self.user_number = len(list(pickle.load(open(user_id_path, 'rb')).values()))
        item_id_path = path + 'item_id_dict'
        self.item_number = len(list(pickle.load(open(item_id_path, 'rb')).values()))
        feature_id_path = path + 'feature_id_dict'
        self.feature_number = len(list(pickle.load(open(feature_id_path, 'rb')).values()))

        self.params['user_number'] = self.user_number
        self.params['item_number'] = self.item_number
        self.params['feature_number'] = self.feature_number
        print('loading data end ...')

        # build for model testing
        self.item_candidates = []
        for k, v in self.test_data.items():
            item = int(k.split('@')[1])
            if item not in self.item_candidates:
                self.item_candidates.append(item)
        self.item_candidates = random.sample(self.item_candidates, 200)

        self.ground_truth = dict()
        for k, v in self.test_data.items():
            user = int(k.split('@')[0])
            item = int(k.split('@')[1])
            if item in self.item_candidates:
                if user not in self.ground_truth.keys():
                    self.ground_truth[user] = [item]
                else:
                    self.ground_truth[user].append(item)

    def make_data(self):
        self.get_train_raw_data()
        self.get_test_raw_data()

    def get_train_raw_data(self):
        if self.args.model == 'efm':
            if self.args.evaluate == 'rmse':
                self.train_users = []
                self.train_items = []
                self.train_features = []
                self.train_a_uis = []
                self.train_x_ufs = []
                self.train_y_ifs = []

                for k, v in self.train_data.items():
                    user = k.split('@')[0]
                    item = k.split('@')[1]
                    for fos in v.split(':'):
                        feature = fos.split('|')[0]
                        a_ui = self.A[user + '@' + item]
                        x_uf = self.X[user + '@' + feature]
                        y_if = self.Y[item + '@' + feature]

                        self.train_users.append(int(user))
                        self.train_items.append(int(item))
                        self.train_features.append(int(feature))
                        self.train_a_uis.append(float(a_ui))
                        self.train_x_ufs.append(float(x_uf))
                        self.train_y_ifs.append(float(y_if))
                self.train_sample_num = len(self.train_users)
            else:
                self.train_users = []
                self.train_pos_items = []
                self.train_neg_items = []
                self.train_features = []
                self.train_x_ufs = []
                self.train_y_ifs = []

                for k, v in self.train_data.items():
                    user = k.split('@')[0]
                    item = k.split('@')[1]
                    for fos in v.split(':'):
                        feature = fos.split('|')[0]
                        x_uf = self.X[user + '@' + feature]
                        y_if = self.Y[item + '@' + feature]
                        for _ in range(self.args.neg_number):
                            neg_item = np.random.choice(self.item_candidates)
                            while neg_item in self.user_purchased_items[user]:
                                neg_item = np.random.choice(self.item_candidates)
                            self.train_users.append(int(user))
                            self.train_pos_items.append(int(item))
                            self.train_neg_items.append(int(neg_item))
                            self.train_features.append(int(feature))
                            self.train_x_ufs.append(float(x_uf))
                            self.train_y_ifs.append(float(y_if))

                self.train_sample_num = len(self.train_users)
        else:
            if self.args.evaluate == 'rmse':
                self.train_users = []
                self.train_items = []
                self.train_pos_features = []
                self.train_neg_features = []
                self.train_a_uis = []

                for k, v in self.train_data.items():
                    user = k.split('@')[0]
                    item = k.split('@')[1]
                    f_list = [i.split('|')[0] for i in v.split(':')]
                    for fos in v.split(':'):
                        pos_feature = fos.split('|')[0]
                        a_ui = self.A[user + '@' + item]
                        for _ in range(self.args.neg_feature_number):
                            neg_feature = np.random.randint(self.feature_number)
                            while neg_feature in f_list:
                                neg_feature = np.random.randint(self.feature_number)
                            self.train_users.append(int(user))
                            self.train_items.append(int(item))
                            self.train_pos_features.append(int(pos_feature))
                            self.train_neg_features.append(int(neg_feature))
                            self.train_a_uis.append(float(a_ui))
                self.train_sample_num = len(self.train_users)
            else:
                self.train_users = []
                self.train_pos_items = []
                self.train_neg_items = []
                self.train_pos_features = []
                self.train_neg_features = []

                for k, v in self.train_data.items():
                    user = k.split('@')[0]
                    item = k.split('@')[1]
                    f_list = [i.split('|')[0] for i in v.split(':')]
                    for fos in v.split(':'):
                        pos_feature = fos.split('|')[0]
                        for _ in range(self.args.neg_number):
                            neg_item = np.random.choice(self.item_candidates)
                            while neg_item in self.user_purchased_items[user]:
                                neg_item = np.random.choice(self.item_candidates)
                            for _ in range(self.args.neg_feature_number):
                                neg_feature = np.random.randint(self.feature_number)
                                while neg_feature in f_list:
                                    neg_feature = np.random.randint(self.feature_number)
                                self.train_users.append(int(user))
                                self.train_pos_items.append(int(item))
                                self.train_neg_items.append(int(neg_item))
                                self.train_pos_features.append(int(pos_feature))
                                self.train_neg_features.append(int(neg_feature))

                self.train_sample_num = len(self.train_users)


    def get_test_raw_data(self):
        if self.args.evaluate == 'rmse':
            self.test_users = []
            self.test_items = []
            self.test_a_uis = []
            for k, v in self.test_data.items():
                user = k.split('@')[0]
                item = k.split('@')[1]
                a_ui = self.A[user + '@' + item]
                self.test_users.append(int(user))
                self.test_items.append(int(item))
                self.test_a_uis.append(float(a_ui))
            self.test_sample_num = len(self.test_users)
        else:
            self.test_users = []
            self.test_items = []
            output_search_result_index = []
            for user in self.ground_truth.keys():
                for item in self.item_candidates:
                    self.test_users.append(int(user))
                    self.test_items.append(int(item))
                    if item in self.ground_truth[int(user)]:
                        tmp = [user, item, 1]
                    else:
                        tmp = [user, item, 0]
                    output_search_result_index.append(tmp)

            self.test_sample_num = len(self.test_users)
            t = pd.DataFrame(output_search_result_index)
            t.to_csv(self.base_path + self.category + 'output_top_n_result_index', index=False, header=None)

    def get_train_batch_data(self, batch_size):
        if self.args.model == 'efm':
            if self.args.evaluate == 'rmse':
                l = len(self.train_users)
                if self.train_batch_id + batch_size > l:
                    batch_train_users = self.train_users[self.train_batch_id:] + self.train_users[
                                                                                 :self.train_batch_id + batch_size - l]
                    batch_train_items = self.train_items[self.train_batch_id:] + self.train_items[
                                                                                 :self.train_batch_id + batch_size - l]
                    batch_train_features = self.train_features[self.train_batch_id:] + self.train_features[
                                                                                       :self.train_batch_id + batch_size - l]
                    batch_train_a_uis = self.train_a_uis[self.train_batch_id:] + self.train_a_uis[
                                                                                 :self.train_batch_id + batch_size - l]
                    batch_train_x_ufs = self.train_x_ufs[self.train_batch_id:] + self.train_x_ufs[
                                                                                 :self.train_batch_id + batch_size - l]
                    batch_train_y_ifs = self.train_y_ifs[self.train_batch_id:] + self.train_y_ifs[
                                                                                 :self.train_batch_id + batch_size - l]
                    self.train_batch_id = self.train_batch_id + batch_size - l

                else:
                    batch_train_users = self.train_users[self.train_batch_id:self.train_batch_id + batch_size]
                    batch_train_items = self.train_items[self.train_batch_id:self.train_batch_id + batch_size]
                    batch_train_features = self.train_features[self.train_batch_id:self.train_batch_id + batch_size]
                    batch_train_a_uis = self.train_a_uis[self.train_batch_id:self.train_batch_id + batch_size]
                    batch_train_x_ufs = self.train_x_ufs[self.train_batch_id:self.train_batch_id + batch_size]
                    batch_train_y_ifs = self.train_y_ifs[self.train_batch_id:self.train_batch_id + batch_size]

                    self.train_batch_id = self.train_batch_id + batch_size

                return [batch_train_users, batch_train_items, batch_train_features,
                        batch_train_a_uis, batch_train_x_ufs, batch_train_y_ifs]
            else:
                l = len(self.train_users)
                if self.train_batch_id + batch_size > l:
                    batch_train_users = self.train_users[self.train_batch_id:] + self.train_users[
                                                                                 :self.train_batch_id + batch_size - l]
                    batch_train_pos_items = self.train_pos_items[self.train_batch_id:] + self.train_pos_items[
                                                                                         :self.train_batch_id + batch_size - l]
                    batch_train_neg_items = self.train_neg_items[self.train_batch_id:] + self.train_neg_items[
                                                                                         :self.train_batch_id + batch_size - l]
                    batch_train_features = self.train_features[self.train_batch_id:] + self.train_features[
                                                                                       :self.train_batch_id + batch_size - l]
                    batch_train_x_ufs = self.train_x_ufs[self.train_batch_id:] + self.train_x_ufs[
                                                                                 :self.train_batch_id + batch_size - l]
                    batch_train_y_ifs = self.train_y_ifs[self.train_batch_id:] + self.train_y_ifs[
                                                                                 :self.train_batch_id + batch_size - l]
                    self.train_batch_id = self.train_batch_id + batch_size - l
                else:
                    batch_train_users = self.train_users[self.train_batch_id:self.train_batch_id + batch_size]
                    batch_train_pos_items = self.train_pos_items[self.train_batch_id:self.train_batch_id + batch_size]
                    batch_train_neg_items = self.train_neg_items[self.train_batch_id:self.train_batch_id + batch_size]
                    batch_train_features = self.train_features[self.train_batch_id:self.train_batch_id + batch_size]
                    batch_train_x_ufs = self.train_x_ufs[self.train_batch_id:self.train_batch_id + batch_size]
                    batch_train_y_ifs = self.train_y_ifs[self.train_batch_id:self.train_batch_id + batch_size]

                    self.train_batch_id = self.train_batch_id + batch_size

                return [batch_train_users, batch_train_pos_items, batch_train_neg_items,
                        batch_train_features, batch_train_x_ufs, batch_train_y_ifs]
        else:
            if self.args.evaluate == 'rmse':
                l = len(self.train_users)
                if self.train_batch_id + batch_size > l:
                    batch_train_users = self.train_users[self.train_batch_id:] + self.train_users[
                                                                                 :self.train_batch_id + batch_size - l]
                    batch_train_items = self.train_items[self.train_batch_id:] + self.train_items[
                                                                                 :self.train_batch_id + batch_size - l]
                    batch_train_pos_features = self.train_pos_features[self.train_batch_id:] + self.train_pos_features[
                                                                                       :self.train_batch_id + batch_size - l]
                    batch_train_neg_features = self.train_neg_features[self.train_batch_id:] + self.train_neg_features[
                                                                                               :self.train_batch_id + batch_size - l]
                    batch_train_a_uis = self.train_a_uis[self.train_batch_id:] + self.train_a_uis[
                                                                                 :self.train_batch_id + batch_size - l]
                    self.train_batch_id = self.train_batch_id + batch_size - l

                else:
                    batch_train_users = self.train_users[self.train_batch_id:self.train_batch_id + batch_size]
                    batch_train_items = self.train_items[self.train_batch_id:self.train_batch_id + batch_size]
                    batch_train_pos_features = self.train_pos_features[self.train_batch_id:self.train_batch_id + batch_size]
                    batch_train_neg_features = self.train_neg_features[self.train_batch_id:self.train_batch_id + batch_size]
                    batch_train_a_uis = self.train_a_uis[self.train_batch_id:self.train_batch_id + batch_size]

                    self.train_batch_id = self.train_batch_id + batch_size

                return [batch_train_users, batch_train_items, batch_train_pos_features,
                        batch_train_neg_features, batch_train_a_uis]
            else:
                l = len(self.train_users)
                if self.train_batch_id + batch_size > l:
                    batch_train_users = self.train_users[self.train_batch_id:] + self.train_users[
                                                                                 :self.train_batch_id + batch_size - l]
                    batch_train_pos_items = self.train_pos_items[self.train_batch_id:] + self.train_pos_items[
                                                                                         :self.train_batch_id + batch_size - l]
                    batch_train_neg_items = self.train_neg_items[self.train_batch_id:] + self.train_neg_items[
                                                                                         :self.train_batch_id + batch_size - l]
                    batch_train_pos_features = self.train_pos_features[self.train_batch_id:] + self.train_pos_features[
                                                                                       :self.train_batch_id + batch_size - l]
                    batch_train_neg_features = self.train_neg_features[self.train_batch_id:] + self.train_neg_features[
                                                                                       :self.train_batch_id + batch_size - l]

                    self.train_batch_id = self.train_batch_id + batch_size - l
                else:
                    batch_train_users = self.train_users[self.train_batch_id:self.train_batch_id + batch_size]
                    batch_train_pos_items = self.train_pos_items[self.train_batch_id:self.train_batch_id + batch_size]
                    batch_train_neg_items = self.train_neg_items[self.train_batch_id:self.train_batch_id + batch_size]
                    batch_train_pos_features = self.train_pos_features[self.train_batch_id:self.train_batch_id + batch_size]
                    batch_train_neg_features = self.train_neg_features[self.train_batch_id:self.train_batch_id + batch_size]

                    self.train_batch_id = self.train_batch_id + batch_size

                return [batch_train_users, batch_train_pos_items, batch_train_neg_items,
                        batch_train_pos_features, batch_train_neg_features]


    def get_test_batch_data(self, batch_size):
        if self.args.evaluate == 'rmse':
            l = len(self.test_users)
            if self.test_batch_id + batch_size > l:
                batch_test_users = self.test_users[self.test_batch_id:] + self.test_users[
                                                                          :self.test_batch_id + batch_size - l]
                batch_test_items = self.test_items[self.test_batch_id:] + self.test_items[
                                                                          :self.test_batch_id + batch_size - l]
                batch_test_a_uis = self.test_a_uis[self.test_batch_id:] + self.test_a_uis[
                                                                          :self.test_batch_id + batch_size - l]
                self.test_batch_id = self.test_batch_id + batch_size - l

            else:
                batch_test_users = self.test_users[self.test_batch_id:self.test_batch_id + batch_size]
                batch_test_items = self.test_items[self.test_batch_id:self.test_batch_id + batch_size]
                batch_test_a_uis = self.test_a_uis[self.test_batch_id:self.test_batch_id + batch_size]
                self.test_batch_id = self.test_batch_id + batch_size

            return [batch_test_users, batch_test_items, batch_test_a_uis]
        else:
            l = len(self.test_users)
            if self.test_batch_id + batch_size > l:
                batch_test_users = self.test_users[self.test_batch_id:] + self.test_users[:self.test_batch_id + batch_size - l]
                batch_test_items = self.test_items[self.test_batch_id:] + self.test_items[:self.test_batch_id + batch_size - l]
                self.test_batch_id = self.test_batch_id + batch_size - l

            else:
                batch_test_users = self.test_users[self.test_batch_id:self.test_batch_id + batch_size]
                batch_test_items = self.test_items[self.test_batch_id:self.test_batch_id + batch_size]
                self.test_batch_id = self.test_batch_id + batch_size

            return [batch_test_users, batch_test_items]


