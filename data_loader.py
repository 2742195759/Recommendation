import pickle
import random

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
        train_data_path = path + 'train_dict'
        self.train_data = pickle.load(open(train_data_path, 'rb'))
        test_data_path = path + 'test_dict'
        self.test_data = pickle.load(open(test_data_path, 'rb'))
        A_path = path + 'A'
        self.A = pickle.load(open(A_path, 'rb'))
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

        # build for model testing
        self.item_candidates = []
        for k, v in self.test_data.items():
            item = int(k.split('@')[1])
            if item not in self.item_candidates:
                self.item_candidates.append(item)

        self.item_candidates = random.sample(self.item_candidates, 100)
        self.grund_truth = dict()

        for k, v in self.test_data.items():
            user = int(k.split('@')[0])
            item = int(k.split('@')[1])
            if item in self.item_candidates:
                if user not in self.grund_truth.keys():
                    self.grund_truth[user] = [item]
                else:
                    self.grund_truth[user].append(item)

    def make_data(self):
        self.get_train_raw_data()
        self.get_test_raw_data()

    def get_train_raw_data(self):
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
            pass

    def get_train_batch_data(self, batch_size):
        l = len(self.train_users)
        if self.train_batch_id + batch_size > l:
            batch_train_users = self.train_users[self.train_batch_id:] + self.train_users[:self.train_batch_id + batch_size - l]
            batch_train_items = self.train_items[self.train_batch_id:] + self.train_items[:self.train_batch_id + batch_size - l]
            batch_train_features = self.train_features[self.train_batch_id:] + self.train_features[:self.train_batch_id + batch_size - l]
            batch_train_a_uis = self.train_a_uis[self.train_batch_id:] + self.train_a_uis[:self.train_batch_id + batch_size - l]
            batch_train_x_ufs = self.train_x_ufs[self.train_batch_id:] + self.train_x_ufs[:self.train_batch_id + batch_size - l]
            batch_train_y_ifs = self.train_y_ifs[self.train_batch_id:] + self.train_y_ifs[:self.train_batch_id + batch_size - l]
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
            pass


