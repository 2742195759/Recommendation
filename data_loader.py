import numpy as np
import pickle
import random

class DataLoader():
    def __init__(self, args):
        self.args = args
        self.params = {}
        for k, v in vars(self.args).items():
            self.params[k] = v

        self.base_path = self.args.base_path
        self.category = self.args.category
        path = self.base_path + self.category
        train_data_path = path + 'train_dict'
        self.train_data = pickle.load(open(train_data_path, 'rb'))
        test_data_path = path + 'test_dict'
        self.test_data = pickle.load(open(test_data_path, 'rb'))

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
            print(k, v)
            user = int(k.split('@')[0])
            item = int(k.split('@')[1])
            for fos in v.split(':'):
                feature = int(fos.split('|')[0])
                print(user,item,feature)






    def get_test_raw_data(self):
        pass

    def get_train_batch_data(self, batch_size):
        l = len(self.train_answers)
        if self.train_batch_id + batch_size > l:
            batch_train_answers = self.train_answers[self.train_batch_id:] + self.train_answers[:self.train_batch_id + batch_size - l]
            batch_train_pos_descriptions = self.train_pos_descriptions[self.train_batch_id:] + self.train_pos_descriptions[:self.train_batch_id + batch_size - l]
            batch_train_neg_descriptions = self.train_neg_descriptions[self.train_batch_id:] + self.train_neg_descriptions[:self.train_batch_id + batch_size - l]

            batch_train_pos_questions = self.train_pos_questions[self.train_batch_id:] + self.train_pos_questions[:self.train_batch_id + batch_size - l]
            batch_train_neg_questions = self.train_neg_questions[self.train_batch_id:] + self.train_neg_questions[:self.train_batch_id + batch_size - l]

            batch_train_answer_masks = self.train_answer_masks[self.train_batch_id:] + self.train_answer_masks[:self.train_batch_id + batch_size - l]
            batch_train_pos_descriptions_masks = self.train_pos_descriptions_masks[self.train_batch_id:] + self.train_pos_descriptions_masks[:self.train_batch_id + batch_size - l]
            batch_train_neg_descriptions_masks = self.train_neg_descriptions_masks[self.train_batch_id:] + self.train_neg_descriptions_masks[:self.train_batch_id + batch_size - l]

            self.train_batch_id = self.train_batch_id + batch_size - l

        else:
            batch_train_answers = self.train_answers[self.train_batch_id:self.train_batch_id + batch_size]
            batch_train_pos_descriptions = self.train_pos_descriptions[self.train_batch_id:self.train_batch_id + batch_size]
            batch_train_neg_descriptions = self.train_neg_descriptions[self.train_batch_id:self.train_batch_id + batch_size]
            batch_train_pos_questions = self.train_pos_questions[self.train_batch_id:self.train_batch_id + batch_size]
            batch_train_neg_questions = self.train_neg_questions[self.train_batch_id:self.train_batch_id + batch_size]
            batch_train_answer_masks = self.train_answer_masks[self.train_batch_id:self.train_batch_id + batch_size]
            batch_train_pos_descriptions_masks = self.train_pos_descriptions_masks[self.train_batch_id:self.train_batch_id + batch_size]
            batch_train_neg_descriptions_masks = self.train_neg_descriptions_masks[self.train_batch_id:self.train_batch_id + batch_size]

            self.train_batch_id = self.train_batch_id + batch_size

        return [batch_train_answers, batch_train_pos_descriptions, batch_train_neg_descriptions, \
               batch_train_pos_questions, batch_train_neg_questions, batch_train_answer_masks, \
               batch_train_pos_descriptions_masks, batch_train_neg_descriptions_masks]

    def get_test_batch_data(self, batch_size):
        l = len(self.test_answers)
        if self.test_batch_id + batch_size > l:
            batch_test_answers = self.test_answers[self.test_batch_id:] + self.test_answers[:self.test_batch_id + batch_size - l]
            batch_test_pos_descriptions = self.test_pos_descriptions[self.test_batch_id:] + self.test_pos_descriptions[:self.test_batch_id + batch_size - l]
            batch_test_pos_questions = self.test_pos_questions[self.test_batch_id:] + self.test_pos_questions[:self.test_batch_id + batch_size - l]
            batch_test_pos_descriptions_masks = self.test_pos_descriptions_masks[self.test_batch_id:] + self.test_pos_descriptions_masks[:self.test_batch_id + batch_size - l]

            self.test_batch_id = self.test_batch_id + batch_size - l

        else:
            batch_test_answers = self.test_answers[self.test_batch_id:self.test_batch_id + batch_size]
            batch_test_pos_descriptions = self.test_pos_descriptions[self.test_batch_id:self.test_batch_id + batch_size]
            batch_test_pos_questions = self.test_pos_questions[self.test_batch_id:self.test_batch_id + batch_size]
            batch_test_pos_descriptions_masks = self.test_pos_descriptions_masks[self.test_batch_id:self.test_batch_id + batch_size]

            self.test_batch_id = self.test_batch_id + batch_size

        return [batch_test_answers, batch_test_pos_descriptions, \
               batch_test_pos_questions, batch_test_pos_descriptions_masks]

