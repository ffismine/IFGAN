import torch
import torch.nn as nn
import torch.nn.functional as functional
import numpy as np
import os
import torch.utils.data


class Movielens:
    def __init__(self, dataset_dir):
        self.pool = {}
        self.pool['train'] = self._load_data(dataset_dir + '/train.txt')
        self.pool['test'] = self._load_data(dataset_dir + '/test.txt')
        self.pool['allofall'] = self._load_data_all(r'..\data\allofall.txt')

    def get_users(self, target='train'):
        if target in self.pool.keys():
            return list(self.pool[target].keys())

    def get_items(self, user, target='train'):
        if target in self.pool.keys():
            return list(self.pool[target][user].keys())

    def get_features(self, user, item, target='train'):
        if target in self.pool.keys():
            return self.pool[target][user][item]['f']

    def get_rank(self, user, item, target='train'):
        if target in self.pool.keys():
            return self.pool[target][user][item]['r']

    def get_pos_users(self, target='train'):
        if target in self.pool.keys():
            return list({user for user in self.get_users(target=target)
                         for item in self.get_items(user, target=target) if
                         self.get_rank(user, item, target=target) >= 5.0})


    def get_pos_items(self, user, target='train'):
        if target in self.pool.keys():
            return list(
                {item for item in self.get_items(user, target=target) if self.get_rank(user, item, target=target) >= 5.0})

    # load itemss and features for a user.
    def _load_data(self, file, feature_size=54):
        user_item_feature = {}
        with open(file) as f:
            for line in f:
                cols = line.strip().split()
                rank = cols[0]
                user = cols[1]
                item = cols[2]
                feature = []
                for i in range(3, 3 + feature_size):
                    feature.append(float(cols[i]))
                if user in user_item_feature.keys():
                    user_item_feature[user][item] = {'r': float(rank), 'f': np.array(feature)}
                else:
                    user_item_feature[user] = {item: {'r': float(rank), 'f': np.array(feature)}}
        return user_item_feature

    def _load_data_all(self, file, feature_size=54):  #1129
        user_item_feature = {}
        with open(file) as f:
            for line in f:
                cols = line.strip().split()
                rank = 0
                user = cols[0]
                item = cols[1]
                feature = []
                for i in range(2, 2 + feature_size):
                    feature.append(float(cols[i]))
                if user in user_item_feature.keys():
                    user_item_feature[user][item] = {'r': float(rank), 'f': np.array(feature)}
                else:
                    user_item_feature[user] = {item: {'r': float(rank), 'f': np.array(feature)}}
        return user_item_feature


class Dataset(Movielens):
    def __init__(self, batch_size, dataset_dir):
        Movielens.__init__(self, dataset_dir=dataset_dir)
        self.batch_size = batch_size
        self.items_pairs = []

    def set_items_pairs(self, g2, device):
        for user in self.get_pos_users():
            can_items = self.get_items(user)
            can_features = [self.get_features(user, item) for item in can_items]

            # can_score = sess.run(generator2.pred_score, feed_dict={generator2.pred_data: can_features})
            can_features = torch.from_numpy(np.asarray(can_features)).to(device).float()

            # get score to calculate probability
            can_score = g2.score(can_features).cpu().numpy()

            # softmax for candidate
            exp_rating = np.exp(can_score)
            prob = exp_rating / np.sum(exp_rating)

            pos_items = self.get_pos_items(user)
            neg_items = []
            for i in range(len(pos_items)):
                while True:
                    item = np.random.choice(can_items, p=np.ndarray.flatten(prob))
                    if item not in pos_items:
                        neg_items.append(item)
                        break

            for i in range(len(pos_items)):
                self.items_pairs.append((user, pos_items[i], neg_items[i]))

    def get_batches(self):
        size = len(self.items_pairs)
        # print(size)
        cut_off = size // self.batch_size

        for i in range(0, self.batch_size * cut_off, self.batch_size):
            batch_pairs = self.items_pairs[i:i + self.batch_size]
            yield np.asarray([self.get_features(p[0], p[1]) for p in batch_pairs]), np.asarray(
                [self.get_features(p[0], p[2]) for p in batch_pairs])


class Dataset_torch(torch.utils.data.Dataset):
    def __init__(self, movielens, target='train'):
        self.fea = []
        self.pos_num = []

        for q in movielens.get_users(target):
            fea = []
            if len(movielens.get_pos_items(q, target)) == 0:
                continue

            for p in movielens.get_pos_items(q, target):
                fea.append(movielens.get_features(q, p, target))

            for n in movielens.get_items(q, target):
                if movielens.get_rank(q, n, target) < 5.0:
                    fea.append(movielens.get_features(q, n, target))

            fea = np.stack(fea)

            self.fea.append(fea)
            self.pos_num.append(len(movielens.get_pos_items(q, target)))

    def __getitem__(self, idx):
        ret = (torch.from_numpy(self.fea[idx]), self.pos_num[idx])
        return ret


    def __len__(self):
        return len(self.fea)


