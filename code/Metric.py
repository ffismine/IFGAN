import torch
import torch.nn as nn
import torch.nn.functional as functional
import numpy as np
import os
import math

def precision_recall_ndcg_at_k(role, dataset, device, k=5):
    idcg_k = 0
    dcg_k = 0
    for user in dataset.get_pos_users(target='test'):
        pos_items_train = dataset.get_pos_items(user, target='train')
        pred_items_allofall = dataset.get_items(user, target='allofall')   #1682
        pred_items = [item for item in pred_items_allofall if item not in pos_items_train]
        items_test = dataset.get_items(user, target='test')

        if len(pred_items) < k:
            continue

        pred_features = np.asarray([dataset.get_features(user, item, target='allofall') for item in pred_items])
        pred_features = torch.from_numpy(pred_features).to(device).float()
        pred_score = role.score(pred_features).view(-1).cpu().tolist()
        pred_item_score = sorted(zip(pred_items, pred_score), key=lambda x: x[1], reverse=True)
        ranked_list=[]
        for i in range(0, k):
            item, _ = pred_item_score[i]
            ranked_list.append(item)
        n_k = k if len(items_test) > k else len(items_test)
        for i in range(n_k):
            idcg_k += 1 / math.log(i + 2, 2)

        b1 = ranked_list
        b2 = items_test
        s2 = set(b2)
        hits = [(idx, val) for idx, val in enumerate(b1) if val in s2]
        count = len(hits)

        for c in range(count):
            dcg_k += 1 / math.log(hits[c][0] + 2, 2)

        return float(count / k), float(count / len(items_test)), float(dcg_k / idcg_k)

def map_mrr(role, dataset, device):
    ap = 0
    map = 0
    mrr = 0
    for user in dataset.get_pos_users(target='test'):
        pos_items_train = dataset.get_pos_items(user, target='train')
        pred_items_allofall = dataset.get_items(user, target='allofall')   #1682
        pred_items = [item for item in pred_items_allofall if item not in pos_items_train]
        items_test = dataset.get_items(user, target='test')

        pred_features = np.asarray([dataset.get_features(user, item, target='allofall') for item in pred_items])
        pred_features = torch.from_numpy(pred_features).to(device).float()
        pred_score = role.score(pred_features).view(-1).cpu().tolist()
        pred_item_score = sorted(zip(pred_items, pred_score), key=lambda x: x[1], reverse=True)
        ranked_list=[]
        for i in range(len(pred_item_score)):
            item, _ = pred_item_score[i]
            ranked_list.append(item)

        b1 = ranked_list
        b2 = items_test
        s2 = set(b2)
        hits = [(idx, val) for idx, val in enumerate(b1) if val in s2]
        count = len(hits)

        for c in range(count):
            ap += (c + 1) / (hits[c][0] + 1)

        if count != 0:
            mrr = 1 / (hits[0][0] + 1)

        if count != 0:
            map = ap / count


        return map, mrr