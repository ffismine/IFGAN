import torch
import torch.nn as nn
import torch.nn.functional as functional
import numpy as np
import os
import math

os.environ["CUDA_VISIBLE_DEVICES"] = '0'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

conf = {
    'feature_size': 54,  #1129
    'num_epochs': 500,
    'batch_size': 32,
    'dataset_dir' : r'..\data' 
}

from LETOR import Dataset, Dataset_torch
dataset = Dataset(conf['batch_size'], dataset_dir=conf['dataset_dir'])

train_dataset = Dataset_torch(dataset, target='train')

train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=1, shuffle=False)

from Generator1 import Generator1, train_g1
from Generator2 import Generator2, train_g2
from Discriminator import Discriminator, train_d
from Metric import precision_recall_ndcg_at_k, map_mrr

g1 = Generator1 (conf).to(device)
d = Discriminator(conf).to(device)
g2 = Generator2(conf).to(device)

g1_optimizer = torch.optim.SGD(g1.parameters(), 0.000001)
d_optimizer = torch.optim.Adam(d.parameters(), 0.00001)
g2_optimizer = torch.optim.Adam(g2.parameters(), 0.00001)

ndcg3_g1 = []
precision3_g1 = []
recall3_g1 = []
ndcg5_g1 = []
precision5_g1 = []
recall5_g1 = []
ndcg10_g1 = []
precision10_g1 = []
recall10_g1 = []
map_g1 = []
mrr_g1 = []

ndcg3_g2 = []
precision3_g2 = []
recall3_g2 = []
ndcg5_g2 = []
precision5_g2 = []
recall5_g2 = []
ndcg10_g2 = []
precision10_g2 = []
recall10_g2 = []
map_g2 = []
mrr_g2 = []

ndcg3_d = []
precision3_d = []
recall3_d = []
ndcg5_d = []
precision5_d = []
recall5_d = []
ndcg10_d = []
precision10_d = []
recall10_d = []
map_d = []
mrr_d = []

g1_ev = [[], [], [], [], [], [], [], [], []]
g2_ev = [[], [], [], [], [], [], [], [], []]
d_ev = [[], [], [], [], [], [], [], [], []]
for epoch in range(conf['num_epochs']):
    print(epoch)
    # 1
    train_g1(train_loader, device, conf, g1, g1_optimizer, g2)
    g2_loss = train_g2(dataset, device, g2, g2_optimizer, g1, d)

    # 2
    g2_loss = train_g2(dataset, device, g2, g2_optimizer, g1, d)
    d_loss = train_d(dataset, device, g1, d, d_optimizer)

    """
    Metric Part
    """

    precision_recall_ndcg_at_3 = precision_recall_ndcg_at_k(g1, dataset, device, k=3)
    precision_recall_ndcg_at_5 = precision_recall_ndcg_at_k(g1, dataset, device, k=5)
    precision_recall_ndcg_at_10 = precision_recall_ndcg_at_k(g1, dataset, device, k=10)
    map_mrr_g1 = map_mrr(g1, dataset, device)

    precision3_g1.append(precision_recall_ndcg_at_3[0])
    recall3_g1.append(precision_recall_ndcg_at_3[1])
    ndcg3_g1.append(precision_recall_ndcg_at_3[2])
    precision5_g1.append(precision_recall_ndcg_at_5[0])
    recall5_g1.append(precision_recall_ndcg_at_5[1])
    ndcg5_g1.append(precision_recall_ndcg_at_5[2])
    precision10_g1.append(precision_recall_ndcg_at_10[0])
    recall10_g1.append(precision_recall_ndcg_at_10[1])
    ndcg10_g1.append(precision_recall_ndcg_at_10[2])
    map_g1.append(map_mrr_g1[0])
    mrr_g1.append(map_mrr_g1[1])

    precision_recall_ndcg_at_3 = precision_recall_ndcg_at_k(g2, dataset, device, k=3)
    precision_recall_ndcg_at_5 = precision_recall_ndcg_at_k(g2, dataset, device, k=5)
    precision_recall_ndcg_at_10 = precision_recall_ndcg_at_k(g2, dataset, device, k=10)
    map_mrr_g2 = map_mrr(g2, dataset, device)
    precision3_g2.append(precision_recall_ndcg_at_3[0])
    recall3_g2.append(precision_recall_ndcg_at_3[1])
    ndcg3_g2.append(precision_recall_ndcg_at_3[2])
    precision5_g2.append(precision_recall_ndcg_at_5[0])
    recall5_g2.append(precision_recall_ndcg_at_5[1])
    ndcg5_g2.append(precision_recall_ndcg_at_5[2])
    precision10_g2.append(precision_recall_ndcg_at_10[0])
    recall10_g2.append(precision_recall_ndcg_at_10[1])
    ndcg10_g2.append(precision_recall_ndcg_at_10[2])
    map_g2.append(map_mrr_g2[0])
    mrr_g2.append(map_mrr_g2[1])

    precision_recall_ndcg_at_3 = precision_recall_ndcg_at_k(d, dataset, device, k=3)
    precision_recall_ndcg_at_5 = precision_recall_ndcg_at_k(d, dataset, device, k=5)
    precision_recall_ndcg_at_10 = precision_recall_ndcg_at_k(d, dataset, device, k=10)
    map_mrr_d = map_mrr(d, dataset, device)
    precision3_d.append(precision_recall_ndcg_at_3[0])
    recall3_d.append(precision_recall_ndcg_at_3[1])
    ndcg3_d.append(precision_recall_ndcg_at_3[2])
    precision5_d.append(precision_recall_ndcg_at_5[0])
    recall5_d.append(precision_recall_ndcg_at_5[1])
    ndcg5_d.append(precision_recall_ndcg_at_5[2])
    precision10_d.append(precision_recall_ndcg_at_10[0])
    recall10_d.append(precision_recall_ndcg_at_10[1])
    ndcg10_d.append(precision_recall_ndcg_at_10[2])
    map_d.append(map_mrr_d[0])
    mrr_d.append(map_mrr_d[1])

    g1_ = [precision5_g1, precision10_g1, recall5_g1, recall10_g1, ndcg3_g1, ndcg5_g1, ndcg10_g1, map_g1, mrr_g1]
    g2_ = [precision5_g2, precision10_g2, recall5_g2, recall10_g2, ndcg3_g2, ndcg5_g2, ndcg10_g2, map_g2, mrr_g2]
    d_ = [precision5_d, precision10_d, recall5_d, recall10_d, ndcg3_d, ndcg5_d, ndcg10_d, map_d, mrr_d]

    for i in range(9):
        g1_ev[i].append(np.mean(g1_[i]))
        g2_ev[i].append(np.mean(g2_[i]))
        d_ev[i].append(np.mean(d_[i]))


    if epoch%2 == 0:

        print('\np5_g1_{} p10_g1_{} r5_g1_{} r10_g1_{} ndcg3_g1_{} ndcg5_g1_{} ndcg10_g1_{} map_g1_{} mrr_g1_{}'
              '\np5_g2_{} p10_g2_{} r5_g2_{} r10_g2_{} ndcg3_g2_{} ndcg5_g2_{} ndcg10_g2_{} map_g2_{} mrr_g2_{}'
              '\npp5_d_{} p10_d_{} r5_d_{} r10_d_{} ndcg3_d_{} ndcg5_d_{} ndcg10_d_{} map_d_{} mrr_d_{}\n'
              .format(np.mean(g1_ev[0]), np.mean(g1_ev[1]), np.mean(g1_ev[2]), np.mean(g1_ev[3]), np.mean(g1_ev[4]), np.mean(g1_ev[5]), np.mean(g1_ev[6]), np.mean(g1_ev[7]), np.mean(g1_ev[8]),
                      np.mean(g2_ev[0]), np.mean(g2_ev[1]), np.mean(g2_ev[2]), np.mean(g2_ev[3]), np.mean(g2_ev[4]), np.mean(g2_ev[5]), np.mean(g2_ev[6]), np.mean(g2_ev[7]), np.mean(g2_ev[8]),
                      np.mean(d_ev[0]), np.mean(d_ev[1]), np.mean(d_ev[2]), np.mean(d_ev[3]), np.mean(d_ev[4]), np.mean(d_ev[5]), np.mean(d_ev[6]), np.mean(d_ev[7]), np.mean(d_ev[8])))

