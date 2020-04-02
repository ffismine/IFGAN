import torch
import torch.nn as nn
import torch.nn.functional as functional
import numpy as np
import os
import math


class Discriminator(nn.Module):
    def __init__(self, conf):
        super(Discriminator, self).__init__()

        layer1 = nn.Linear(conf['feature_size'], conf['feature_size'])
        layer2 = nn.Linear(conf['feature_size'], 1)
        dp = nn.Dropout(0.5)
        tanh = nn.Tanh()

        self.ranker = torch.nn.Sequential(
            layer1,
            tanh,
            layer2,
        )

        self.ranker_dp = torch.nn.Sequential(
            layer1,
            # dp,
            tanh,
            layer2,
        )

    def reward(self, pos_features, neg_features):
        with torch.no_grad():
            pos_score = self.ranker(pos_features)
            neg_score = self.ranker(neg_features)

            reward = torch.log(1 + torch.exp(neg_score - pos_score))
            return reward

    def score(self, can_features):
        with torch.no_grad():
            score = self.ranker(can_features)
            return score

    def forward(self, pos_data, neg_data):
        pos_score = self.ranker_dp(pos_data)
        neg_score = self.ranker_dp(neg_data)

        loss = -functional.logsigmoid(pos_score - neg_score).mean()
        return loss


def train_d(dataset, device, g2, d, d_optimizer):
    dataset.set_items_pairs(g2, device)

    for input_pos, input_neg in dataset.get_batches():
        input_pos = torch.from_numpy(np.asarray(input_pos)).to(device).float()
        input_neg = torch.from_numpy(np.asarray(input_neg)).to(device).float()
        d_loss = d(input_pos, input_neg)
        d_optimizer.zero_grad()
        d_loss.backward()
        d_optimizer.step()

    return d_loss.item()
