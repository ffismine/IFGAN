import torch
import torch.nn as nn
import torch.nn.functional as functional
import numpy as np
import os
import math


class Generator2(nn.Module):
    def __init__(self, conf):
        super(Generator2, self).__init__()

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

    def score(self, can_features):
        with torch.no_grad():
            score = self.ranker(can_features)
            return score

    def forward(self, reward, pred_data, sample_index):
        score = self.ranker_dp(pred_data)

        opt_prob = torch.sigmoid(score)[sample_index]

        loss = -(torch.log(opt_prob) * reward).mean()
        return loss

    def reward(self, features):
        with torch.no_grad():
            score = self.ranker(features)
            reward = torch.log(1 + torch.exp(score))
            return reward


def train_g2(dataset, device, g2, g2_optimizer, g1, d):
    for user in dataset.get_pos_users():
        pos_items = dataset.get_pos_items(user)
        can_items = dataset.get_items(user)

        can_features = [dataset.get_features(user, item) for item in can_items]
        can_features = torch.from_numpy(np.asarray(can_features)).to(device).float()

        # get score to calculate probability
        can_score = g1.score(can_features).cpu().numpy()

        exp_rating = np.exp(can_score)
        prob = exp_rating / np.sum(exp_rating)

        # sample same number of neg and pos
        neg_index = np.random.choice(np.arange(len(can_items)), size=[len(pos_items)], p=np.ndarray.flatten(prob))
        neg_items = np.array(can_items)[neg_index]

        pos_features = [dataset.get_features(user, item) for item in pos_items]
        neg_features = [dataset.get_features(user, item) for item in neg_items]
        pos_features = torch.from_numpy(np.asarray(pos_features)).to(device).float()
        neg_features = torch.from_numpy(np.asarray(neg_features)).to(device).float()

        # calculate reward
        neg_reward = d.reward(pos_features, neg_features)

        g2_optimizer.zero_grad()
        g2_loss = g2(neg_reward, can_features, neg_index)

        g2_loss.backward()
        g2_optimizer.step()
    return g2_loss.item()
