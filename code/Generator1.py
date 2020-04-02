import torch
import torch.nn as nn
import torch.nn.functional as functional
import numpy as np
import os
import math


class Generator1(nn.Module):
    def __init__(self, conf):
        super(Generator1, self).__init__()

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

    def forward(self, pos_data, neg_data):
        pos_score = self.ranker_dp(pos_data)
        neg_score = self.ranker_dp(neg_data)

        loss = -functional.logsigmoid(pos_score - neg_score).mean()
        return loss

    def score_with_loss(self, can_features):
        score = self.ranker(can_features)
        return score


# %%
def idcg(n_rel):
    # Assuming binary relevance.
    nums = np.ones(n_rel)
    denoms = np.log2(np.arange(n_rel) + 1 + 1)
    return (nums / denoms).sum()


def train_g1(train_loader, device, conf, g1, g1_optimizer, g2):
    for i, (items, pos_num) in enumerate(train_loader):
        items = items.to(device).float().view(-1, conf['feature_size'])
        n_items = items.size(0)
        n_rel = int(pos_num[0])
        n_irr = n_items - n_rel

        # print(n_items,n_rel)
        for i in range(1):
            items_scores = g1.score_with_loss(items)
            (sorted_scores, sorted_idxs) = items_scores.sort(dim=0, descending=True)
            items_ranks = torch.zeros(n_items).to(device).long()
            items_ranks[sorted_idxs] = 1 + torch.arange(n_items).view((n_items, 1)).to(device)
            items_ranks = items_ranks.view((n_items, 1)).float()
            diffs = items_scores[:n_rel] - items_scores[n_rel:].view(n_irr)
            #
            exped = diffs.exp()

            N = 1 / idcg(n_rel)
            ndcg_diffs = (1 / (1 + items_ranks[:n_rel])).log2() - (1 / (1 + items_ranks[n_rel:])).log2().view(n_irr)
            lamb_updates = -1 / (1 + exped) * N * ndcg_diffs.abs()

            # See section 6.1 in [1], but lambdas have opposite signs from [2].
            lambs = torch.zeros((n_items, 1)).to(device)
            lambs[:n_rel] -= lamb_updates.sum(dim=1, keepdim=True)
            lambs[n_rel:] += lamb_updates.sum(dim=0, keepdim=True).t()

            reward = g2.reward(items)
            lambs = lambs * reward
            lambs = -lambs

            g1.zero_grad()
            items_scores.backward(lambs)
            g1_optimizer.step()