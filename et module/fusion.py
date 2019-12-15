import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class Normalize(nn.Module):
    def __init__(self):
        super(Normalize, self).__init__()
        pass

    def forward(self, score, seq_lens):
        # score: batch_size 1 seq_lens
        score_backup = score.data.clone()
        max_len = score.size(2)

        for i, seq_len in enumerate(seq_lens):
            if seq_len == max_len:
                continue
            score.data[i, :, int(seq_len):] = -1e30

        normalized_score = F.softmax(score, dim=-1)
        score.data.copy_(score_backup)
        return normalized_score


class Fusion(nn.Module):
    def __init__(self, hidden_size):
        super(Fusion, self).__init__()
        self.r = nn.Linear(hidden_size*3, hidden_size)
        self.g = nn.Linear(hidden_size*3, hidden_size)

    def gelu(self, x):
        return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))

    def forward(self, x, y):
        r_ = self.gelu(self.r(torch.cat([x, y, x-y], dim=-1)))
        g_ = torch.sigmoid(self.g(torch.cat([x, y, x-y], dim=-1)))
        return g_ * r_ + (1 - g_) * x
