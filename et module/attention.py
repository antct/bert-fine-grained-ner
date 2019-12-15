import torch
import torch.nn as nn
import numpy as np

class SelfAttentiveSum(nn.Module):
  def __init__(self, output_dim, hidden_dim):
    super(SelfAttentiveSum, self).__init__()
    self.hidden_dim = hidden_dim
    self.act = nn.ReLU()
    self.softmax = nn.Softmax(dim=1)
    self.key_hidden = nn.Linear(output_dim, hidden_dim, bias=False)
    self.key_output = nn.Linear(hidden_dim, 1, bias=False)

  def forward(self, input_embed):
    input_embed_squeezed = input_embed.view(-1, input_embed.size()[2])
    keys = self.act(self.key_hidden(input_embed_squeezed))
    if self.hidden_dim == 1:
      keys = keys.view(input_embed.size()[0], -1)
    else:
      keys = self.key_output(keys).view(input_embed.size()[0], -1)

    weighted_keys = self.softmax(keys).view(input_embed.size()[0], -1, 1)
    weighted_values = torch.sum(weighted_keys * input_embed, 1)
    return weighted_values, weighted_keys


class MultiHeadAttention(nn.Module):
    def __init__(self, model_dim=512, num_heads=8, dropout=0.0):
        super(MultiHeadAttention, self).__init__()

        self.dim_per_head = model_dim // num_heads
        self.num_heads = num_heads
        self.linear_k = nn.Linear(model_dim, self.dim_per_head * num_heads)
        self.linear_v = nn.Linear(model_dim, self.dim_per_head * num_heads)
        self.linear_q = nn.Linear(model_dim, self.dim_per_head * num_heads)

        self.dot_product_attention = ScaledDotProductAttention(dropout)
        self.linear_final = nn.Linear(model_dim, model_dim)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(model_dim)

    def forward(self, key, value, query, mask=None):
        residual = query

        dim_per_head = self.dim_per_head
        num_heads = self.num_heads
        batch_size = key.size(0)

        # linear projection
        key = self.linear_k(key)
        value = self.linear_v(value)
        query = self.linear_q(query)

        # split by heads
        key = key.view(batch_size * num_heads, -1, dim_per_head)
        value = value.view(batch_size * num_heads, -1, dim_per_head)
        query = query.view(batch_size * num_heads, -1, dim_per_head)

        if mask is not None:
            mask = mask.repeat(num_heads, 1, 1)
        # scaled dot product attention
        scale = (key.size(-1) // num_heads) ** -0.5
        context, attention = self.dot_product_attention(
            query, key, value, scale, mask)

        # concat heads
        context = context.view(batch_size, -1, dim_per_head * num_heads)

        # final linear projection
        output = self.linear_final(context)

        # dropout
        output = self.dropout(output)

        # add residual and norm layer
        output = self.layer_norm(residual + output)

        return output, attention


class ScaledDotProductAttention(nn.Module):
    def __init__(self, attention_dropout=0.0):
        super(ScaledDotProductAttention, self).__init__()
        self.dropout = nn.Dropout(attention_dropout)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, q, k, v, scale=None, mask=None):
        attention = torch.bmm(q, k.transpose(1, 2))
        if scale:
            attention = attention * scale
        if mask is not None:
            attention = attention.masked_fill_(mask, -np.inf)
        attention = self.softmax(attention)
        attention = self.dropout(attention)
        context = torch.bmm(attention, v)
        return context, attention
