import torch
import torch.nn as nn
import torch.nn.functional as F

class CharCNN(nn.Module):
  def __init__(self, embedding_num=115, embedding_dim=100, filters=[[3, 50]], output_dim=50):
    super(CharCNN, self).__init__()
    self.embedding_num = embedding_num
    self.embedding_dim = embedding_dim
    self.conv_output_dim = sum([x[1] for x in filters])
    self.convs = nn.ModuleList([nn.Conv1d(embedding_dim, x[1], x[0]) for x in filters])
    self.char_embedding = nn.Embedding(embedding_num, embedding_dim)
    self.linear = nn.Linear(self.conv_output_dim, output_dim)

  def forward(self, char_input_ids):
    char_embed = self.char_embedding(char_input_ids).transpose(1, 2)
    convs = [conv(char_embed) for conv in self.convs]
    convs = [F.relu(conv) for conv in convs]
    convs = [F.max_pool1d(conv, conv.size(2)) for conv in convs]
    convs = torch.squeeze(torch.cat(convs, 1), 2)
    convs_rep = F.leaky_relu(self.linear(convs))
    return convs_rep

