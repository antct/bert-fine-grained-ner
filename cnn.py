import torch
import torch.nn as nn
import torch.nn.init as I
import torch.nn.functional as F


class CharCNN(nn.Module):
    def __init__(self, embedding_num, embedding_dim, filters, padding_idx=0, output_size=None):
        super(CharCNN, self).__init__()

        self.embedding_num = embedding_num
        self.embedding_dim = embedding_dim
        self.conv_output_size = sum([x[1] for x in filters])
        self.output_size = output_size if output_size else self.conv_output_size
        self.filters = filters

        self.char_embed = nn.Embedding(embedding_num, embedding_dim,
                                       padding_idx=padding_idx)
        self.convs = nn.ModuleList([nn.Conv2d(1, x[1], (x[0], embedding_dim))
                                    for x in filters])
        self.linear = nn.Linear(self.conv_output_size, self.output_size)

    def forward(self, inputs):
        batch_size, seq_len, char_dim = inputs.size()
        inputs_embed = self.char_embed.forward(inputs)
        inputs_embed = inputs_embed.view(-1, char_dim, self.embedding_dim)
        inputs_embed = inputs_embed.unsqueeze(1)

        conv_outputs = [F.tanh(conv.forward(inputs_embed)).squeeze(3)
                        for conv in self.convs]
        conv_outputs_max = [F.max_pool1d(i, i.size(2)).squeeze(2)
                            for i in conv_outputs]
        outputs = torch.cat(conv_outputs_max, 1)
        # outputs = F.tanh(self.linear(outputs))
        outputs = F.leaky_relu(self.linear(outputs))
        outputs = outputs.view(batch_size, seq_len, self.output_size)
        return outputs
