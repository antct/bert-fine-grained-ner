import torch
import torch.nn as nn
from config import args


class BiLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_dirs):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_dirs = num_dirs
        self.rnn = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size // num_dirs,
            num_layers=num_layers,
            bias=True,
            batch_first=True,
            bidirectional=(num_dirs == 2)
        )

    def init_states(self, x, device):
        h0 = torch.zeros(self.num_layers*self.num_dirs, x.size(0),
                         self.hidden_size//self.num_dirs).to(device)
        c0 = torch.zeros(self.num_layers*self.num_dirs, x.size(0),
                         self.hidden_size//self.num_dirs).to(device)
        return (h0, c0)

    def forward(self, x, mask, pack=True):
        s = self.init_states(x, x.device)

        if pack:
            lens = mask.sum(1)
            sorted_lens, sorted_idx = torch.sort(lens, descending=True)
            _, unsorted_idx = torch.sort(sorted_idx)

            x = x.index_select(0, sorted_idx)
            packed_x = nn.utils.rnn.pack_padded_sequence(
                x, sorted_lens.cpu().numpy(), batch_first=True)

            self.rnn.flatten_parameters()
            out, _ = self.rnn(packed_x, s)
            padded_out, _ = nn.utils.rnn.pad_packed_sequence(
                out, batch_first=True)
            out = padded_out.index_select(0, unsorted_idx)
        else:
            self.rnn.flatten_parameters()
            out, _ = self.rnn(x, s)
        return out
