import torch
import torch.nn as nn

from crf import CRF
from cnn import CharCNN
from bilstm import BiLSTM
from attention import MultiHeadAttention
from bert.modeling import BertModel
from util import VOCAB, CHAR_VOCAB
from config import args


class BertNERNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.bert = BertModel.from_pretrained('bert_base/')

        if args.bert_freeze:
            for param in self.bert.parameters():
                param.requires_grad = False

        self.lstm = BiLSTM(
            input_size=args.bert_hidden_size+args.cnn_output_size,
            hidden_size=args.rnn_hidden_size+args.cnn_output_size,
            num_layers=args.rnn_num_layers,
            num_dirs=args.rnn_num_dirs
        )

        self.lstm_dropout = nn.Dropout(
            p=args.rnn_dropout
        )

        self.cnn = CharCNN(
            embedding_num=len(CHAR_VOCAB),
            embedding_dim=args.cnn_embedding_dim,
            filters=eval(args.cnn_filters),
            output_size=args.cnn_output_size
        )

        self.crf = CRF(
            target_size=len(VOCAB)+2,
            use_cuda=args.crf_use_cuda
        )

        self.linear = nn.Linear(
            in_features=args.rnn_hidden_size+args.cnn_output_size,
            out_features=len(VOCAB)+2
        )

        self.attn = MultiHeadAttention(
            model_dim=args.rnn_hidden_size+args.cnn_output_size,
            num_heads=args.attn_num_heads,
            dropout=args.attn_dropout
        )

        self.feat_dropout = nn.Dropout(
            p=args.feat_dropout
        )

    def forward(self, input_ids, token_type_ids, attention_mask, input_char_ids, labels=None):
        encoded_layers, _ = self.bert(
            input_ids=input_ids,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask
        )
        bert_feats = encoded_layers[-1]

        char_feats = self.cnn(input_char_ids)

        feats = torch.cat([bert_feats, char_feats], dim=2)

        feats = self.feat_dropout(feats)

        # a simple way to get mask
        # mask = x.data.gt(0).float()

        lstm_feats = self.lstm(feats, attention_mask, False)
        lstm_feats, _ = self.attn(lstm_feats, lstm_feats, lstm_feats, None)
        lstm_feats = self.lstm_dropout(lstm_feats)

        crf_feats = self.linear(lstm_feats)

        path_score, best_path = self.crf(
            feats=crf_feats,
            mask=attention_mask
        )

        if labels is not None:
            loss_value = self.crf.neg_log_likelihood_loss(
                feats=crf_feats,
                mask=attention_mask,
                tags=labels
            )
            return path_score, best_path, loss_value
        else:
            return path_score, best_path
