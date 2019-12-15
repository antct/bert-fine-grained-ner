import torch
import torch.nn as nn
import torch.nn.functional as F

from bert import BertModel
from ernie import BertModel as ErnieModel
from cnn import CharCNN
from fusion import Fusion, Normalize
from bilstm import BiLSTM
from attention import SelfAttentiveSum, MultiHeadAttention
from util import labels, char_vocab
from config import args

class BertMentionETNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.bert = BertModel.from_pretrained('bert_large/')
        if args.bert_freeze:
            for param in self.bert.parameters():
                param.requires_grad = False

        self.context_dropout = nn.Dropout(args.context_dropout)
        self.mention_dropout = nn.Dropout(args.mention_dropout)
        
        self.layer_norm = nn.LayerNorm(args.bert_hidden_size)
        self.multi_head_atten = MultiHeadAttention(args.bert_hidden_size, num_heads=8, dropout=0.1)
        self.mention_char_atten = MultiHeadAttention(args.bert_hidden_size, num_heads=8, dropout=0.1)

        self.context_lstm = BiLSTM(
            input_size=args.bert_hidden_size,
            hidden_size=args.rnn_hidden_size,
            num_layers=args.rnn_num_layers,
            dropout=args.rnn_dropout,
            num_dirs=args.rnn_num_dirs
        )

        self.mention_lstm = BiLSTM(
            input_size=args.bert_hidden_size,
            hidden_size=args.rnn_hidden_size,
            num_layers=args.rnn_num_layers,
            dropout=args.rnn_dropout,
            num_dirs=args.rnn_num_dirs
        )

        self.context_attn_sum = SelfAttentiveSum(args.bert_hidden_size, 100)
        self.mention_attn_sum = SelfAttentiveSum(args.bert_hidden_size, 1)


        self.char_cnn = CharCNN(
            embedding_num=len(char_vocab),
            embedding_dim=args.cnn_embedding_dim,
            filters=eval(args.cnn_filters),
            output_dim=args.cnn_output_dim
        )

        self.linear = nn.Linear(
            in_features=2*args.bert_hidden_size + args.cnn_output_dim, 
            out_features=len(labels),
            bias=True
        )

        if args.interaction:
            self.mention_linear = nn.Linear(
                in_features=args.bert_hidden_size + args.cnn_output_dim,
                out_features=args.bert_hidden_size,
                bias=True
            )
            self.affinity_matrix = nn.Linear(args.bert_hidden_size, args.bert_hidden_size)
            self.fusion = Fusion(args.bert_hidden_size)
            self.normalize = Normalize()
            self.fusion_linear = nn.Linear(
                in_features=2*args.bert_hidden_size,
                out_features=len(labels),
                bias=True
            )


    def forward(self, context_input_ids, context_token_type_ids, context_attention_mask, \
        mention_start_idxs, mention_end_idxs, mention_mask, char_input_ids):

        # consider more about context vector and mention vector
        # now adpat a way that simply concat two vecots [mention, context]
        # can introduce attention mechanism to generate a more informative representation?
        # context
        encoderd_layers, pooled_output = self.bert(
            input_ids=context_input_ids,
            token_type_ids=context_token_type_ids,
            attention_mask=context_attention_mask,
        )
        context_embed = encoderd_layers[-1]
        context_embed = self.context_dropout(context_embed)
        context_rep = context_embed[:, 0, :]

        mention_embed = context_embed * mention_mask.unsqueeze(-1).float()

        if args.enhance_mention:
            mention_embed = []
            for i in range(len(mention_start_idxs)):
                cur_mention_embed = context_embed[i][mention_start_idxs[i][0] : mention_end_idxs[i][0]]
                cur_mention_embed, _score = self.multi_head_atten(cur_mention_embed.unsqueeze(0), cur_mention_embed.unsqueeze(0), context_rep[i].unsqueeze(0).unsqueeze(0))
                cur_mention_embed = cur_mention_embed.squeeze(1)
                mention_embed.append(cur_mention_embed)
            mention_embed = torch.cat(mention_embed, dim=0)
        else:
            mention_embed = torch.sum(self.layer_norm(mention_embed), 1) / (mention_end_idxs - mention_start_idxs).unsqueeze().float()

        # char
        char_embed = self.char_cnn(char_input_ids)

        mention_rep = torch.cat((char_embed, mention_embed), 1)
        mention_rep = self.mention_dropout(mention_rep)


        if args.interaction:
            mention_rep_proj = self.mention_linear(mention_rep)
            # [batch_size, 1, hidden_size] * [batch_size, hidden_size, hidden_size] * [batch_size, hidden_size, seq_length]
            affinity = self.affinity_matrix(mention_rep_proj.unsqueeze(1)).bmm(context_embed.transpose(2, 1))
            # [batch_size 1]
            seq_lens = context_attention_mask.sum(dim=1)
            
            m_over_c = self.normalize(affinity, seq_lens.tolist())
            # m_over_c = self.normalize(affinity, seq_lens.squeeze().tolist())
            # [batch_size, 1, seq_length] * [batch_size, seq_length, hidden_size]
            retrieved_context_rep = torch.bmm(m_over_c, context_embed)
            # [batch_size, hidden_size]
            fusioned_rep = self.fusion(retrieved_context_rep.squeeze(1), mention_rep_proj)

            rep = torch.cat([context_rep, fusioned_rep], dim=1)
            # rep = F.dropout(torch.cat([fusioned, context_rep], dim=1), 0.2, self.training)
            logits = self.fusion_linear(rep)

        else:
            rep = torch.cat((context_rep, mention_rep), dim=1)
            logits = self.linear(rep)

        return logits


class BertETNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.bert = BertModel.from_pretrained('bert_base/')
        if args.bert_freeze:
            for param in self.bert.parameters():
                param.requires_grad = False

        self.context_dropout = nn.Dropout(args.context_dropout)
        self.mention_dropout = nn.Dropout(args.mention_dropout)

        self.context_lstm = BiLSTM(
            input_size=args.bert_hidden_size,
            hidden_size=args.rnn_hidden_size,
            num_layers=args.rnn_num_layers,
            dropout=args.rnn_dropout,
            num_dirs=args.rnn_num_dirs
        )

        self.mention_lstm = BiLSTM(
            input_size=args.bert_hidden_size,
            hidden_size=args.rnn_hidden_size,
            num_layers=args.rnn_num_layers,
            dropout=args.rnn_dropout,
            num_dirs=args.rnn_num_dirs
        )

        self.context_attn_sum = SelfAttentiveSum(args.bert_hidden_size, 100)
        self.mention_attn_sum = SelfAttentiveSum(args.bert_hidden_size, 1)


        self.char_cnn = CharCNN(
            embedding_num=len(char_vocab),
            embedding_dim=args.cnn_embedding_dim,
            filters=eval(args.cnn_filters),
            output_dim=args.cnn_output_dim
        )

        self.linear = nn.Linear(
            in_features=2*args.bert_hidden_size + args.cnn_output_dim, 
            out_features=len(labels),
            bias=True
        )

        if args.interaction:
            self.mention_linear = nn.Linear(
                in_features=args.bert_hidden_size + args.cnn_output_dim,
                out_features=args.bert_hidden_size,
                bias=True
            )
            self.affinity_matrix = nn.Linear(args.bert_hidden_size, args.bert_hidden_size)
            self.fusion = Fusion(args.bert_hidden_size)
            self.normalize = Normalize()
            self.fusion_linear = nn.Linear(
                in_features=2*args.bert_hidden_size,
                out_features=len(labels),
                bias=True
            )


    def forward(self, context_input_ids, context_token_type_ids, context_attention_mask, \
        mention_input_ids, mention_token_type_ids, mention_attention_mask, char_input_ids):

        # consider more about context vector and mention vector
        # now adpat a way that simply concat two vecots [mention, context]
        # can introduce attention mechanism to generate a more informative representation?
        # context
        encoderd_layers, pooled_output = self.bert(
            input_ids=context_input_ids,
            token_type_ids=context_token_type_ids,
            attention_mask=context_attention_mask,
        )
        context_embed = encoderd_layers[-1]
        context_embed = self.context_dropout(context_embed)
        context_embed = self.context_lstm(context_embed, context_attention_mask)
        context_embed = context_embed.contiguous()
        context_rep, _ = self.context_attn_sum(context_embed)

        # mention
        encoded_layers, pooled_output = self.bert(
            input_ids=mention_input_ids,
            token_type_ids=mention_token_type_ids,
            attention_mask=mention_attention_mask,
        )
        mention_embed = encoded_layers[-1]

        if args.enhance_mention:
            mention_embed = self.mention_lstm(mention_embed, mention_attention_mask)
            mention_embed = mention_embed.contiguous()
            mention_embed, _ = self.mention_attn_sum(mention_embed)
        else:
            mention_embed = torch.sum(mention_embed, 1)
        # char
        char_embed = self.char_cnn(char_input_ids)

        mention_rep = torch.cat((char_embed, mention_embed), 1)
        mention_rep = self.mention_dropout(mention_rep)


        if args.interaction:
            mention_rep_proj = self.mention_linear(mention_rep)
            # [batch_size, 1, hidden_size] * [batch_size, hidden_size, hidden_size] * [batch_size, hidden_size, seq_length]
            affinity = self.affinity_matrix(mention_rep_proj.unsqueeze(1)).bmm(context_embed.transpose(2, 1))
            # [batch_size 1]
            seq_lens = context_attention_mask.sum(dim=1)

            m_over_c = self.normalize(affinity, seq_lens.tolist())
            # m_over_c = self.normalize(affinity, seq_lens.squeeze().tolist())
            # [batch_size, 1, seq_length] * [batch_size, seq_length, hidden_size]
            retrieved_context_rep = torch.bmm(m_over_c, context_embed)
            # [batch_size, hidden_size]
            fusioned_rep = self.fusion(retrieved_context_rep.squeeze(1), mention_rep_proj)

            rep = torch.cat([context_rep, fusioned_rep], dim=1)
            # rep = F.dropout(torch.cat([fusioned, context_rep], dim=1), 0.2, self.training)
            logits = self.fusion_linear(rep)

        else:
            rep = torch.cat((context_rep, mention_rep), dim=1)
            logits = self.linear(rep)

        return logits


class ErnieETNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.ernie, _ = ErnieModel.from_pretrained('ernie_base/')
        if args.bert_freeze:
            for param in self.ernie.parameters():
                param.requires_grad = False

        self.context_dropout = nn.Dropout(args.context_dropout)
        self.mention_dropout = nn.Dropout(args.mention_dropout)

        self.context_lstm = BiLSTM(
            input_size=args.bert_hidden_size,
            hidden_size=args.rnn_hidden_size,
            num_layers=args.rnn_num_layers,
            dropout=args.rnn_dropout,
            num_dirs=args.rnn_num_dirs
        )

        self.mention_lstm = BiLSTM(
            input_size=args.bert_hidden_size,
            hidden_size=args.rnn_hidden_size,
            num_layers=args.rnn_num_layers,
            dropout=args.rnn_dropout,
            num_dirs=args.rnn_num_dirs
        )

        self.context_attn_sum = SelfAttentiveSum(args.bert_hidden_size, 100)
        self.mention_attn_sum = SelfAttentiveSum(args.bert_hidden_size, 1)


        self.char_cnn = CharCNN(
            embedding_num=len(char_vocab),
            embedding_dim=args.cnn_embedding_dim,
            filters=eval(args.cnn_filters),
            output_dim=args.cnn_output_dim
        )

        self.linear = nn.Linear(
            in_features=2*args.bert_hidden_size + args.cnn_output_dim, 
            out_features=len(labels),
            bias=True
        )

        if args.interaction:
            self.mention_linear = nn.Linear(
                in_features=args.bert_hidden_size + args.cnn_output_dim,
                out_features=args.bert_hidden_size,
                bias=True
            )
            self.affinity_matrix = nn.Linear(args.bert_hidden_size, args.bert_hidden_size)
            self.fusion = Fusion(args.bert_hidden_size)
            self.normalize = Normalize()
            self.fusion_linear = nn.Linear(
                in_features=2*args.bert_hidden_size,
                out_features=len(labels),
                bias=True
            )
        
    def forward(self, context_input_ids, context_token_type_ids, context_attention_mask, \
        context_input_ent, context_ent_mask, \
        mention_input_ids, mention_token_type_ids, mention_attention_mask, \
        mention_input_ent, mention_ent_mask, char_input_ids):

        encoderd_layers, pooled_output = self.ernie(
            input_ids=context_input_ids,
            token_type_ids=context_token_type_ids,
            attention_mask=context_attention_mask,
            input_ent=context_input_ent,
            ent_mask=context_ent_mask
        )
        context_embed = encoderd_layers[-1]
        context_embed = self.context_dropout(context_embed)
        context_embed = self.context_lstm(context_embed, context_attention_mask)
        context_embed = context_embed.contiguous()
        context_rep, _ = self.context_attn_sum(context_embed)

        # mention
        encoded_layers, pooled_output = self.ernie(
            input_ids=mention_input_ids,
            token_type_ids=mention_token_type_ids,
            attention_mask=mention_attention_mask,
            input_ent=mention_input_ent,
            ent_mask=mention_ent_mask
        )
        mention_embed = encoded_layers[-1]

        if args.enhance_mention:
            mention_embed = self.mention_lstm(mention_embed, mention_attention_mask)
            mention_embed = mention_embed.contiguous()
            mention_embed, _ = self.mention_attn_sum(mention_embed)
        else:
            mention_embed = torch.sum(mention_embed, 1)
        char_embed = self.char_cnn(char_input_ids)
        mention_rep = torch.cat((char_embed, mention_embed), 1)
        mention_rep = self.mention_dropout(mention_rep)

        if args.interaction:
            mention_rep_proj = self.mention_linear(mention_rep)
            affinity = self.affinity_matrix(mention_rep_proj.unsqueeze(1)).bmm(context_embed.transpose(2, 1))
            seq_lens = context_attention_mask.sum(dim=1)
            m_over_c = self.normalize(affinity, seq_lens.squeeze().tolist())
            retrieved_context_rep = torch.bmm(m_over_c, context_embed)
            fusioned_rep = self.fusion(retrieved_context_rep.squeeze(1), mention_rep_proj)
            rep = torch.cat([context_rep, fusioned_rep], dim=1)
            logits = self.fusion_linear(rep)
        else:
            rep = torch.cat((context_rep, mention_rep), dim=1)
            logits = self.linear(rep)

        return logits


# yet a simple network, just use bert + FC, hard to converge
class BertNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.bert = BertModel.from_pretrained('bert_base/')
        if args.bert_freeze:
            for param in self.bert.parameters():
                param.requires_grad = False
        self.dropout = nn.Dropout(args.bert_dropout)
        self.linear = nn.Linear(args.bert_hidden_size, len(labels), bias=True)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None):
        _, pooled_output = self.bert(
            input_ids=input_ids,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask,
            output_all_encoded_layers=False
        )
        pooled_output = self.dropout(pooled_output)
        logits = self.linear(pooled_output)
        return logits


class ErnieNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.ernie, _ = ErnieModel.from_pretrained('ernie_base/')
        if args.bert_freeze:
            for param in self.ernie.parameters():
                param.requires_grad = False
        self.dropout = nn.Dropout(args.bert_dropout)
        self.linear = nn.Linear(args.bert_hidden_size, len(labels), bias=True)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, input_ent=None, ent_mask=None):
        _, pooled_output = self.ernie(
            input_ids=input_ids,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask,
            input_ent=input_ent,
            ent_mask=ent_mask,
            output_all_encoded_layers=False
        )
        pooled_output = self.dropout(pooled_output)
        logits = self.linear(pooled_output)
        return logits
