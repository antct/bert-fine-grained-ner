import numpy as np
import torch
import json
import pickle

from torch.utils import data
from ernie.typing import BertTokenizer as ErnieTokenizer_label
from ernie.tokenization import BertTokenizer as ErnieTokenizer
from bert.tokenization import BertTokenizer
from tqdm import tqdm
from config import args
from log import Logger
from collections import defaultdict

logger = Logger('run.log', level='debug').getLogger()

# label for open type

logger.info('load char vocab')
char_dict = defaultdict(int)
char_vocab = [u"<unk>"]
with open('wiki/char_vocab.english.txt') as f:
    char_vocab.extend(c.strip() for c in f.readlines())
    char_dict.update({c: i for i, c in enumerate(char_vocab)})
logger.info('char vocab dim {}'.format(len(char_vocab)))

logger.info('load all labels')
labels = open('wiki/types.txt', 'r', encoding='utf-8-sig').read().strip().splitlines()
labels = [i.split('\t')[0] for i in labels]

logger.info('label count {}'.format(len(labels)))
label_map = {label: i for i, label in enumerate(labels)}

# prior
#logger.info('load prior and tune matrix')
#type2id = {label: i for i, label in enumerate(labels)}
#
#def create_prior(alpha=1.0):
#    num_types = len(type2id)
#    prior_numpy = np.zeros((num_types, num_types), dtype=np.float32)
#    for x in type2id.keys():
#        t = np.zeros(num_types, dtype=np.float32)
#        t[type2id[x]] = 1.0
#        for y in type2son[x]:
#            t[type2id[y]] = alpha
#        prior_numpy[:, type2id[x]] = t
#    return prior_numpy
#
#prior = torch.from_numpy(create_prior())
#tune = torch.from_numpy(np.transpose(create_prior(args.hierarchy_alpha)))


logger.info('load bert and ernie tokenizer')
ernie_tokenizer_label = ErnieTokenizer_label.from_pretrained('ernie_base/', do_lower_case=args.bert_low_case)

ernie_tokenizer = ErnieTokenizer.from_pretrained('ernie_base/', do_lower_case=args.bert_low_case)

bert_tokenizer = BertTokenizer.from_pretrained('bert_large/', do_lower_case=args.bert_low_case)


# dataset for open type
# left context token + mention_span + right_context_token
class OpenDataset(data.Dataset):
    def __init__(self, path):
        entries = open(path, 'r').read().strip().splitlines()
        self.left_context, self.right_context, self.mention_span, self.labels = [], [], [], []
        def trans(x): return x[x.rfind('/')+1:]
        for entry in entries:
            entry = dict(eval(entry))
            ys = entry['y_str']
            ys = [trans(i) for i in ys]
            label = [0] * len(labels)
            for y in ys:
                label[label_map[y]] = 1.0
            self.left_context.append(" ".join(entry['left_context_token']))
            self.right_context.append(" ".join(entry['right_context_token']))
            self.mention_span.append(entry['mention_span'])
            self.labels.append(label)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):

        left_context, right_context = self.left_context[idx], self.right_context[idx]
        mention_span, label = self.mention_span[idx], self.labels[idx]

        left_context_tokens = bert_tokenizer.tokenize(left_context)
        right_context_tokens = bert_tokenizer.tokenize(right_context)
        mention_tokens = bert_tokenizer.tokenize(mention_span)

        left_length = len(left_context_tokens)
        right_length = len(right_context_tokens)
        mention_length = len(mention_tokens)

        mention_tokens = mention_tokens[:args.bert_mention_max_len]
        mention_length = len(mention_tokens)

        total_length = left_length + right_length + 2 * mention_length

        if total_length + 3 >= args.bert_max_len:
            mid_length = int((args.bert_max_len - 3 - 2 * mention_length)/2)
            if left_length >= mid_length and right_length >= mid_length:
                cutoff_length = mid_length
                left_context_tokens = left_context_tokens[-cutoff_length:]
                right_context_tokens = right_context_tokens[:cutoff_length]
            if left_length >= mid_length and right_length < mid_length:
                cutoff_length = mid_length + mid_length - right_length
                left_context_tokens = left_context_tokens[-cutoff_length:]
            if left_length < mid_length and right_length >= mid_length:
                cutoff_length = mid_length + mid_length - left_length
                right_context_tokens = right_context_tokens[:cutoff_length]

        sent_tokens = left_context_tokens + mention_tokens + right_context_tokens

        tokens = ['[CLS]'] + sent_tokens + ['[SEP]'] + mention_tokens + ['[SEP]']

        input_ids = bert_tokenizer.convert_tokens_to_ids(tokens)

        segment_ids = [0] * (len(sent_tokens) + 2) + [1] * (len(mention_tokens) + 1)
        input_mask = [1] * (len(sent_tokens) + 2) + [1] * (len(mention_tokens) + 1)

        assert len(input_ids) == len(segment_ids) == len(input_mask)

        padding = [0] * (args.bert_max_len - len(input_ids))
        input_ids += padding
        segment_ids += padding
        input_mask += padding

        assert len(input_ids) == len(segment_ids) == len(input_mask) == args.bert_max_len

        input_ids = torch.LongTensor(input_ids)
        segment_ids = torch.LongTensor(segment_ids)
        input_mask = torch.LongTensor(input_mask)

        label = torch.FloatTensor(label)

        return input_ids, segment_ids, input_mask, label


class BertETDataset(data.Dataset):
    def __init__(self, path):
        entries = json.load(open(path, 'r'))
        self.sents, self.labels, self.starts, self.ends = [], [], [], []
        def trans(x): return x[x.rfind('/')+1:]
        def bug(x): return '/WEA/Club' if x == '/WEA/Clu' else x
        for entry in entries:

            ys = entry['labels']
            ys = [bug(i) for i in ys]
            label = [0] * len(labels)
            for y in ys:
                label[label_map[y]] = 1.0
            self.sents.append(entry['sent'])
            self.starts.append(int(entry['start']))
            self.ends.append(int(entry['end']))
            self.labels.append(label)

    def __len__(self):
        return len(self.sents)

    def __getitem__(self, idx):

        sent, label = self.sents[idx], self.labels[idx]
        start, end = self.starts[idx], self.ends[idx]

        left_context = sent[:start]
        right_context = sent[end:]
        mention_span = sent[start:end]

        left_context_tokens = bert_tokenizer.tokenize(left_context)
        right_context_tokens = bert_tokenizer.tokenize(right_context)
        mention_tokens = bert_tokenizer.tokenize(mention_span)

        left_length = len(left_context_tokens)
        right_length = len(right_context_tokens)
        mention_length = len(mention_tokens)

        # confirm mention tokens length
        mention_tokens = mention_tokens[:args.bert_mention_max_len - 2]
        mention_length = len(mention_tokens)

        total_length = left_length + right_length + mention_length

        if total_length + 2 >= args.bert_max_len:
            mid_length = int((args.bert_max_len - 2 - mention_length)/2)
            if left_length >= mid_length and right_length >= mid_length:
                cutoff_length = mid_length
                left_context_tokens = left_context_tokens[-cutoff_length:]
                right_context_tokens = right_context_tokens[:cutoff_length]
            if left_length >= mid_length and right_length < mid_length:
                cutoff_length = mid_length + mid_length - right_length
                left_context_tokens = left_context_tokens[-cutoff_length:]
            if left_length < mid_length and right_length >= mid_length:
                cutoff_length = mid_length + mid_length - left_length
                right_context_tokens = right_context_tokens[:cutoff_length]

        sent_tokens = left_context_tokens + mention_tokens + right_context_tokens

        bert_sent_tokens = ['[CLS]'] + sent_tokens + ['[SEP]']

        sent_input_ids = bert_tokenizer.convert_tokens_to_ids(bert_sent_tokens)

        sent_segment_ids = [0] * (len(bert_sent_tokens))
        sent_input_mask = [1] * (len(bert_sent_tokens))

        assert len(sent_input_ids) == len(sent_segment_ids) == len(sent_input_mask)

        sent_padding = [0] * (args.bert_max_len - len(sent_input_ids))
        sent_input_ids += sent_padding
        sent_segment_ids += sent_padding
        sent_input_mask += sent_padding

        sent_input_ids = torch.LongTensor(sent_input_ids)
        sent_segment_ids = torch.LongTensor(sent_segment_ids)
        sent_input_mask = torch.LongTensor(sent_input_mask)

        # mention
        bert_mention_tokens = ['[CLS]'] + mention_tokens + ['[SEP]']

        mention_input_ids = bert_tokenizer.convert_tokens_to_ids(bert_mention_tokens)

        mention_segment_ids = [0] * (len(bert_mention_tokens))
        mention_input_mask = [1] * (len(bert_mention_tokens))

        assert len(mention_input_ids) == len(mention_segment_ids) == len(mention_input_mask)

        mention_padding = [0] * (args.bert_mention_max_len - len(mention_input_ids))
        mention_input_ids += mention_padding
        mention_segment_ids += mention_padding
        mention_input_mask += mention_padding

        mention_input_ids = torch.LongTensor(mention_input_ids)
        mention_segment_ids = torch.LongTensor(mention_segment_ids)
        mention_input_mask = torch.LongTensor(mention_input_mask)

        # char
        char_input_ids = [char_dict[i] for i in mention_span]
        char_input_ids = char_input_ids[:args.bert_char_max_len]
        char_padding = [0] * (args.bert_char_max_len - len(char_input_ids))
        char_input_ids += char_padding
        char_input_ids = torch.LongTensor(char_input_ids)

        label = torch.FloatTensor(label)

        return sent_input_ids, sent_segment_ids, sent_input_mask, \
                mention_input_ids, mention_segment_ids, mention_input_mask, \
                char_input_ids, \
                label


def BertETInput(sent, start, end):
    left_context = sent[:start]
    right_context = sent[end:]
    mention_span = sent[start:end]

    left_context_tokens = bert_tokenizer.tokenize(left_context)
    right_context_tokens = bert_tokenizer.tokenize(right_context)
    mention_tokens = bert_tokenizer.tokenize(mention_span)

    left_length = len(left_context_tokens)
    right_length = len(right_context_tokens)
    mention_length = len(mention_tokens)

    # confirm mention tokens length
    mention_tokens = mention_tokens[:args.bert_mention_max_len - 2]
    mention_length = len(mention_tokens)

    total_length = left_length + right_length + mention_length

    if total_length + 2 >= args.bert_max_len:
        mid_length = int((args.bert_max_len - 2 - mention_length)/2)
        if left_length >= mid_length and right_length >= mid_length:
            cutoff_length = mid_length
            left_context_tokens = left_context_tokens[-cutoff_length:]
            right_context_tokens = right_context_tokens[:cutoff_length]
        if left_length >= mid_length and right_length < mid_length:
            cutoff_length = mid_length + mid_length - right_length
            left_context_tokens = left_context_tokens[-cutoff_length:]
        if left_length < mid_length and right_length >= mid_length:
            cutoff_length = mid_length + mid_length - left_length
            right_context_tokens = right_context_tokens[:cutoff_length]

    sent_tokens = left_context_tokens + mention_tokens + right_context_tokens

    bert_sent_tokens = ['[CLS]'] + sent_tokens + ['[SEP]']

    sent_input_ids = bert_tokenizer.convert_tokens_to_ids(bert_sent_tokens)

    sent_segment_ids = [0] * (len(bert_sent_tokens))
    sent_input_mask = [1] * (len(bert_sent_tokens))

    assert len(sent_input_ids) == len(sent_segment_ids) == len(sent_input_mask)

    sent_padding = [0] * (args.bert_max_len - len(sent_input_ids))
    sent_input_ids += sent_padding
    sent_segment_ids += sent_padding
    sent_input_mask += sent_padding

    sent_input_ids = torch.LongTensor(sent_input_ids)
    sent_segment_ids = torch.LongTensor(sent_segment_ids)
    sent_input_mask = torch.LongTensor(sent_input_mask)

    # mention
    bert_mention_tokens = ['[CLS]'] + mention_tokens + ['[SEP]']

    mention_input_ids = bert_tokenizer.convert_tokens_to_ids(bert_mention_tokens)

    mention_segment_ids = [0] * (len(bert_mention_tokens))
    mention_input_mask = [1] * (len(bert_mention_tokens))

    assert len(mention_input_ids) == len(mention_segment_ids) == len(mention_input_mask)

    mention_padding = [0] * (args.bert_mention_max_len - len(mention_input_ids))
    mention_input_ids += mention_padding
    mention_segment_ids += mention_padding
    mention_input_mask += mention_padding

    mention_input_ids = torch.LongTensor(mention_input_ids)
    mention_segment_ids = torch.LongTensor(mention_segment_ids)
    mention_input_mask = torch.LongTensor(mention_input_mask)

    # char
    char_input_ids = [char_dict[i] for i in mention_span]
    char_input_ids = char_input_ids[:args.bert_char_max_len]
    char_padding = [0] * (args.bert_char_max_len - len(char_input_ids))
    char_input_ids += char_padding
    char_input_ids = torch.LongTensor(char_input_ids)
    
    sent_input_ids = sent_input_ids.unsqueeze(0)
    sent_segment_ids = sent_segment_ids.unsqueeze(0)
    sent_input_mask = sent_input_mask.unsqueeze(0)
    mention_input_ids = mention_input_ids.unsqueeze(0)
    mention_segment_ids = mention_segment_ids.unsqueeze(0)
    mention_input_mask = mention_input_mask.unsqueeze(0)
    char_input_ids = char_input_ids.unsqueeze(0)


    return sent_input_ids, sent_segment_ids, sent_input_mask, \
            mention_input_ids, mention_segment_ids, mention_input_mask, \
            char_input_ids


def BertETBatchInput(sents, starts, ends):
    batch_sent_input_ids = []
    batch_sent_segment_ids = []
    batch_sent_input_mask = []
    batch_mention_input_ids = []
    batch_mention_segment_ids = []
    batch_mention_input_mask = []
    batch_char_input_ids = []

    batchify = lambda x: torch.stack(x, dim=0)

    for sent, start, end in zip(sents, starts, ends):
        left_context = sent[:start]
        right_context = sent[end:]
        mention_span = sent[start:end]

        left_context_tokens = bert_tokenizer.tokenize(left_context)
        right_context_tokens = bert_tokenizer.tokenize(right_context)
        mention_tokens = bert_tokenizer.tokenize(mention_span)

        left_length = len(left_context_tokens)
        right_length = len(right_context_tokens)
        mention_length = len(mention_tokens)

        # confirm mention tokens length
        mention_tokens = mention_tokens[:args.bert_mention_max_len - 2]
        mention_length = len(mention_tokens)

        total_length = left_length + right_length + mention_length

        if total_length + 2 >= args.bert_max_len:
            mid_length = int((args.bert_max_len - 2 - mention_length)/2)
            if left_length >= mid_length and right_length >= mid_length:
                cutoff_length = mid_length
                left_context_tokens = left_context_tokens[-cutoff_length:]
                right_context_tokens = right_context_tokens[:cutoff_length]
            if left_length >= mid_length and right_length < mid_length:
                cutoff_length = mid_length + mid_length - right_length
                left_context_tokens = left_context_tokens[-cutoff_length:]
            if left_length < mid_length and right_length >= mid_length:
                cutoff_length = mid_length + mid_length - left_length
                right_context_tokens = right_context_tokens[:cutoff_length]

        sent_tokens = left_context_tokens + mention_tokens + right_context_tokens

        bert_sent_tokens = ['[CLS]'] + sent_tokens + ['[SEP]']

        sent_input_ids = bert_tokenizer.convert_tokens_to_ids(bert_sent_tokens)

        sent_segment_ids = [0] * (len(bert_sent_tokens))
        sent_input_mask = [1] * (len(bert_sent_tokens))

        assert len(sent_input_ids) == len(sent_segment_ids) == len(sent_input_mask)

        sent_padding = [0] * (args.bert_max_len - len(sent_input_ids))
        sent_input_ids += sent_padding
        sent_segment_ids += sent_padding
        sent_input_mask += sent_padding

        sent_input_ids = torch.LongTensor(sent_input_ids)
        sent_segment_ids = torch.LongTensor(sent_segment_ids)
        sent_input_mask = torch.LongTensor(sent_input_mask)

        # mention
        bert_mention_tokens = ['[CLS]'] + mention_tokens + ['[SEP]']

        mention_input_ids = bert_tokenizer.convert_tokens_to_ids(bert_mention_tokens)

        mention_segment_ids = [0] * (len(bert_mention_tokens))
        mention_input_mask = [1] * (len(bert_mention_tokens))

        assert len(mention_input_ids) == len(mention_segment_ids) == len(mention_input_mask)

        mention_padding = [0] * (args.bert_mention_max_len - len(mention_input_ids))
        mention_input_ids += mention_padding
        mention_segment_ids += mention_padding
        mention_input_mask += mention_padding

        mention_input_ids = torch.LongTensor(mention_input_ids)
        mention_segment_ids = torch.LongTensor(mention_segment_ids)
        mention_input_mask = torch.LongTensor(mention_input_mask)

        # char
        char_input_ids = [char_dict[i] for i in mention_span]
        char_input_ids = char_input_ids[:args.bert_char_max_len]
        char_padding = [0] * (args.bert_char_max_len - len(char_input_ids))
        char_input_ids += char_padding
        char_input_ids = torch.LongTensor(char_input_ids)
        
        # sent_input_ids = sent_input_ids.unsqueeze(0)
        # sent_segment_ids = sent_segment_ids.unsqueeze(0)
        # sent_input_mask = sent_input_mask.unsqueeze(0)
        # mention_input_ids = mention_input_ids.unsqueeze(0)
        # mention_segment_ids = mention_segment_ids.unsqueeze(0)
        # mention_input_mask = mention_input_mask.unsqueeze(0)
        # char_input_ids = char_input_ids.unsqueeze(0)

        batch_sent_input_ids.append(sent_input_ids)
        batch_sent_segment_ids.append(sent_segment_ids)
        batch_sent_input_mask.append(sent_input_mask)
        batch_mention_input_ids.append(mention_input_ids)
        batch_mention_segment_ids.append(mention_segment_ids)
        batch_mention_input_mask.append(mention_input_mask)
        batch_char_input_ids.append(char_input_ids)

    batch_sent_input_ids = batchify(batch_sent_input_ids)
    batch_sent_segment_ids = batchify(batch_sent_segment_ids)
    batch_sent_input_mask = batchify(batch_sent_input_mask)
    batch_mention_input_ids = batchify(batch_mention_input_ids)
    batch_mention_segment_ids = batchify(batch_mention_segment_ids)
    batch_mention_input_mask = batchify(batch_mention_input_mask)
    batch_char_input_ids = batchify(batch_char_input_ids)


    return batch_sent_input_ids, batch_sent_segment_ids, batch_sent_input_mask, \
        batch_mention_input_ids, batch_mention_segment_ids, batch_mention_input_mask, \
            batch_char_input_ids

def BertMentionETBatchInput(sents, starts, ends):
    batch_sent_input_ids = []
    batch_sent_segment_ids = []
    batch_sent_input_mask = []
    batch_mention_start_idx = []
    batch_mention_end_idx = []
    batch_mention_mask = []
    batch_char_input_ids = []

    batchify = lambda x: torch.stack(x, dim=0)

    for sent, start, end in zip(sents, starts, ends):
        left_context = sent[:start]
        right_context = sent[end:]
        mention_span = sent[start:end]

        left_context_tokens = bert_tokenizer.tokenize(left_context)
        right_context_tokens = bert_tokenizer.tokenize(right_context)
        mention_tokens = bert_tokenizer.tokenize(mention_span)

        left_length = len(left_context_tokens)
        right_length = len(right_context_tokens)
        mention_length = len(mention_tokens)

        # confirm mention tokens length
        mention_tokens = mention_tokens[:args.bert_mention_max_len - 2]
        mention_length = len(mention_tokens)

        total_length = left_length + right_length + mention_length

        if total_length + 2 >= args.bert_max_len:
            mid_length = int((args.bert_max_len - 2 - mention_length)/2)
            if left_length >= mid_length and right_length >= mid_length:
                cutoff_length = mid_length
                left_context_tokens = left_context_tokens[-cutoff_length:]
                right_context_tokens = right_context_tokens[:cutoff_length]
            if left_length >= mid_length and right_length < mid_length:
                cutoff_length = mid_length + mid_length - right_length
                left_context_tokens = left_context_tokens[-cutoff_length:]
            if left_length < mid_length and right_length >= mid_length:
                cutoff_length = mid_length + mid_length - left_length
                right_context_tokens = right_context_tokens[:cutoff_length]

        sent_tokens = left_context_tokens + mention_tokens + right_context_tokens

        bert_sent_tokens = ['[CLS]'] + sent_tokens + ['[SEP]']

        sent_input_ids = bert_tokenizer.convert_tokens_to_ids(bert_sent_tokens)

        sent_segment_ids = [0] * (len(bert_sent_tokens))
        sent_input_mask = [1] * (len(bert_sent_tokens))

        assert len(sent_input_ids) == len(sent_segment_ids) == len(sent_input_mask)

        sent_padding = [0] * (args.bert_max_len - len(sent_input_ids))
        sent_input_ids += sent_padding
        sent_segment_ids += sent_padding
        sent_input_mask += sent_padding

        sent_input_ids = torch.LongTensor(sent_input_ids)
        sent_segment_ids = torch.LongTensor(sent_segment_ids)
        sent_input_mask = torch.LongTensor(sent_input_mask)

        # mention
        mention_start_idx = len(left_context_tokens) + 1
        mention_end_idx = len(left_context_tokens) + 1 + len(mention_tokens)
        mention_mask = [0] * (len(left_context_tokens) + 1) + [1] * len(mention_tokens)
        mention_mask += [1] * (len(sent_input_ids) - len(mention_mask))
        mention_ids = bert_tokenizer.convert_tokens_to_ids(mention_tokens) 
        mention_ids = torch.LongTensor(mention_ids)
       
        assert len(mention_mask) == len(sent_input_ids)
        assert (mention_ids == sent_input_ids[mention_start_idx:mention_end_idx]).all()

        mention_start_idx = torch.LongTensor([mention_start_idx])
        mention_end_idx = torch.LongTensor([mention_end_idx])
        mention_mask = torch.LongTensor(mention_mask)

        # char
        char_input_ids = [char_dict[i] for i in mention_span]
        char_input_ids = char_input_ids[:args.bert_char_max_len]
        char_padding = [0] * (args.bert_char_max_len - len(char_input_ids))
        char_input_ids += char_padding
        char_input_ids = torch.LongTensor(char_input_ids)
        
        # sent_input_ids = sent_input_ids.unsqueeze(0)
        # sent_segment_ids = sent_segment_ids.unsqueeze(0)
        # sent_input_mask = sent_input_mask.unsqueeze(0)
        # mention_input_ids = mention_input_ids.unsqueeze(0)
        # mention_segment_ids = mention_segment_ids.unsqueeze(0)
        # mention_input_mask = mention_input_mask.unsqueeze(0)
        # char_input_ids = char_input_ids.unsqueeze(0)

        batch_sent_input_ids.append(sent_input_ids)
        batch_sent_segment_ids.append(sent_segment_ids)
        batch_sent_input_mask.append(sent_input_mask)
        batch_mention_start_idx.append(mention_start_idx)
        batch_mention_end_idx.append(mention_end_idx)
        batch_mention_mask.append(mention_mask)
        batch_char_input_ids.append(char_input_ids)

    batch_sent_input_ids = batchify(batch_sent_input_ids)
    batch_sent_segment_ids = batchify(batch_sent_segment_ids)
    batch_sent_input_mask = batchify(batch_sent_input_mask)
    batch_mention_start_idx = batchify(batch_mention_start_idx)
    batch_mention_end_idx = batchify(batch_mention_end_idx)
    batch_mention_mask = batchify(batch_mention_mask)
    batch_char_input_ids = batchify(batch_char_input_ids)


    return batch_sent_input_ids, batch_sent_segment_ids, batch_sent_input_mask, \
        batch_mention_start_idx, batch_mention_end_idx, batch_mention_mask, \
            batch_char_input_ids


# mention max length need be considered
class ErnieETDataset(data.Dataset):
    def __init__(self, path):
        entries = json.load(open(path, 'r'))
        self.sents, self.labels, self.ents, self.starts, self.ends = [], [], [], [], []
        self.entity2id = {}
        with open("kb/entity2id.txt") as fin:
            fin.readline()
            for line in fin:
                qid, eid = line.strip().split('\t')
                self.entity2id[qid] = int(eid)
        def trans(x): return x[x.rfind('/')+1:]
        for entry in entries:
            ys = entry['labels']
            #ys = [trans(i) for i in ys]
            label = [0] * len(labels)
            for y in ys:
                label[label_map[y]] = 1.0
            self.ents.append(entry['ents'])
            self.sents.append(entry['sent'])
            self.labels.append(label)
            self.starts.append(int(entry['start']))
            self.ends.append(int(entry['end']))

    def __len__(self):
        return len(self.sents)

    # TODO: a wise cutoff way
    def __getitem__(self, idx):
        sent, ents, label = self.sents[idx], self.ents[idx], self.labels[idx]
        start, end = self.starts[idx], self.ends[idx]

        left_context = sent[:start]
        right_context = sent[end:]
        mention_span = sent[start:end]

        left_context_tokens, _ = ernie_tokenizer_label.tokenize(left_context, [])
        right_context_tokens, _ = ernie_tokenizer_label.tokenize(right_context, [])
        mention_tokens, _ = ernie_tokenizer_label.tokenize(mention_span, [])
        
        left_length = len(left_context_tokens)
        right_length = len(right_context_tokens)
        mention_length = len(mention_tokens)
        
        total_length = left_length + mention_length + right_length

        left_cutoff = None
        right_cutoff = None

        # 2 dots [CLS] [SEP]
        if total_length + 2 + 2 > args.bert_max_len:
            mid_length = int((args.bert_max_len - 2 - 2 - mention_length)/2)
            if left_length >= mid_length and right_length >= mid_length:
                cutoff_length = mid_length
                left_cutoff = left_length - cutoff_length
                right_cutoff = left_length + mention_length + 2 + cutoff_length
            if left_length >= mid_length and right_length < mid_length:
                cutoff_length = mid_length + mid_length - right_length
                left_cutoff = left_length - cutoff_length
            if left_length < mid_length and right_length >= mid_length:
                cutoff_length = mid_length + mid_length - left_length
                right_cutoff = left_length + mention_length + 2 + cutoff_length

        h = ["SPAN", start, end]
        text = left_context + ". " + mention_span + " ." + right_context
        begin, end = h[1:3]
        # .\space
        h[1] += 2
        h[2] += 2
        sent, mention = ernie_tokenizer_label.tokenize(text, [h])

        filter_ents = [x for x in ents if x[-1] > args.bert_entity_threshold]
        for x in filter_ents:
            if x[1] > end:
                # .\space \space.
                x[1] += 4
                x[2] += 4
            elif x[1] >= begin:
                # .\space
                x[1] += 2
                x[2] += 2
        _, entities = ernie_tokenizer.tokenize(text, filter_ents)

        sent = sent[left_cutoff:right_cutoff]
        mention = mention[left_cutoff:right_cutoff]
        entities = entities[left_cutoff:right_cutoff]

        tokens = ["[CLS]"] + sent + ["[SEP]"]
        ents = ["UNK"] + mention + ["UNK"]
        real_ents = ["UNK"] + entities + ["UNK"]

        segment_ids = [0] * len(tokens)
        input_ids = ernie_tokenizer.convert_tokens_to_ids(tokens)

        span_mask = []
        for ent in ents:
            if ent != "UNK":
                span_mask.append(1)
            else:
                span_mask.append(0)


        input_ent = []
        ent_mask = []
        for ent in real_ents:
            if ent != "UNK" and ent in self.entity2id:
                input_ent.append(self.entity2id[ent])
                ent_mask.append(1)
            else:
                input_ent.append(-1)
                ent_mask.append(0)
        ent_mask[0] = 1


        if sum(span_mask) == 0:
            # continue
            # lazy process
            logger.info('no entity when cut')
            idx = idx - 1 if idx > 0 else idx + 1
            return self.__getitem__(idx)

        input_mask = [1] * len(input_ids)

        padding = [0] * (args.bert_max_len - len(input_ids))
        padding_ = [-1] * (args.bert_max_len - len(input_ids))
        input_ids += padding
        input_mask += padding
        segment_ids += padding
        ent_mask += padding
        input_ent += padding_

        assert len(input_ids) == len(input_mask) == len(segment_ids) == args.bert_max_len
        assert len(input_ent) == len(ent_mask) == args.bert_max_len

        sent_input_ids = torch.LongTensor(input_ids)
        sent_segment_ids = torch.LongTensor(segment_ids)
        sent_input_mask = torch.LongTensor(input_mask)
        sent_input_ent = torch.LongTensor(input_ent)
        sent_ent_mask = torch.LongTensor(ent_mask)

        # mention
        mention_text = mention_span
        mention_start, mention_end = 0, len(mention_text)
        mention_h = ["SPAN", mention_start, mention_end]
        mention_sent, _ = ernie_tokenizer_label.tokenize(mention_text, [mention_h])

        _, mention_ents = ernie_tokenizer.tokenize(mention_text, [])

        mention_sent = mention_sent[:args.bert_mention_max_len-2]
        mention_ents = mention_ents[:args.bert_mention_max_len-2]

        mention_real_ents = ["UNK"] + mention_ents + ["UNK"]
        mention_sent_tokens = ['[CLS]'] + mention_sent + ['[SEP]']

        mention_input_ids = ernie_tokenizer.convert_tokens_to_ids(mention_sent_tokens)

        mention_segment_ids = [0] * (len(mention_sent_tokens))
        mention_input_mask = [1] * (len(mention_sent_tokens))

        mention_input_ent = []
        mention_ent_mask = []

        for ent in mention_real_ents:
            if ent != "UNK" and ent in self.entity2id:
                mention_input_ent.append(self.entity2id[ent])
                mention_ent_mask.append(1)
            else:
                mention_input_ent.append(-1)
                mention_ent_mask.append(0)
        # mention_ent_mask[0] = 1

        assert len(mention_input_ids) == len(mention_segment_ids) == len(mention_input_mask)

        mention_padding = [0] * (args.bert_mention_max_len - len(mention_input_ids))
        mention_padding_ = [-1] * (args.bert_mention_max_len - len(mention_input_ids))
        mention_input_ids += mention_padding
        mention_segment_ids += mention_padding
        mention_input_mask += mention_padding
        mention_input_ent += mention_padding_
        mention_ent_mask += mention_padding

        assert len(mention_input_ids) == len(mention_segment_ids) == len(mention_input_mask) == args.bert_mention_max_len
        assert len(mention_input_ent) == len(mention_ent_mask) == args.bert_mention_max_len

        mention_input_ids = torch.LongTensor(mention_input_ids)
        mention_segment_ids = torch.LongTensor(mention_segment_ids)
        mention_input_mask = torch.LongTensor(mention_input_mask)
        mention_input_ent = torch.LongTensor(mention_input_ent)
        mention_ent_mask = torch.LongTensor(mention_ent_mask)

        # char
        char_input_ids = [char_dict[i] for i in mention_span]
        char_input_ids = char_input_ids[:args.bert_char_max_len]
        char_padding = [0] * (args.bert_char_max_len - len(char_input_ids))
        char_input_ids += char_padding
        char_input_ids = torch.LongTensor(char_input_ids)

        label = torch.FloatTensor(label)

        return sent_input_ids, sent_segment_ids, sent_input_mask, sent_input_ent, sent_ent_mask, \
                mention_input_ids, mention_segment_ids, mention_input_mask, mention_input_ent, mention_ent_mask, \
                char_input_ids, label


class BertDataset(data.Dataset):
    def __init__(self, path):
        entries = json.load(open(path, 'r'))
        self.sents, self.labels, self.starts, self.ends = [], [], [], []
        def trans(x): return x[x.rfind('/')+1:]
        for entry in entries:
            ys = entry['labels']
            #ys = [trans(i) for i in ys]
            label = [0] * len(labels)
            for y in ys:
                label[label_map[y]] = 1.0
            self.sents.append(entry['sent'])
            self.starts.append(int(entry['start']))
            self.ends.append(int(entry['end']))
            self.labels.append(label)

    def __len__(self):
        return len(self.sents)

    def __getitem__(self, idx):

        sent, label = self.sents[idx], self.labels[idx]
        start, end = self.starts[idx], self.ends[idx]

        left_context = sent[:start]
        right_context = sent[end:]
        mention_span = sent[start:end]

        left_context_tokens = bert_tokenizer.tokenize(left_context)
        right_context_tokens = bert_tokenizer.tokenize(right_context)
        mention_tokens = bert_tokenizer.tokenize(mention_span)

        left_length = len(left_context_tokens)
        right_length = len(right_context_tokens)
        mention_length = len(mention_tokens)

        # confirm mention tokens length
        mention_tokens = mention_tokens[:args.bert_mention_max_len]
        mention_length = len(mention_tokens)

        total_length = left_length + right_length + 2 * mention_length

        if total_length + 3 >= args.bert_max_len:
            mid_length = int((args.bert_max_len - 3 - 2 * mention_length)/2)
            if left_length >= mid_length and right_length >= mid_length:
                cutoff_length = mid_length
                left_context_tokens = left_context_tokens[-cutoff_length:]
                right_context_tokens = right_context_tokens[:cutoff_length]
            if left_length >= mid_length and right_length < mid_length:
                cutoff_length = mid_length + mid_length - right_length
                left_context_tokens = left_context_tokens[-cutoff_length:]
            if left_length < mid_length and right_length >= mid_length:
                cutoff_length = mid_length + mid_length - left_length
                right_context_tokens = right_context_tokens[:cutoff_length]

        sent_tokens = left_context_tokens + mention_tokens + right_context_tokens

        tokens = ['[CLS]'] + sent_tokens + ['[SEP]'] + mention_tokens + ['[SEP]']

        input_ids = bert_tokenizer.convert_tokens_to_ids(tokens)

        segment_ids = [0] * (len(sent_tokens) + 2) + [1] * (len(mention_tokens) + 1)
        input_mask = [1] * (len(sent_tokens) + 2) + [1] * (len(mention_tokens) + 1)

        assert len(input_ids) == len(segment_ids) == len(input_mask)

        padding = [0] * (args.bert_max_len - len(input_ids))
        input_ids += padding
        segment_ids += padding
        input_mask += padding

        assert len(input_ids) == len(segment_ids) == len(input_mask) == args.bert_max_len

        input_ids = torch.LongTensor(input_ids)
        segment_ids = torch.LongTensor(segment_ids)
        input_mask = torch.LongTensor(input_mask)

        label = torch.FloatTensor(label)

        return input_ids, segment_ids, input_mask, label


class ErnieDataset(data.Dataset):
    def __init__(self, path):
        entries = json.load(open(path, 'r'))
        self.sents, self.labels, self.ents, self.starts, self.ends = [], [], [], [], []
        self.entity2id = {}
        with open("kb/entity2id.txt") as fin:
            fin.readline()
            for line in fin:
                qid, eid = line.strip().split('\t')
                self.entity2id[qid] = int(eid)
        #def trans(x): return x[x.rfind('/')+1:]
        for entry in entries:
            ys = entry['labels']
            ys = [trans(i) for i in ys]
            label = [0] * len(labels)
            for y in ys:
                label[label_map[y]] = 1.0
            self.ents.append(entry['ents'])
            self.sents.append(entry['sent'])
            self.labels.append(label)
            self.starts.append(int(entry['start']))
            self.ends.append(int(entry['end']))

    def __len__(self):
        return len(self.sents)

    # TODO: a wise cutoff way
    def __getitem__(self, idx):
        sent, ents, label = self.sents[idx], self.ents[idx], self.labels[idx]
        start, end = self.starts[idx], self.ends[idx]

        left_context = sent[:start]
        right_context = sent[end:]
        mention_span = sent[start:end]

        left_context_tokens, _ = ernie_tokenizer_label.tokenize(left_context, [])
        right_context_tokens, _ = ernie_tokenizer_label.tokenize(right_context, [])
        mention_tokens, _ = ernie_tokenizer_label.tokenize(mention_span, [])
        
        left_length = len(left_context_tokens)
        right_length = len(right_context_tokens)
        mention_length = len(mention_tokens)
        
        total_length = left_length + mention_length + right_length

        left_cutoff = None
        right_cutoff = None

        # 2 dots [CLS] [SEP]
        if total_length + 2 + 2 > args.bert_max_len:
            mid_length = int((args.bert_max_len - 2 - 2 - mention_length)/2)
            if left_length >= mid_length and right_length >= mid_length:
                cutoff_length = mid_length
                left_cutoff = left_length - cutoff_length
                right_cutoff = left_length + mention_length + 2 + cutoff_length
            if left_length >= mid_length and right_length < mid_length:
                cutoff_length = mid_length + mid_length - right_length
                left_cutoff = left_length - cutoff_length
            if left_length < mid_length and right_length >= mid_length:
                cutoff_length = mid_length + mid_length - left_length
                right_cutoff = left_length + mention_length + 2 + cutoff_length

        h = ["SPAN", start, end]
        text = left_context + ". " + mention_span + " ." + right_context
        begin, end = h[1:3]
        # .\space
        h[1] += 2
        h[2] += 2
        sent, mention = ernie_tokenizer_label.tokenize(text, [h])

        filter_ents = [x for x in ents if x[-1] > args.bert_entity_threshold]
        for x in filter_ents:
            if x[1] > end:
                # .\space \space.
                x[1] += 4
                x[2] += 4
            elif x[1] >= begin:
                # .\space
                x[1] += 2
                x[2] += 2
        _, entities = ernie_tokenizer.tokenize(text, filter_ents)

        sent = sent[left_cutoff:right_cutoff]
        mention = mention[left_cutoff:right_cutoff]
        entities = entities[left_cutoff:right_cutoff]

        tokens = ["[CLS]"] + sent + ["[SEP]"]
        ents = ["UNK"] + mention + ["UNK"]
        real_ents = ["UNK"] + entities + ["UNK"]

        segment_ids = [0] * len(tokens)
        input_ids = ernie_tokenizer.convert_tokens_to_ids(tokens)

        span_mask = []
        for ent in ents:
            if ent != "UNK":
                span_mask.append(1)
            else:
                span_mask.append(0)


        input_ent = []
        ent_mask = []
        for ent in real_ents:
            if ent != "UNK" and ent in self.entity2id:
                input_ent.append(self.entity2id[ent])
                ent_mask.append(1)
            else:
                input_ent.append(-1)
                ent_mask.append(0)
        ent_mask[0] = 1


        if sum(span_mask) == 0:
            # continue
            # lazy process
            logger.info('no entity when cut')
            idx = idx - 1 if idx > 0 else idx + 1
            return self.__getitem__(idx)

        input_mask = [1] * len(input_ids)

        padding = [0] * (args.bert_max_len - len(input_ids))
        padding_ = [-1] * (args.bert_max_len - len(input_ids))
        input_ids += padding
        input_mask += padding
        segment_ids += padding
        ent_mask += padding
        input_ent += padding_

        assert len(input_ids) == len(input_mask) == len(segment_ids) == args.bert_max_len
        assert len(input_ent) == len(ent_mask) == args.bert_max_len

        input_ids = torch.LongTensor(input_ids)
        segment_ids = torch.LongTensor(segment_ids)
        input_mask = torch.LongTensor(input_mask)
        input_ent = torch.LongTensor(input_ent)
        ent_mask = torch.LongTensor(ent_mask)

        label = torch.FloatTensor(label)

        return input_ids, segment_ids, input_mask, input_ent, ent_mask, label


if __name__ == "__main__":
    pass
