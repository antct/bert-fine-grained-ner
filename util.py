import numpy as np
import torch

from torch.utils import data
from bert.tokenization import BertTokenizer
from config import args
from log import Logger
from collections import defaultdict

logger = Logger(filename='run.log', level='debug').getLogger()

logger.info('load char vocab')
CHAR_DICT = defaultdict(int)
CHAR_VOCAB = [u"<unk>"]
with open('data/char_vocab.english.txt') as f:
    CHAR_VOCAB.extend(c.strip() for c in f.readlines())
    CHAR_DICT.update({c: i for i, c in enumerate(CHAR_VOCAB)})

logger.info('load vocab')
VOCAB = ('<PAD>', 'O', 'I', 'B')
PAD_IDX = 0
O_IDX = 1
I_IDX = 2
B_IDX = 3
TAG2IDX = {tag: idx for idx, tag in enumerate(VOCAB)}
IDX2TAG = {idx: tag for idx, tag in enumerate(VOCAB)}

logger.info('load bert tokenizer')
tokenizer = BertTokenizer.from_pretrained('bert_base/', do_lower_case=args.bert_low_case)


class CoNLLDataset(data.Dataset):
    def __init__(self, path):
        entries = open(path, 'r').read().strip().split("\n\n")
        self.words, self.tags = [], []
        for entry in entries:
            words = [line.split()[0] for line in entry.splitlines()]
            def trans(x): return x[0:x.find('-')] if x.find('-') != -1 else x
            tags = ([trans(line.split()[-1]) for line in entry.splitlines()])
            self.words.append(words)
            self.tags.append(tags)

    def __len__(self):
        return len(self.words)

    def __getitem__(self, idx):
        # words and tags are both list
        words, tags = self.words[idx], self.tags[idx]
        xs, ys, chars, heads = [], [], [], []
        for word, tag in zip(words, tags):
            x = tokenizer.tokenize(word)
            if not len(x):
                continue
            head = [1] + [0] * (len(x) - 1)
            y = [tag] + ["<PAD>"] * (len(x) - 1)
            char = [[CHAR_DICT[i] for i in token] for token in x]
            assert len(x) == len(y) == len(head) == len(char)
            if len(xs) + len(x) >= args.bert_max_len - 2:
                break
            xs.extend(x)
            heads.extend(head)
            ys.extend(y)
            chars.extend(char)

        input_tokens = ['[CLS]'] + xs + ['[SEP]']
        input_char_ids = [[0]] + chars + [[0]]
        input_ids = tokenizer.convert_tokens_to_ids(input_tokens)
        heads = [0] + heads + [0]
        labels = ['<PAD>'] + ys + ['<PAD>']
        labels = [TAG2IDX[label] for label in labels]

        segment_ids = [0] * len(input_ids)
        input_mask = [1] * len(input_ids)

        assert len(input_ids) == len(segment_ids) == \
            len(input_mask) == len(labels) == len(heads)

        padding = [0] * (args.bert_max_len - len(input_ids))

        input_ids += padding
        segment_ids += padding
        input_mask += padding
        labels += padding
        heads += padding

        input_char_ids = [i[:args.char_max_len] for i in input_char_ids]

        input_char_ids = [i + [0] * (args.char_max_len-len(i)) for i in input_char_ids] + \
            [[0] * args.char_max_len] * (args.bert_max_len - len(input_char_ids))

        input_ids = torch.LongTensor(input_ids)
        segment_ids = torch.LongTensor(segment_ids)
        input_mask = torch.LongTensor(input_mask)
        input_char_ids = torch.LongTensor(input_char_ids)
        heads = torch.LongTensor(heads)
        labels = torch.LongTensor(labels)

        return input_ids, segment_ids, input_mask, input_char_ids, heads, labels

def CoNLLInput(tokens):
    words = tokens
    xs, chars, heads = [], [], []
    for word in words:
        x = tokenizer.tokenize(word)
        if not len(x):
            continue
        head = [1] + [0] * (len(x) - 1)
        char = [[CHAR_DICT[i] for i in token] for token in x]
        assert len(x) == len(head) == len(char)
        if len(xs) + len(x) >= args.bert_max_len - 2:
            break
        xs.extend(x)
        heads.extend(head)
        chars.extend(char)

    input_tokens = ['[CLS]'] + xs + ['[SEP]']
    input_char_ids = [[0]] + chars + [[0]]
    input_ids = tokenizer.convert_tokens_to_ids(input_tokens)
    heads = [0] + heads + [0]

    segment_ids = [0] * len(input_ids)
    input_mask = [1] * len(input_ids)

    assert len(input_ids) == len(segment_ids) == len(input_mask) == len(heads)

    padding = [0] * (args.bert_max_len - len(input_ids))

    input_ids += padding
    segment_ids += padding
    input_mask += padding
    heads += padding

    input_char_ids = [i[:args.char_max_len] for i in input_char_ids]

    input_char_ids = [i + [0] * (args.char_max_len-len(i)) for i in input_char_ids] + \
        [[0] * args.char_max_len] * (args.bert_max_len - len(input_char_ids))

    input_ids = torch.LongTensor(input_ids)
    segment_ids = torch.LongTensor(segment_ids)
    input_mask = torch.LongTensor(input_mask)
    input_char_ids = torch.LongTensor(input_char_ids)
    heads = torch.LongTensor(heads)

    input_ids = input_ids.unsqueeze(0)
    segment_ids = segment_ids.unsqueeze(0)
    input_mask = input_mask.unsqueeze(0)
    input_char_ids = input_char_ids.unsqueeze(0)
    heads = heads.unsqueeze(0)

    return input_ids, segment_ids, input_mask, input_char_ids, heads


def CoNLLBatchInput(tokens_list):
    batch_input_ids = []
    batch_segment_ids = []
    batch_input_mask = []
    batch_input_char_ids = []
    batch_heads = []

    batchify = lambda x: torch.stack(x, dim=0)

    for tokens in tokens_list:
        words = tokens
        xs, chars, heads = [], [], []
        for word in words:
            x = tokenizer.tokenize(word)
            if not len(x):
                continue
            head = [1] + [0] * (len(x) - 1)
            char = [[CHAR_DICT[i] for i in token] for token in x]
            assert len(x) == len(head) == len(char)
            if len(xs) + len(x) >= args.bert_max_len - 2:
                break
            xs.extend(x)
            heads.extend(head)
            chars.extend(char)

        input_tokens = ['[CLS]'] + xs + ['[SEP]']
        input_char_ids = [[0]] + chars + [[0]]
        input_ids = tokenizer.convert_tokens_to_ids(input_tokens)
        heads = [0] + heads + [0]

        segment_ids = [0] * len(input_ids)
        input_mask = [1] * len(input_ids)

        assert len(input_ids) == len(segment_ids) == len(input_mask) == len(heads)

        padding = [0] * (args.bert_max_len - len(input_ids))

        input_ids += padding
        segment_ids += padding
        input_mask += padding
        heads += padding

        input_char_ids = [i[:args.char_max_len] for i in input_char_ids]

        input_char_ids = [i + [0] * (args.char_max_len-len(i)) for i in input_char_ids] + \
            [[0] * args.char_max_len] * (args.bert_max_len - len(input_char_ids))

        input_ids = torch.LongTensor(input_ids)
        segment_ids = torch.LongTensor(segment_ids)
        input_mask = torch.LongTensor(input_mask)
        input_char_ids = torch.LongTensor(input_char_ids)
        heads = torch.LongTensor(heads)

        # input_ids = input_ids.unsqueeze(0)
        # segment_ids = segment_ids.unsqueeze(0)
        # input_mask = input_mask.unsqueeze(0)
        # input_char_ids = input_char_ids.unsqueeze(0)
        # heads = heads.unsqueeze(0)

        batch_input_ids.append(input_ids)
        batch_segment_ids.append(segment_ids)
        batch_input_mask.append(input_mask)
        batch_input_char_ids.append(input_char_ids)
        batch_heads.append(heads)

    batch_input_ids = batchify(batch_input_ids)
    batch_segment_ids = batchify(batch_segment_ids)
    batch_input_mask = batchify(batch_input_mask)
    batch_input_char_ids = batchify(batch_input_char_ids)
    batch_heads = batchify(batch_heads)

    return batch_input_ids, batch_segment_ids, batch_input_mask, \
        batch_input_char_ids, batch_heads