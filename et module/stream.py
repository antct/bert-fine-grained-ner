from gevent.pywsgi import WSGIServer
from flask import Flask, request, jsonify, Blueprint
from model import BertETNet
from config import args
from util import BertETInput, BertETBatchInput, labels

import torch
import os
import torch.nn as nn
import numpy as np
import random

from service_streamer import ManagedModel, Streamer
from gevent import monkey
monkey.patch_all()

app = Flask(__name__)

args.batch_size = 1
args.enhance_mention = True
args.interaction = True
args.model_path = 'best.pt'
args.bert_freeze = True

seed = args.seed
os.environ['PYTHONHASHSEED'] = str(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
np.random.seed(seed)  # Numpy module.
random.seed(seed)  # Python random module.
torch.manual_seed(seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True


def topk(logits, k=1):
    sigmoid_fn = nn.Sigmoid()
    logits = sigmoid_fn(logits).data.cpu().clone()
    top_idx = []
    for logit in logits:
        t = []
        logit = logit.numpy()
        logit_map = [[i, float(score)] for i, score in enumerate(logit)]
        sorted_logit_map = sorted(logit_map, key=lambda x: x[1], reverse=True)
        sorted_logit_idx = [i for i in sorted_logit_map]
        t.extend(sorted_logit_idx[:k])
        top_idx.append(t)
    return top_idx


class Model(ManagedModel):
    def init_model(self):
        print(vars(args))
        self.model = BertETNet()
        self.model.eval()
        self.model = self.model.cuda()
        model_name = '{}/{}'.format(args.model_dir, args.model_path)
        self.model.load_state_dict(torch.load(model_name, map_location='cuda'))

    def predict(self, batch):
        sent = [i[0] for i in batch]
        start = [i[1] for i in batch]
        end = [i[2] for i in batch]
        with torch.no_grad():
            sent_input_ids, sent_segment_ids, sent_input_mask, \
                mention_input_ids, mention_segment_ids, mention_input_mask, \
                char_input_ids = BertETBatchInput(sent, start, end)
            sent_input_ids = sent_input_ids.cuda()
            sent_segment_ids = sent_segment_ids.cuda()
            sent_input_mask = sent_input_mask.cuda()
            mention_input_ids = mention_input_ids.cuda()
            mention_segment_ids = mention_segment_ids.cuda()
            mention_input_mask = mention_input_mask.cuda()
            char_input_ids = char_input_ids.cuda()
            logits = self.model(
                context_input_ids=sent_input_ids,
                context_token_type_ids=sent_segment_ids,
                context_attention_mask=sent_input_mask,
                mention_input_ids=mention_input_ids,
                mention_token_type_ids=mention_segment_ids,
                mention_attention_mask=mention_input_mask,
                char_input_ids=char_input_ids
            )

            topk_idxs = topk(logits, k=None)
            topk_labels = [[[labels[idx[0]], idx[1]] for idx in idxs] for idxs in topk_idxs]
            return topk_labels


@app.route('/batch', methods=['POST'])
def batch():
    sent = request.form.getlist('sent')
    start = request.form.getlist('start')
    end = request.form.getlist('end')
    start = [int(i) for i in start]
    end = [int(i) for i in end]

    batch = []
    batch = [[sent[i], start[i], end[i]] for i in range(len(sent))]

    topk_labels = streamer.predict(batch)
    resp = topk_labels
    return jsonify(resp)


if __name__ == "__main__":
    streamer = Streamer(Model, batch_size=16, max_latency=0.1, worker_num=8, cuda_devices=(0, 1))
    WSGIServer(("0.0.0.0", 3101), app).serve_forever()
