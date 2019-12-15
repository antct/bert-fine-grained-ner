from flask import Flask, request, jsonify, Blueprint
from model import BertETNet
from config import args
from util import BertETInput, BertETBatchInput, labels

import torch
import os
import torch.nn as nn
import numpy as np
import random

worker_id = int(os.environ.get('APP_WORKER_ID', 1))
devices = os.environ.get('CUDA_VISIBLE_DEVICES', '')

if not devices:
    print('current environment did not get CUDA_VISIBLE_DEVICES env ,so use the default')

rand_max = 9527
gpu_index = (worker_id + rand_max) % torch.cuda.device_count()

print('current worker id  {} set the gpu id :{}'.format(worker_id, gpu_index))
torch.cuda.set_device(int(gpu_index))


app = Flask(__name__)

args.batch_size = 1
args.enhance_mention = True
args.interaction = True
args.model_path = 'best.pt'

model = BertETNet()

model = model.cuda(int(gpu_index))

model_name = '{}/{}'.format(args.model_dir, args.model_path)
model.load_state_dict(torch.load(model_name, map_location='cuda'))

model.eval()

print(vars(args))


def topk(logits, k=1):
    sigmoid_fn = nn.Sigmoid()
    logits = sigmoid_fn(logits).data.cpu().clone()
    top_idx = []
    for logit in logits:
        t = []
        logit = logit.numpy()
        logit_map = [[i, score] for i, score in enumerate(logit)]
        sorted_logit_map = sorted(logit_map, key=lambda x: x[1], reverse=True)
        sorted_logit_idx = [i[0] for i in sorted_logit_map]
        t.extend(sorted_logit_idx[:k])
        top_idx.append(t)
    return top_idx


@app.route('/', methods=['GET'])
def index():

    sent = request.args.get('sent')
    start = int(request.args.get('start'))
    end = int(request.args.get('end'))

    with torch.no_grad():
        sent_input_ids, sent_segment_ids, sent_input_mask, \
            mention_input_ids, mention_segment_ids, mention_input_mask, \
            char_input_ids = BertETInput(sent, start, end)
        sent_input_ids = sent_input_ids.cuda(int(gpu_index))
        sent_segment_ids = sent_segment_ids.cuda(int(gpu_index))
        sent_input_mask = sent_input_mask.cuda(int(gpu_index))
        mention_input_ids = mention_input_ids.cuda(int(gpu_index))
        mention_segment_ids = mention_segment_ids.cuda(int(gpu_index))
        mention_input_mask = mention_input_mask.cuda(int(gpu_index))
        char_input_ids = char_input_ids.cuda(int(gpu_index))
        logits = model(
            context_input_ids=sent_input_ids,
            context_token_type_ids=sent_segment_ids,
            context_attention_mask=sent_input_mask,
            mention_input_ids=mention_input_ids,
            mention_token_type_ids=mention_segment_ids,
            mention_attention_mask=mention_input_mask,
            char_input_ids=char_input_ids
        )

        topk_idxs = topk(logits, k=None)
        topk_labels = [[labels[idx] for idx in idxs] for idxs in topk_idxs]

        topk_labels = topk_labels[0]

        resp = topk_labels

        return jsonify(resp)


@app.route('/batch', methods=['POST'])
def batch():

    sent = request.form.getlist('sent')
    start = request.form.getlist('start')
    end = request.form.getlist('end')

    start = [int(i) for i in start]
    end = [int(i) for i in end]

    with torch.no_grad():
        sent_input_ids, sent_segment_ids, sent_input_mask, \
            mention_input_ids, mention_segment_ids, mention_input_mask, \
            char_input_ids = BertETBatchInput(sent, start, end)
        sent_input_ids = sent_input_ids.cuda(int(gpu_index))
        sent_segment_ids = sent_segment_ids.cuda(int(gpu_index))
        sent_input_mask = sent_input_mask.cuda(int(gpu_index))
        mention_input_ids = mention_input_ids.cuda(int(gpu_index))
        mention_segment_ids = mention_segment_ids.cuda(int(gpu_index))
        mention_input_mask = mention_input_mask.cuda(int(gpu_index))
        char_input_ids = char_input_ids.cuda(int(gpu_index))
        logits = model(
            context_input_ids=sent_input_ids,
            context_token_type_ids=sent_segment_ids,
            context_attention_mask=sent_input_mask,
            mention_input_ids=mention_input_ids,
            mention_token_type_ids=mention_segment_ids,
            mention_attention_mask=mention_input_mask,
            char_input_ids=char_input_ids
        )

        topk_idxs = topk(logits, k=None)
        topk_labels = [[labels[idx] for idx in idxs] for idxs in topk_idxs]

        topk_labels = topk_labels

        resp = topk_labels

        return jsonify(resp)


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=3101, threaded=True)
