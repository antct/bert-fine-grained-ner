import torch
import torch.nn as nn

from util import logger, CoNLLInput
from config import args
from metric import iob2location
from model import BertNERNet
from flask import Flask, request, jsonify

app = Flask(__name__)


@app.route('/', methods=['GET'])
def index():
    sent = request.args.get('sent')
    input_ids, segment_ids, input_mask, input_char_ids, heads = CoNLLInput(
        sent)
    model.eval()
    all_ys_, all_heads = [], []
    with torch.no_grad():
        _, ys_ = model(
            input_ids=input_ids,
            token_type_ids=segment_ids,
            attention_mask=input_mask,
            input_char_ids=input_char_ids,
            labels=None
        )

        all_ys_.extend(ys_.cpu().numpy().tolist())
        all_heads.extend(heads.cpu().numpy().tolist())

    preds = []

    for heads, ys_ in zip(all_heads, all_ys_):
        preds.extend([y_ for head, y_ in zip(heads, ys_) if head == 1])

    location_idxs = iob2location(preds)

    tokens = sent.split()
    idxs = [0]
    for token in tokens:
        t = sent[idxs[-1]:].find(token)
        idxs.append(t+idxs[-1])
    idxs = idxs[1:]

    locations = []
    for location_idx in location_idxs:
        left, right = location_idx
        start = idxs[left]
        end = idxs[right-1] + len(tokens[right-1])
        locations.append([start, end])

    resp = {
        'locations': locations
    }

    return jsonify(resp)


if __name__ == "__main__":
    # cpu version model
    device = 'cpu'

    args.batch_size = 1
    # need fix
    args.crf_use_cuda = False

    model = BertNERNet()

    model_name = '{}/{}'.format(args.model_dir, args.model_path)
    model.load_state_dict(torch.load(model_name, map_location='cpu'))

    app.run()
