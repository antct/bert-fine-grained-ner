from gevent import monkey
from service_streamer import ManagedModel, Streamer
from gevent.pywsgi import WSGIServer
from flask import Flask, request, jsonify
from model import BertNERNet
from metric import iob2location
from config import args
from util import logger, CoNLLInput, CoNLLBatchInput
import torch
import torch.nn as nn
import os

import spacy
nlp = spacy.load('en_core_web_lg')

monkey.patch_all()

# args.batch_size = 1
args.crf_use_cuda = True
args.model_path = 'best.pt'

nam = ['PERSON', 'NORP', 'FAC', 'ORG', 'GPE', 'LOC', 'PRODUCT', 'EVENT', 'WORK_OF_ART', 'LAW']


def may_in(x, y): return x[0] >= y[0] and x[1] <= y[1]


def may_same(x, y): return not (x[1] <= y[0] or x[0] >= y[1])


class Model(ManagedModel):

    def init_model(self):
        self.model = BertNERNet()
        self.model.eval()
        self.model = self.model.cuda()
        model_name = '{}/{}'.format(args.model_dir, args.model_path)
        self.model.load_state_dict(torch.load(model_name))

    def predict(self, sents):
        spacy_list = [nlp(sent) for sent in sents]
        tokens_list = [[token.text for token in i] for i in spacy_list]
        indexs_list = [[token.idx for token in i] for i in spacy_list]
        ners_list = [[[ent.start_char, ent.end_char]
                      for ent in i.ents if ent.label_ in nam] for i in spacy_list]

        # sents_list = [[[token.text, token.idx] for token in nlp(sent)] for sent in sents]

        # ners_list = [[[ent.start_char, ent.end_char] for ent in nlp(sent).ents if ent.label_ in nam] for sent in sents]
        # tokens_list = [[i[0] for i in sent] for sent in sents_list]
        # indexs_list = [[i[1] for i in sent] for sent in sents_list]
        input_ids, segment_ids, input_mask, \
            input_char_ids, heads = CoNLLBatchInput(tokens_list)
        input_ids = input_ids.cuda()
        segment_ids = segment_ids.cuda()
        input_mask = input_mask.cuda()
        input_char_ids = input_char_ids.cuda()
        heads = heads.cuda()
        all_ys_, all_heads = [], []
        with torch.no_grad():
            _, ys_ = self.model(
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
            preds.append([y_ for head, y_ in zip(heads, ys_) if head == 1])

        batch_locations = []
        for i in range(len(preds)):
            pred = preds[i]

            location_idxs = iob2location(pred)

            tokens = tokens_list[i]
            indexs = indexs_list[i]
            ners = ners_list[i]
            ners_flag = [0] * len(ners)

            locations = []
            for location_idx in location_idxs:
                left, right = location_idx
                start = indexs[left]
                end = indexs[right-1] + len(tokens[right-1])
                # base rule
                # while not sent[i][end-1].isalpha():
                #     end -= 1

                # rule1
                for j in range(len(ners)):
                    if ners_flag[j] == 1:
                        continue
                    ner = ners[j]
                    if may_in([start, end], ner):
                        ners_flag[j] = 1
                        # if there is '-' in ner result
                        if sents[i][ner[0]:ner[1]].find('-') != -1:
                            start = ner[0]
                            end = ner[1]
                            break
                        # if there is 'the' or 'The' in ner result
                        if sents[i][ner[0]:ner[1]].find('the') != -1 or sents[i][ner[0]:ner[1]].find('The') != -1:
                            start = ner[0]
                            end = ner[1]
                            break

                locations.append([start, end])

            # rule2
            # if there is 'the' or 'The' in ner result and not exists in location result
            for j in range(len(ners)):
                if ners_flag[j] == 1:
                    continue
                ner = ners[j]
                if sents[i][ner[0]:ner[1]].find('the') != -1 or sents[i][ner[0]:ner[1]].find('The') != -1:
                    locations.append(ner)

            # rule3
            remain_locations = []
            for location in locations:
                flag = False
                for ner in ners:
                    # may_in or may_same?
                    if may_same(location, ner):
                        flag = True
                        remain_locations.append(location)
                        break
                if not flag:
                    if not sents[i][location[0]:location[1]].islower() and len(sents[i][location[0]:location[1]]) > 1:
                        remain_locations.append(location)

            locations = sorted(remain_locations, key=lambda x: x[0])

            batch_locations.append(locations)

            mention_ner = [sents[i][t[0]:t[1]] for t in ners]
            mention_location = [sents[i][t[0]:t[1]] for t in locations]

        return batch_locations


app = Flask(__name__)


@app.route('/batch', methods=['POST'])
def batch():
    sent = request.form.getlist('sent')
    locations = streamer.predict(sent)
    resp = {
        'locations': locations
    }
    return jsonify(resp)


if __name__ == "__main__":
    streamer = Streamer(Model, batch_size=16, max_latency=0.1, worker_num=16, cuda_devices=(0, 1, 2, 3))
    WSGIServer(("0.0.0.0", 3104), app).serve_forever()
