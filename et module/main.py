import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import os

from tqdm import tqdm
from torch.optim import lr_scheduler
from torch.utils import data
from torch.utils.tensorboard import SummaryWriter
from model import BertNet, ErnieNet, BertETNet, ErnieETNet
from util import OpenDataset, BertDataset, ErnieDataset, BertETDataset, ErnieETDataset, logger, labels
from config import args
from bert.optimization import BertAdam
from metric import loose_micro, loose_macro


def cal_loss(logits, targets, cutoff=[None], hierarchy=False):
    if hierarchy:
        from util import prior, tune
        loss_func = nn.BCELoss()
        tune = tune.cuda()
        process_func = nn.Sigmoid()
        logits = process_func(logits)
        logits = torch.mm(logits, tune)
        logits = torch.clamp(logits, 1e-10, 1.0)

    else:
        loss_func = nn.BCEWithLogitsLoss()

    loss = 0.
    comparison_tensor = torch.Tensor([1.0]).cuda()
    # 200, 500, None
    for i in range(len(cutoff)):
        if i == 0:
            # 0-200
            layer_targets = targets[:, :cutoff[i]]
        else:
            layer_targets = targets[:, cutoff[i-1]: cutoff[i]]

        layer_targets_sum = torch.sum(layer_targets, 1)
        if torch.sum(layer_targets_sum.data) > 0:
            mask = torch.squeeze(torch.nonzero(
                torch.min(layer_targets_sum.data, comparison_tensor)), dim=1)
            if i == 0:
                logits_masked = logits[:, :cutoff[i]][mask, :]
            else:
                logits_masked = logits[:, cutoff[i-1]: cutoff[i]][mask, :]
            mask = torch.autograd.Variable(mask).cuda()

            targets_masked = layer_targets.index_select(0, mask)
            layer_loss = loss_func(logits_masked, targets_masked)
            loss += layer_loss

    return loss


def filter_logits(logits, threshold=0.5):
    filter_idx = []
    sigmoid_fn = nn.Sigmoid()
    logits = sigmoid_fn(logits).data.cpu().clone()
    for logit in logits:
        t = []
        logit = logit.numpy()
        argmax_idx = np.argmax(logit)
        t.append(argmax_idx)
        t.extend([i for i in range(len(logit)) if logit[i] > threshold and i != argmax_idx])
        filter_idx.append(t)
    return filter_idx


def evaluate(model, iterator, epoch):
    model.eval()

    logger.info('eval examples {}'.format(len(iterator.dataset)))
    logger.info('eval batch size {}'.format(args.batch_size))

    true_labels, pred_labels = [], []
    eval_losses = []
    with torch.no_grad():
        for i, batch in enumerate(iterator):
            if args.bert_type == 'bert':
                sent_input_ids, sent_segment_ids, sent_input_mask, \
                    mention_input_ids, mention_segment_ids, mention_input_mask, \
                        char_input_ids, label = batch
                sent_input_ids = sent_input_ids.cuda()
                mention_input_ids = mention_input_ids.cuda()
                char_input_ids = char_input_ids.cuda()
                label = label.cuda()

                logits = model(
                    context_input_ids=sent_input_ids,
                    context_token_type_ids=sent_segment_ids,
                    context_attention_mask=sent_input_mask,
                    mention_input_ids=mention_input_ids,
                    mention_token_type_ids=mention_segment_ids,
                    mention_attention_mask=mention_input_mask,
                    char_input_ids=char_input_ids
                )
            else:
                sent_input_ids, sent_segment_ids, sent_input_mask, \
                    sent_input_ent, sent_ent_mask, \
                        mention_input_ids, mention_segment_ids, mention_input_mask, \
                            mention_input_ent, mention_ent_mask, \
                                char_input_ids, label = batch
                sent_input_ent = kb_embedding(sent_input_ent+1)
                mention_input_ent = kb_embedding(mention_input_ent+1)
                sent_input_ids = sent_input_ids.cuda()
                mention_input_ids = mention_input_ids.cuda()
                char_input_ids = char_input_ids.cuda()
                label = label.cuda()

                logits = model(
                    context_input_ids=sent_input_ids,
                    context_token_type_ids=sent_segment_ids,
                    context_attention_mask=sent_input_mask,
                    context_input_ent=sent_input_ent,
                    context_ent_mask=sent_ent_mask,
                    mention_input_ids=mention_input_ids,
                    mention_token_type_ids=mention_segment_ids,
                    mention_attention_mask=mention_input_mask,
                    mention_input_ent=mention_input_ent,
                    mention_ent_mask=mention_ent_mask,
                    char_input_ids=char_input_ids
                )


            loss_func = nn.BCEWithLogitsLoss()
            loss = loss_func(logits, label)

            idx_y_ = filter_logits(logits, threshold=args.bert_threshold)
            idx_y = [[i for i in range(0, len(labels)) if j[i] == 1.0] for j in label.cpu().numpy().tolist()]
            true_labels.extend(idx_y)
            pred_labels.extend(idx_y_)
            eval_losses.append(loss.mean().item())

        assert len(true_labels) == len(pred_labels)

        micro_p, micro_r, micro_f = loose_micro(true_labels, pred_labels)
        macro_p, macro_r, macro_f = loose_macro(true_labels, pred_labels)

        logger.info('mi_p: {}\tmi_r: {}\tmi_f: {}'.format(micro_p, micro_r, micro_f))
        logger.info('ma_p: {}\tma_r: {}\tma_f: {}'.format(macro_p, macro_r, macro_f))
        logger.info('eval loss: {}'.format(sum(eval_losses)/len(eval_losses)))

        writer.add_scalar('eval/micro-p', micro_p, epoch)
        writer.add_scalar('eval/micro-r', micro_r, epoch)
        writer.add_scalar('eval/micro-f', micro_f, epoch)

        writer.add_scalar('eval/macro-p', macro_p, epoch)
        writer.add_scalar('eval/macro-r', macro_r, epoch)
        writer.add_scalar('eval/macro-f', macro_f, epoch)

        writer.add_scalar('eval/loss', sum(eval_losses) / len(eval_losses), epoch)

        if args.mode != 'test':
            state = {
                'net': model.state_dict(),
                'optimizer': optimizer.state_dict(),
            }
            model_name = '{}/{}_{}_{}.pt'.format(args.model_dir, args.mode, args.bert_type, epoch)
            torch.save(state, model_name)
            logger.info('save model to {}'.format(model_name))


def train(model, iterator, optimizer, epoch):
    model.train()

    start_steps = len(iterator) * (epoch - 1)

    for i, batch in enumerate(iterator):
        global_steps = start_steps + i + 1

        if args.bert_type == 'bert':
            sent_input_ids, sent_segment_ids, sent_input_mask, \
                mention_input_ids, mention_segment_ids, mention_input_mask, \
                    char_input_ids, label = batch
            sent_input_ids = sent_input_ids.cuda()
            mention_input_ids = mention_input_ids.cuda()
            char_input_ids = char_input_ids.cuda()
            label = label.cuda()

            logits = model(
                context_input_ids=sent_input_ids,
                context_token_type_ids=sent_segment_ids,
                context_attention_mask=sent_input_mask,
                mention_input_ids=mention_input_ids,
                mention_token_type_ids=mention_segment_ids,
                mention_attention_mask=mention_input_mask,
                char_input_ids=char_input_ids
            )
        else:
            sent_input_ids, sent_segment_ids, sent_input_mask, \
                sent_input_ent, sent_ent_mask, \
                    mention_input_ids, mention_segment_ids, mention_input_mask, \
                        mention_input_ent, mention_ent_mask, \
                            char_input_ids, label = batch
            sent_input_ent = kb_embedding(sent_input_ent+1)
            mention_input_ent = kb_embedding(mention_input_ent+1)
            sent_input_ids = sent_input_ids.cuda()
            mention_input_ids = mention_input_ids.cuda()
            char_input_ids = char_input_ids.cuda()
            label = label.cuda()

            logits = model(
                context_input_ids=sent_input_ids,
                context_token_type_ids=sent_segment_ids,
                context_attention_mask=sent_input_mask,
                context_input_ent=sent_input_ent,
                context_ent_mask=sent_ent_mask,
                mention_input_ids=mention_input_ids,
                mention_token_type_ids=mention_segment_ids,
                mention_attention_mask=mention_input_mask,
                mention_input_ent=mention_input_ent,
                mention_ent_mask=mention_ent_mask,
                char_input_ids=char_input_ids
            )

        optimizer.zero_grad()

        loss = cal_loss(logits, label, cutoff=[None], hierarchy=False)

        loss.mean().backward()

        optimizer.step()

        if i % 10 == 0:
            writer.add_scalar('train/loss', loss.mean().item(), global_steps)
            logger.info("epoch: {:02d}\tstep: {:04d}\tloss: {}".format(epoch, i, loss.mean().item()))

        if i != 0 and i % 100 == 0:
            true_labels, pred_labels = [], []
            idx_y_ = filter_logits(logits, threshold=args.bert_threshold)
            idx_y = [[i for i in range(0, len(labels)) if j[i] == 1.0] for j in label.cpu().numpy().tolist()]
            
            true_labels.extend(idx_y)
            pred_labels.extend(idx_y_)

            micro_p, micro_r, micro_f = loose_micro(true_labels, pred_labels)
            macro_p, macro_r, macro_f = loose_macro(true_labels, pred_labels)

            writer.add_scalar('train/micro-p', micro_p, global_steps)
            writer.add_scalar('train/micro-r', micro_r, global_steps)
            writer.add_scalar('train/micro-f', micro_f, global_steps)

            writer.add_scalar('train/macro-p', macro_p, global_steps)
            writer.add_scalar('train/macro-r', macro_r, global_steps)
            writer.add_scalar('train/macro-f', macro_f, global_steps)

            logger.info('steps: {}\tmi_p: {:.4f}\tmi_r: {:.4f}\tmi_f: {:.4f}'.format(global_steps, micro_p, micro_r, micro_f))
            logger.info('steps: {}\tma_p: {:.4f}\tma_r: {:.4f}\tma_f: {:.4f}'.format(global_steps, macro_p, macro_r, macro_f))


if __name__ == "__main__":
    writer = SummaryWriter('summary/')

    logger.info(vars(args))

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    logger.info('device {}'.format(device))
    device_num = torch.cuda.device_count()
    logger.info('device count {}'.format(device_num))

    # set fix seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if device_num > 0:
        torch.cuda.manual_seed_all(args.seed)

    logger.info('model {}'.format(args.bert_type))
    if args.bert_type == 'bert':
        model = BertETNet()
    elif args.bert_type == 'ernie':
        model = ErnieETNet()
    else:
        raise Exception('model type error')

    # logger.info('write model graph')
    # dummy_long = torch.zeros(
    #     args.batch_size, args.bert_max_len, dtype=torch.long)
    # dummy_long_embed = torch.zeros(
    #     args.batch_size, args.bert_max_len, 100, dtype=torch.float)
    # if args.bert_type == 'bert':
    #     dummy_input = (dummy_long, ) * 3
    # else:
    #     dummy_input = (dummy_long, ) * 3
    #     dummy_input += (dummy_long_embed, )
    #     dummy_input += (dummy_long, )

    # writer.add_graph(model, dummy_input)

    logger.info('model cuda & parallel')
    if device == 'cuda':
        model = model.cuda()
    model = nn.DataParallel(model)

    if args.bert_type == 'ernie':
        logger.info('load kb embedding')
        vecs = []
        vecs.append([0] * 100)
        with open("kb/entity2vec.vec", 'r') as fin:
            lines = fin.readlines()

        lines = [[float(x) for x in line.strip().split('\t')] for line in lines]
        vecs.extend(lines)

        # for line in fin:
        #     vec = line.strip().split('\t')
        #     vec = [float(x) for x in vec]
        #     vecs.append(vec)

        kb_embedding = torch.FloatTensor(vecs)
        kb_embedding = torch.nn.Embedding.from_pretrained(kb_embedding)
        logger.info('kb embedding count {} dim {}'.format(len(vecs), len(vecs[0])))
        logger.info('kb embedding example {}'.format(vecs[1]))
        del vecs

    dataset_func = BertETDataset if args.bert_type == 'bert' else ErnieETDataset

    if args.mode == 'train':
        args.bert_freeze = True

        logger.info('load train iter')
        train_iter = data.DataLoader(
            dataset=dataset_func(args.train_dataset),
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers
        )

        logger.info('load eval iter')
        eval_iter = data.DataLoader(
            dataset=dataset_func(args.eval_dataset),
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers
        )

        param_optimizer = list(model.named_parameters())
        no_decay = ['bias', 'gamma', 'beta']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
             'weight_decay_rate': args.weight_decay_rate},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
             'weight_decay_rate': 0.0}
        ]

        optimizer = optim.Adam(optimizer_grouped_parameters, lr=args.lr)
        scheduler = lr_scheduler.StepLR(optimizer, args.lr_decay_epochs, args.lr_decay_rate)

        if not os.path.exists('{}'.format(args.model_dir)):
            os.makedirs('{}'.format(args.model_dir))

        for epoch in range(1, args.num_epochs+1):
            logger.info("==========epoch {} train start==========".format(epoch))
            logger.info('train examples {}'.format(len(train_iter.dataset)))
            logger.info('train batch size {}'.format(args.batch_size))
            logger.info('train lr {}'.format(scheduler.get_lr()[0]))
            writer.add_scalar('lr', scheduler.get_lr()[0], epoch)
            train(model, train_iter, optimizer, epoch)
            scheduler.step()
            logger.info("==========epoch {} train end==========".format(epoch))
            logger.info("==========epoch {} eval start==========".format(epoch))
            evaluate(model, eval_iter, epoch)
            logger.info("==========epoch {} eval end==========".format(epoch))


    if args.mode == 'fine_tune':
        args.bert_freeze = False
        logger.info('load model from {}/{}'.format(args.model_dir, args.model_path))
        model.load_state_dict(torch.load('{}/{}'.format(args.model_dir, args.model_path))['net'])
        logger.info('load train iter')
        train_iter = data.DataLoader(
            dataset=dataset_func(args.train_dataset),
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers
        )

        logger.info('load eval iter')
        eval_iter = data.DataLoader(
            dataset=dataset_func(args.eval_dataset),
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers
        )

        param_optimizer = list(model.named_parameters())

        if args.bert_type == 'bert':
            no_decay = ['bias', 'gamma', 'beta']
            
        else:
            no_grad = ['bert.encoder.layer.11.output.dense_ent', 'bert.encoder.layer.11.output.LayerNorm_ent']
            param_optimizer = [(n, p) for n, p in param_optimizer if not any(nd in n for nd in no_grad)]
            no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        
        
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
             'weight_decay_rate': args.weight_decay_rate},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
             'weight_decay_rate': 0.0}
        ]

        num_train_steps = int(len(train_iter.dataset) / args.batch_size * device_num) * args.num_epochs
        num_train_steps = num_train_steps if args.t_total else -1
        
        optimizer = BertAdam(
            params=optimizer_grouped_parameters,
            lr=args.bert_lr,
            warmup=args.warmup,
            t_total=num_train_steps
        )

        for epoch in range(1, args.num_epochs+1):
            logger.info("==========epoch {} fine tune start==========".format(epoch))
            logger.info('train examples {}'.format(len(train_iter.dataset)))
            logger.info('train batch size {}'.format(args.batch_size))
            logger.info('train lr {}'.format(optimizer.get_lr()[0]))
            writer.add_scalar('lr', optimizer.get_lr()[0], epoch)
            train(model, train_iter, optimizer, epoch)
            logger.info("==========epoch {} fine tune end==========".format(epoch))
            logger.info("==========epoch {} eval start==========".format(epoch))
            evaluate(model, eval_iter, epoch)
            logger.info("==========epoch {} eval end==========".format(epoch))



    if args.mode == 'test':
        logger.info('load model from {}/{}'.format(args.model_dir, args.model_path))
        model.load_state_dict(torch.load('{}/{}'.format(args.model_dir, args.model_path))['net'])

        logger.info('load test iter')
        test_iter = data.DataLoader(
            dataset=dataset_func(args.test_dataset),
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers
        )
        evaluate(model, test_iter, 1)

    if args.mode == 'trans':
        logger.info('load model from {}/{}'.format(args.model_dir, args.model_path))
        model.load_state_dict(torch.load('{}/{}'.format(args.model_dir, args.model_path))['net'])
        
        model_name = '{}/best.pt'.format(args.model_dir)
        torch.save(model.module.state_dict(), model_name)
        logger.info('save model to {}'.format(model_name))

    writer.close()
