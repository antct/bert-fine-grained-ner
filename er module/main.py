import torch
import torch.nn as nn
import torch.optim as optim
import os
import random
import numpy as np

from torch.optim import lr_scheduler
from torch.utils import data
from torch.utils.tensorboard import SummaryWriter
from model import BertNERNet
from bert.optimization import BertAdam
from util import CoNLLDataset, logger, CoNLLInput
from config import args
from metric import token_f, mention_f


def train(model, iterator, optimizer, epoch):
    model.train()
    start_steps = len(iterator) * (epoch - 1)
    for i, batch in enumerate(iterator):
        global_steps = start_steps + i + 1
        input_ids, segment_ids, input_mask, input_char_ids, heads, labels = batch
        input_ids = input_ids.cuda()
        labels = labels.cuda()
        optimizer.zero_grad()
        _, y_, loss = model(
            input_ids=input_ids,
            token_type_ids=segment_ids,
            attention_mask=input_mask,
            input_char_ids=input_char_ids,
            labels=labels
        )
        loss.mean().backward()
        optimizer.step()
        if i % 10 == 0:
            logger.info("epoch: {}\tstep: {:04d}\tloss: {}".format(epoch, i, loss.mean().item()))
            writer.add_scalar('train/loss', loss.mean().item(), global_steps)
        if i != 0 and i % 100 == 0:
            pass


def evaluate(model, iterator, epoch):
    model.eval()
    all_heads, all_ys, all_ys_ = [], [], []
    with torch.no_grad():
        for i, batch in enumerate(iterator):
            input_ids, segment_ids, input_mask, input_char_ids, heads, labels = batch
            input_ids = input_ids.cuda()
            labels = labels.cuda()
            _, y_, _ = model(
                input_ids=input_ids,
                token_type_ids=segment_ids,
                attention_mask=input_mask,
                input_char_ids=input_char_ids,
                labels=labels
            )
            all_heads.extend(heads.cpu().numpy().tolist())
            all_ys.extend(labels.cpu().numpy().tolist())
            all_ys_.extend(y_.cpu().numpy().tolist())

    true_labels, pred_labels = [], []
    for heads, ys, ys_ in zip(all_heads, all_ys, all_ys_):
        ys_ = [y_ for head, y_ in zip(heads, ys_) if head == 1]
        ys = [y for head, y in zip(heads, ys) if head == 1]
        true_labels.extend(ys)
        pred_labels.extend(ys_)

    assert len(true_labels) == len(pred_labels)

    t_p, t_r, t_f = token_f(true_labels, pred_labels)
    m_p, m_r, m_f = mention_f(true_labels, pred_labels)

    logger.info('epoch: {}\tt_p: {}\tt_r: {}\tt_f: {}'.format(epoch, t_p, t_r, t_f))
    logger.info('epoch: {}\tm_p: {}\tm_r: {}\tm_f: {}'.format(epoch, m_p, m_r, m_f))

    model_name = "{}/{}_{}.pt".format(args.model_dir, args.mode, epoch)
    torch.save(model.state_dict(), model_name)


if __name__ == "__main__":

    writer = SummaryWriter('summary/')
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info('device {}'.format(device))
    device_num = torch.cuda.device_count()
    logger.info('device count {}'.format(device_num))

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if device_num > 0:
        torch.cuda.manual_seed_all(args.seed)

    logger.info('model cuda & parallel')
    model = BertNERNet()
    if device == 'cuda':
        model = model.cuda()
    model = nn.DataParallel(model)

    if args.mode == 'train':
        args.bert_freeze = True
        logger.info('load train iter')
        train_iter = data.DataLoader(
            dataset=CoNLLDataset(args.train_dataset),
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers
        )
        logger.info('load eval iter')
        eval_iter = data.DataLoader(
            dataset=CoNLLDataset(args.eval_dataset),
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

        if not os.path.exists(args.model_dir):
            os.makedirs(args.model_dir)

        for epoch in range(1, args.num_epochs+1):
            logger.info('==========epoch {} train start=========='.format(epoch))
            logger.info('train examples {}'.format(len(train_iter.dataset)))
            logger.info('train batch size {}'.format(args.batch_size))
            logger.info('train lr {}'.format(scheduler.get_lr()[0]))
            train(model, train_iter, optimizer, epoch)
            logger.info('==========epoch {} train end=========='.format(epoch))
            scheduler.step()
            logger.info('==========epoch {} eval start=========='.format(epoch))
            logger.info('eval examples {}'.format(len(eval_iter.dataset)))
            logger.info('eval batch size {}'.format(args.batch_size))
            evaluate(model, eval_iter, epoch)
            logger.info('==========epoch {} eval end=========='.format(epoch))

    if args.mode == 'fine_tune':
        model_name = '{}/{}'.format(args.model_dir, args.model_path)
        model.load_state_dict(torch.load(model_name))
        logger.info('load model from {}'.format(model_name))

        args.bert_freeze = False

        logger.info('load train iter')
        train_iter = data.DataLoader(
            dataset=CoNLLDataset(args.train_dataset),
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers
        )

        logger.info('load eval iter')
        eval_iter = data.DataLoader(
            dataset=CoNLLDataset(args.eval_dataset),
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

        num_train_steps = int(len(train_iter.dataset) / args.batch_size * device_num) * args.num_epochs
        num_train_steps = num_train_steps if args.t_total else -1

        optimizer = BertAdam(
            params=optimizer_grouped_parameters,
            lr=args.bert_lr,
            warmup=args.warm_up,
            t_total=num_train_steps
        )

        if not os.path.exists(args.model_dir):
            os.makedirs(args.model_dir)

        for epoch in range(1, args.num_epochs+1):
            logger.info('==========epoch {} train start=========='.format(epoch))
            logger.info('train examples {}'.format(len(train_iter.dataset)))
            logger.info('train batch size {}'.format(args.batch_size))
            logger.info('train lr {}'.format(optimizer.get_lr()[0]))
            train(model, train_iter, optimizer, epoch)
            logger.info('==========epoch {} train end=========='.format(epoch))
            logger.info('==========epoch {} eval start=========='.format(epoch))
            logger.info('eval examples {}'.format(len(eval_iter.dataset)))
            logger.info('eval batch size {}'.format(args.batch_size))
            evaluate(model, eval_iter, epoch)
            logger.info('==========epoch {} eval end=========='.format(epoch))

    if args.mode == 'test':
        model_name = '{}/{}'.format(args.model_dir, args.model_path)
        model.load_state_dict(torch.load(model_name))
        test_iter = data.DataLoader(
            dataset=CoNLLDataset(args.test_dataset),
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers
        )
        logger.info('test examples {}'.format(len(test_iter.dataset)))
        logger.info('test batch size {}'.format(args.batch_size))
        evaluate(model, test_iter, 0)

    if args.mode == 'trans':
        model_name = '{}/{}'.format(args.model_dir, args.model_path)
        model.load_state_dict(torch.load(model_name))
        logger.info('load model from {}'.format(model_name))

        model_name = "{}/best.pt".format(args.model_dir)
        torch.save(model.module.state_dict(), model_name)
        logger.info('save model to {}'.format(model_name))
        
