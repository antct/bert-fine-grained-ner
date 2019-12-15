import numpy as np


def f1(p, r):
    if r == 0.:
        return 0.
    return 2 * p * r / float(p + r)


def loose_macro(true, pred):
    num_entities = len(true)
    p = 0.
    r = 0.
    pred_count = 0
    true_count = 0
    for true_labels, pred_labels in zip(true, pred):
        if len(pred_labels):
            p += len(set(pred_labels).intersection(set(true_labels))) / float(len(pred_labels))
            pred_count += 1
        if len(true_labels):
            r += len(set(pred_labels).intersection(set(true_labels))) / float(len(true_labels))
            true_count += 1
    precision = p / pred_count
    recall = r / true_count
    return precision, recall, f1(precision, recall)


def loose_micro(true, pred):
    num_predicted_labels = 0.
    num_true_labels = 0.
    num_correct_labels = 0.
    for true_labels, pred_labels in zip(true, pred):
        num_predicted_labels += len(pred_labels)
        num_true_labels += len(true_labels)
        num_correct_labels += len(set(pred_labels).intersection(set(true_labels)))
    if num_predicted_labels > 0:
        precision = num_correct_labels / num_predicted_labels
    else:
        precision = 0.
    recall = num_correct_labels / num_true_labels
    return precision, recall, f1(precision, recall)
