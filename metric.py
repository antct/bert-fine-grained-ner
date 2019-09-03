from util import TAG2IDX, IDX2TAG


def iob2location(labels):
    import re
    locations = []
    labels = list(labels)
    labels = [str(i) for i in labels]
    str_labels = ''.join(labels)
    for i in re.finditer('32*', str_labels):
        locations.append(i.span())
    return locations


def f1(p, r):
    if r == 0.:
        return 0.
    return 2 * p * r / float(p + r)


def tag_f(true_labels, pred_labels):
    pass


def token_f(true_labels, pred_labels):
    p_true, p_false = 0, 0
    r_true, r_false = 0, 0
    for true_label, pred_label in zip(true_labels, pred_labels):
        if true_label > 0:
            if true_label == pred_label:
                p_true += 1
            else:
                p_false += 1
        if pred_label > 0:
            if true_label == pred_label:
                r_true += 1
            else:
                r_false += 1
    p = p_true / (p_true + p_false)
    r = r_true / (r_true + r_false)
    f = f1(p, r)
    return p, r, f


def mention_f(true_labels, pred_labels):
    true_labels = iob2location(true_labels)
    pred_labels = iob2location(pred_labels)

    p_true, p_false = 0, 0
    r_true, r_false = 0, 0
    for i in true_labels:
        if i in pred_labels:
            p_true += 1
        else:
            p_false += 1
    for i in pred_labels:
        if i in true_labels:
            r_true += 1
        else:
            r_false += 1
    p = p_true / (p_true + p_false)
    r = r_true / (r_true + r_false)
    f = f1(p, r)
    return p, r, f
