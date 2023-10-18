from collections import defaultdict
import os
import numpy as np
from metrics import CorefEvaluator
import itertools
import random
from datetime import datetime
import sys
import torch


def is_rank_0() -> bool:
    return int(os.environ.get("RANK", "0")) == 0


def extract_clusters(sentence, start_mention_token, end_mention_token,
                     start_ent_token, end_ent_token, remove_singletons=False):
    # extract clusters from annotated sentence
    # TODO: fix bug: should not batch_decode, use token ids to get indices
    """

    :param sentence: annotated  sentence
    :param start_mention_token:
    :param end_mention_token:
    :param start_ent_token:
    :param end_ent_token:
    :param remove_singletons: if True remove singletons
    :return: clusters = tuple(c), c = tuple(m), m = tuple([s,e])
            mention_to_cluster = dict(tuple(m):tuple(c))
    """
    tokens = sentence.strip().split(' ')
    k = 0  # original sentence index
    m_starts = []
    mentions = []
    ents = []
    clusters = defaultdict(list)
    status = 'o'
    for i, s in enumerate(tokens):
        if s == start_mention_token:
            m_starts.append(k)
        elif s == end_mention_token:
            mentions.append((m_starts.pop(-1), k - 1))
        elif s == start_ent_token:
            status = 'e'
        elif s == end_ent_token:
            clusters[ents.pop(-1)].append(mentions.pop(-1))
            status = 'o'
        else:
            if status == 'e':
                ents.append(int(s))
            else:
                k += 1
    # cluster id from 0 to K
    # TODO: remove singletons?
    # remove empty mentions
    assert sorted(list(clusters.keys())) == list(range(len(clusters)))
    cluster_ls = [tuple(tuple(m) for m in clusters[k] if m[0] <= m[1]) for k in
                  sorted(list(clusters.keys()))]
    if remove_singletons:
        cluster_ls = [c for c in cluster_ls if len(c) > 1]
    else:
        cluster_ls = [c for c in cluster_ls if len(c) > 0]
    return cluster_ls


def get_mention_to_clusters(clusters):
    mention_to_cluster = {}
    for c in clusters:
        for mention in c:
            mention_to_cluster[tuple(mention)] = c
    return mention_to_cluster


def get_comput_metrics(tokenizer, start_mention_token, end_mention_token,
                       start_ent_token, end_ent_token, remove_singletons=False):
    def compute_metrics(eval_preds):
        mention_evaluator = MentionEvaluator()
        coref_evaluator = CorefEvaluator()
        preds, labels = eval_preds
        # In case the model returns more than the prediction logits
        if isinstance(preds, tuple):
            preds = preds[0]

        # somehow preds are also getting padded with -100s...
        preds = np.where(preds != -100, preds, tokenizer.pad_token_id)
        decoded_preds = tokenizer.batch_decode(preds,
                                               skip_special_tokens=True,
                                               clean_up_tokenization_spaces=False)

        # Replace -100s in the labels as we can't decode them
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(labels,
                                                skip_special_tokens=True,
                                                clean_up_tokenization_spaces=False)
        # extract clusters
        for pred, label in zip(decoded_preds, decoded_labels):
            pred_clusters = extract_clusters(pred,
                                             start_mention_token,
                                             end_mention_token,
                                             start_ent_token, end_ent_token,
                                             remove_singletons)
            gold_clusters = extract_clusters(label,
                                             start_mention_token,
                                             end_mention_token,
                                             start_ent_token,
                                             end_ent_token,
                                             remove_singletons)
            mention_to_predict = get_mention_to_clusters(pred_clusters)
            mention_to_gold = get_mention_to_clusters(gold_clusters)
            mentions_gold = list(mention_to_gold.keys())
            mentions_predict = list(mention_to_predict.keys())
            coref_evaluator.update(pred_clusters,
                                   gold_clusters,
                                   mention_to_predict,
                                   mention_to_gold)
            mention_evaluator.update(mentions_predict, mentions_gold)
        mention_p, mention_r, mention_f1 = mention_evaluator.get_prf()
        p, r, f1 = coref_evaluator.get_prf()
        results = {
            "mention precision": mention_p,
            "mention recall": mention_r,
            "mention f1": mention_f1,
            "precision": p,
            "recall": r,
            "f1": f1
        }
        return results

    return compute_metrics


def extract_mentions(token_ids, start_mention_id, end_mention_id):
    k = 0  # original sentence index
    m_starts = []
    mentions = []
    for i, s in enumerate(token_ids):
        if s == start_mention_id:
            m_starts.append(k)
        elif s == end_mention_id:
            mentions.append((m_starts.pop(-1), k - 1))
        else:
            k += 1
    mentions = [tuple(m) for m in mentions]
    return mentions


def get_mention_compute_metrics(tokenizer, start_mention_token,
                                end_mention_token):
    start_mention_id = tokenizer.encode(start_mention_token,
                                        add_special_tokens=False)
    end_mention_id = tokenizer.encode(end_mention_token,
                                      add_special_tokens=False)
    assert (len(start_mention_id) == 1 and len(end_mention_id) == 1)
    start_mention_id = start_mention_id[0]
    end_mention_id = end_mention_id[0]

    def compute_metrics(eval_preds):
        mention_evaluator = MentionEvaluator()
        preds, labels = eval_preds
        # In case the model returns more than the prediction logits
        if isinstance(preds, tuple):
            preds = preds[0]

        # somehow preds are also getting padded with -100s...
        preds = np.where(preds != -100, preds, tokenizer.pad_token_id)

        # Replace -100s in the labels as we can't decode them
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)

        # extract clusters
        for pred, label in zip(preds, labels):
            pred_mentions = extract_mentions(pred, start_mention_id,
                                             end_mention_id)
            gold_mentions = extract_mentions(label, start_mention_id,
                                             end_mention_id)
            mention_evaluator.update(pred_mentions, gold_mentions)
        mention_p, mention_r, mention_f1 = mention_evaluator.get_prf()
        results = {
            "mention_precision": mention_p,
            "mention_recall": mention_r,
            "mention_f1": mention_f1
        }
        return results

    return compute_metrics


def flat_lists(ls):
    return [l for s in ls for l in s]


def set_seeds(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)


def strtime(datetime_checkpoint):
    diff = datetime.now() - datetime_checkpoint
    return str(diff).rsplit('.')[0]  # Ignore below seconds


class Logger(object):

    def __init__(self, log_path, on=True):
        self.log_path = log_path
        self.on = on

        if self.on:
            while os.path.isfile(self.log_path):
                self.log_path += '+'

    def log(self, string, newline=True, force=False):
        if self.on or force:
            with open(self.log_path, 'a') as logf:
                logf.write(string)
                if newline: logf.write('\n')

            sys.stdout.write(string)
            if newline: sys.stdout.write('\n')
            sys.stdout.flush()


def split_list(ls, delimiter, include_delimiter):
    if not include_delimiter:
        spl = [list(y) for x, y in itertools.groupby(
            ls, lambda z: z == delimiter) if
               not x]
    else:
        spl = []
        for x, y in itertools.groupby(ls, lambda z: z == delimiter):
            if x:
                spl.append([])
            spl[-1].extend(y)
    return spl




