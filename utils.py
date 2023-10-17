import csv
import json
from dataclasses import dataclass
from typing import Tuple, List, Dict
from collections import defaultdict
import os
import re
import numpy as np
from metrics import CorefEvaluator
import logging
import itertools
import random
import torch
from datetime import datetime
import sys
import torch
from apex.multi_tensor_apply import multi_tensor_applier
# import bitsandbytes as bnb


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


def find_all_linear_names(bits, model):
    cls = bnb.nn.Linear4bit if bits == 4 else (
        bnb.nn.Linear8bitLt if bits == 8 else torch.nn.Linear)
    lora_module_names = set()
    for name, module in model.named_modules():
        if isinstance(module, cls):
            names = name.split('.')
            lora_module_names.add(
                names[0] if len(names) == 1 else f".{names[-1]}")

    # if 'lm_head' in lora_module_names:  # needed for 16-bit
    #     lora_module_names.remove('lm_head')
    return list(lora_module_names)


# Adapted from apex FuseAdam


class FusedAdam(torch.optim.Optimizer):
    """Implements Adam algorithm.

    Currently GPU-only.  Requires Apex to be installed via
    ``pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./``.

    This version of fused Adam implements 2 fusions.

      * Fusion of the Adam update's elementwise operations
      * A multi-tensor apply launch that batches the elementwise updates applied to all the model's parameters into one or a few kernel launches.

    :class:`apex.optimizers.FusedAdam` may be used as a drop-in replacement for ``torch.optim.AdamW``,
    or ``torch.optim.Adam`` with ``adam_w_mode=False``::

        opt = apex.optimizers.FusedAdam(model.parameters(), lr = ....)
        ...
        opt.step()

    :class:`apex.optimizers.FusedAdam` may be used with or without Amp.  If you wish to use :class:`FusedAdam` with Amp,
    you may choose any ``opt_level``::

        opt = apex.optimizers.FusedAdam(model.parameters(), lr = ....)
        model, opt = amp.initialize(model, opt, opt_level="O0" or "O1 or "O2")
        ...
        opt.step()

    In general, ``opt_level="O1"`` is recommended.


    .. warning::
        A previous version of :class:`FusedAdam` allowed a number of additional arguments to ``step``.  These additional arguments
        are now deprecated and unnecessary.

    Adam was been proposed in `Adam: A Method for Stochastic Optimization`_.

    Arguments:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups.
        lr (float, optional): learning rate. (default: 1e-3)
        betas (Tuple[float, float], optional): coefficients used for computing
            running averages of gradient and its square. (default: (0.9, 0.999))
        eps (float, optional): term added to the denominator to improve
            numerical stability. (default: 1e-8)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
        amsgrad (boolean, optional): whether to use the AMSGrad variant of this
            algorithm from the paper `On the Convergence of Adam and Beyond`_
            (default: False) NOT SUPPORTED in FusedAdam!
        adam_w_mode (boolean, optional): Apply L2 regularization or weight decay
            True for decoupled weight decay(also known as AdamW) (default: True)
        set_grad_none (bool, optional): whether set grad to None when zero_grad()
            method is called. (default: True)

    .. _Adam - A Method for Stochastic Optimization:
        https://arxiv.org/abs/1412.6980
    .. _On the Convergence of Adam and Beyond:
        https://openreview.net/forum?id=ryQu7f-RZ
    """

    def __init__(self, params, lr=1e-3, bias_correction=True,
                 betas=(0.9, 0.999), eps=1e-8, adam_w_mode=True,
                 weight_decay=0., amsgrad=False, set_grad_none=True):

        if amsgrad:
            raise RuntimeError(
                'FusedAdam does not support the AMSGrad variant.')
        defaults = dict(lr=lr, bias_correction=bias_correction,
                        betas=betas, eps=eps, weight_decay=weight_decay)
        super(FusedAdam, self).__init__(params, defaults)
        self.adam_w_mode = 1 if adam_w_mode else 0
        self.set_grad_none = set_grad_none
        if multi_tensor_applier.available:
            import amp_C
            # Skip buffer
            self._dummy_overflow_buf = torch.cuda.IntTensor([0])
            self.multi_tensor_adam = amp_C.multi_tensor_adam
        else:
            raise RuntimeError(
                'apex.optimizers.FusedAdam requires cuda extensions')

    def zero_grad(self):
        if self.set_grad_none:
            for group in self.param_groups:
                for p in group['params']:
                    p.grad = None
        else:
            super(FusedAdam, self).zero_grad()

    def step(self, closure=None, grads=None, output_params=None, scale=None,
             grad_norms=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.

        The remaining arguments are deprecated, and are only retained (for the moment) for error-checking purposes.
        """
        if any(p is not None for p in
               [grads, output_params, scale, grad_norms]):
            raise RuntimeError(
                'FusedAdam has been updated.  Simply initialize it identically to torch.optim.Adam, and call step() with no arguments.')
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            bias_correction = 1 if group['bias_correction'] else 0
            beta1, beta2 = group['betas']

            # assume same step across group now to simplify things
            # per parameter step can be easily support by making it tensor, or pass list into kernel
            if 'step' in group:
                group['step'] += 1
            else:
                group['step'] = 1

            # Modified:
            # enable for multi-gpu training:
            # create lists for multi-tensor apply
            # Group them by device
            device_to_group = defaultdict(
                lambda: {'g_16': [], 'p_16': [], 'm_16': [], 'v_16': [],
                         'g_bf': [], 'p_bf': [], 'm_bf': [], 'v_bf': [],
                         'g_32': [], 'p_32': [], 'm_32': [], 'v_32': []
                         }
            )

            for p in group['params']:
                # TODO:: Here, check the device of p
                p_device = p.get_device()

                if p.grad is None:
                    continue
                if p.grad.data.is_sparse:
                    raise RuntimeError(
                        'FusedAdam does not support sparse gradients, please consider SparseAdam instead')

                state = self.state[p]
                # State initialization
                if len(state) == 0:
                    # Exponential moving average of gradient values
                    state['exp_avg'] = torch.zeros_like(p.data)
                    # Exponential moving average of squared gradient values
                    state['exp_avg_sq'] = torch.zeros_like(p.data)
                # Reloaded optimizer, move back to the correct place
                if state['exp_avg'].get_device() != p_device:
                    state['exp_avg'] = state['exp_avg'].to(p_device)
                if state['exp_avg_sq'].get_device() != p_device:
                    state['exp_avg_sq'] = state['exp_avg_sq'].to(p_device)

                if p.dtype == torch.float16:
                    device_to_group[p_device]['g_16'].append(p.grad.data)
                    device_to_group[p_device]['p_16'].append(p.data)
                    device_to_group[p_device]['m_16'].append(state['exp_avg'])
                    device_to_group[p_device]['v_16'].append(
                        state['exp_avg_sq'])
                elif p.dtype == torch.bfloat16:
                    device_to_group[p_device]['g_bf'].append(p.grad)
                    device_to_group[p_device]['p_bf'].append(p)
                    device_to_group[p_device]['m_bf'].append(state['exp_avg'])
                    device_to_group[p_device]['v_bf'].append(
                        state['exp_avg_sq'])
                elif p.dtype == torch.float32:
                    device_to_group[p_device]['g_32'].append(p.grad.data)
                    device_to_group[p_device]['p_32'].append(p.data)
                    device_to_group[p_device]['m_32'].append(state['exp_avg'])
                    device_to_group[p_device]['v_32'].append(
                        state['exp_avg_sq'])
                else:
                    raise RuntimeError('FusedAdam only support fp16 and fp32.')

            for device in device_to_group:
                g_16, p_16, m_16, v_16 = device_to_group[device]['g_16'], \
                                         device_to_group[device]['p_16'], \
                                         device_to_group[device]['m_16'], \
                                         device_to_group[device]['v_16']
                g_bf, p_bf, m_bf, v_bf = device_to_group[device]['g_bf'], \
                                         device_to_group[device]['p_bf'], \
                                         device_to_group[device]['m_bf'], \
                                         device_to_group[device]['v_bf']
                g_32, p_32, m_32, v_32 = device_to_group[device]['g_32'], \
                                         device_to_group[device]['p_32'], \
                                         device_to_group[device]['m_32'], \
                                         device_to_group[device]['v_32']

                if (len(g_16) > 0):
                    multi_tensor_applier(self.multi_tensor_adam,
                                         self._dummy_overflow_buf,
                                         [g_16, p_16, m_16, v_16],
                                         group['lr'],
                                         beta1,
                                         beta2,
                                         group['eps'],
                                         group['step'],
                                         self.adam_w_mode,
                                         bias_correction,
                                         group['weight_decay'])
                if g_bf:
                    multi_tensor_applier(
                        self.multi_tensor_adam,
                        self._dummy_overflow_buf,
                        [g_bf, p_bf, m_bf, v_bf],
                        group['lr'],
                        beta1,
                        beta2,
                        group['eps'],
                        group['step'],
                        self.adam_w_mode,
                        bias_correction,
                        group['weight_decay'],
                    )
                if (len(g_32) > 0):
                    multi_tensor_applier(self.multi_tensor_adam,
                                         self._dummy_overflow_buf,
                                         [g_32, p_32, m_32, v_32],
                                         group['lr'],
                                         beta1,
                                         beta2,
                                         group['eps'],
                                         group['step'],
                                         self.adam_w_mode,
                                         bias_correction,
                                         group['weight_decay'])

        return loss
