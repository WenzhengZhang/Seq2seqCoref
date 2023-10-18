import networkx as nx
import json
import os
import random
from torch.utils.data import Dataset, DataLoader
from transformers import DataCollatorForSeq2Seq
from collections import defaultdict
import re
import torch
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, List, Optional, Tuple, Union
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from transformers.utils import PaddingStrategy
from alignment import global_align, affine_global_align
from utils import split_list


class JointDataset(Dataset):

    def __init__(self, tokenizer,
                 data_args, train_args, split):
        self.tokenizer = tokenizer
        self.data_args = data_args
        self.train_args = train_args
        self.split = split
        self.all_samples, self.doc_labels, self.id_to_name = self.load_dataset()
        self.samples = None if self.split == 'train' else [
            s for data_samples in self.all_samples.values() for s in
            data_samples
        ]

    def __len__(self):
        if self.split == 'train':
            num_samples = 0
            for s in self.all_samples.values():
                num_samples += min(self.data_args.joint_num_samples, len(s))
        else:
            num_samples = len(self.samples)
        return num_samples

    def set_samples(self, epoch):
        # subsample larger datasets and then concat them
        sample_seed = self.train_args.seed + epoch
        min_num_samples = min(len(s) for s in self.all_samples.values())
        samples = []
        for data_name, data_samples in self.all_samples.items():
            if len(data_samples) > min_num_samples:
                subsamples = random.Random(sample_seed).sample(
                    data_samples, self.data_args.joint_num_samples)
            else:
                subsamples = data_samples
            samples += subsamples
        self.samples = samples

    def _load_single_data(self, data_dir,
                          data_name,
                          max_len,
                          thred):

        samples = []
        doc_labels = {}
        id_to_name = {}
        data_path = os.path.join(
            data_dir,
            f'{self.split}.t5-small.english.{max_len}.jsonlines')
        with open(data_path, 'r') as f:
            for line in f:
                item = json.loads(line)
                doc_key = item['doc_key']
                doc_id = re.sub(r'_\d+$', '', doc_key)
                id_to_name[doc_id] = data_name
                if (self.train_args.seq2seq_type == 'action' or
                    self.train_args.seq2seq_type == 'input_feed') and \
                        self.train_args.action_type == 'non_integer' and \
                        self.train_args.add_mention_end:
                    target_sent = self.tokenizer.convert_tokens_to_ids(
                        item['target_mention_end_sentence'])
                else:
                    target_sent = self.tokenizer.convert_tokens_to_ids(
                        item['target_sentence'])
                if self.train_args.seq2seq_type == 'action' or \
                        self.train_args.seq2seq_type == 'input_feed':
                    if self.train_args.action_type == 'integer':
                        target_seq = self.tokenizer.convert_tokens_to_ids(
                            item['target_action'])
                    elif self.train_args.action_type == 'non_integer':
                        if self.train_args.add_mention_end:
                            target_seq = self.tokenizer.convert_tokens_to_ids(
                                item['target_mention_end_action'])
                        else:
                            target_seq = self.tokenizer.convert_tokens_to_ids(
                                item['target_action'])
                    else:
                        raise ValueError("wrong action type ("
                                         "integer/non_integer)")
                elif self.train_args.seq2seq_type == 'short_seq':
                    target_seq = self.tokenizer.convert_tokens_to_ids(
                        item['target_short_sentence'])
                elif self.train_args.seq2seq_type == 'full_seq':
                    target_seq = deepcopy(target_sent)
                elif self.train_args.seq2seq_type == 'tagging':
                    target_seq = self.tokenizer.convert_tokens_to_ids(
                        item['target_action'])
                    # set the last token as eos token
                    target_seq[-1] = self.tokenizer.eos_token_id
                else:
                    raise ValueError('wrong seq2seq type')
                sample = {'doc_key': doc_key,
                          'sentence': self.tokenizer.convert_tokens_to_ids(
                              item['sentence']),
                          'target_sentence': target_sent,
                          'target_seq': target_seq,
                          'subtoken_map': item['subtoken_map'],
                          'seg_clusters': [[tuple(m) for m in c] for c in item[
                              'seg_clusters'] if len(c) >= thred],
                          'offset': item['offset']
                          }
                doc_labels[doc_id] = [[tuple(m) for m in c] for c in item[
                    'gold_clusters']]
                samples.append(sample)
        return samples, doc_labels, id_to_name

    def load_dataset(self):
        doc_labels = {}
        id_to_name = {}
        samples = {}
        max_lens = self.data_args.joint_max_train_lens.split(
            ',') if self.split == 'train' else \
            self.data_args.joint_max_eval_lens.split(',')
        max_lens = [int(l) for l in max_lens]
        threds = self.train_args.joint_threds.split(',')
        threds = [int(t) for t in threds]
        data_dirs = self.data_args.joint_data_dirs.split(',')
        data_names = self.train_args.joint_data_names.split(',')
        for data_dir, data_name, max_len, thred in zip(
                data_dirs, data_names, max_lens, threds):
            single_samples, single_doc_labels, single_id_to_name = \
                self._load_single_data(data_dir, data_name, max_len, thred)
            samples[data_name] = single_samples
            doc_labels.update(single_doc_labels)
            id_to_name.update(single_id_to_name)
        return samples, doc_labels, id_to_name

    def __getitem__(self, index):
        sample = self.samples[index]
        input_ids = torch.tensor(sample['sentence'], dtype=torch.long)
        if self.train_args.seq2seq_type == 'action' or \
                self.train_args.seq2seq_type == 'input_feed':
            label_ids = torch.tensor(sample['target_sentence'],
                                     dtype=torch.long)
            target_ids = torch.tensor(sample['target_seq'], dtype=torch.long)
            input_len, tgt_len = input_ids.size(0), label_ids.size(0)
            attention_mask = torch.tensor([1] * input_len, dtype=torch.long)
            src_encoding = {'input_ids': input_ids,
                            'attention_mask': attention_mask,
                            'decoder_labels': label_ids,
                            'labels': target_ids
                            }
        else:
            label_ids = torch.tensor(sample['target_seq'],
                                     dtype=torch.long)
            input_len, tgt_len = input_ids.size(0), label_ids.size(0)
            attention_mask = torch.tensor([1] * input_len, dtype=torch.long)
            src_encoding = {'input_ids': input_ids,
                            'attention_mask': attention_mask,
                            'labels': label_ids,
                            }
        return src_encoding


class CorefDataset(Dataset):

    def __init__(self, tokenizer,
                 data_args, train_args, split):
        self.tokenizer = tokenizer
        self.data_args = data_args
        self.train_args = train_args
        self.split = split
        # self.task_prefix = self.data_args.task_prefix
        # convert tokens to ids for each sample
        self.samples, self.doc_labels = self.load_dataset()

    def __len__(self):
        return len(self.samples)

    def load_dataset(self):
        max_len = self.data_args.max_train_len if self.split == 'train' else \
            self.data_args.max_eval_len
        data_path = os.path.join(
            self.data_args.data_dir,
            f'{self.split}.t5-small.english.{max_len}.jsonlines')
        samples = []
        doc_labels = {}
        thred = 1 if self.train_args.allow_singletons else 2
        with open(data_path, 'r') as f:
            for line in f:
                item = json.loads(line)
                doc_key = item['doc_key']
                doc_id = re.sub(r'_\d+$', '', doc_key)
                if (self.train_args.seq2seq_type == 'action' or
                    self.train_args.seq2seq_type == 'input_feed') and \
                        self.train_args.action_type == 'non_integer' and \
                        self.train_args.add_mention_end:
                    target_sent = self.tokenizer.convert_tokens_to_ids(
                        item['target_mention_end_sentence'])
                else:
                    target_sent = self.tokenizer.convert_tokens_to_ids(
                        item['target_sentence'])
                if self.train_args.seq2seq_type == 'action' or \
                        self.train_args.seq2seq_type == 'input_feed':
                    if self.train_args.action_type == 'integer':
                        target_seq = self.tokenizer.convert_tokens_to_ids(
                            item['target_action'])
                    elif self.train_args.action_type == 'non_integer':
                        if self.train_args.add_mention_end:
                            target_seq = self.tokenizer.convert_tokens_to_ids(
                                item['target_mention_end_action'])
                        else:
                            target_seq = self.tokenizer.convert_tokens_to_ids(
                                item['target_action'])
                    else:
                        raise ValueError("wrong action type ("
                                         "integer/non_integer)")
                elif self.train_args.seq2seq_type == 'short_seq':
                    target_seq = self.tokenizer.convert_tokens_to_ids(
                        item['target_short_sentence'])
                elif self.train_args.seq2seq_type == 'full_seq':
                    target_seq = deepcopy(target_sent)
                elif self.train_args.seq2seq_type == 'tagging':
                    target_seq = self.tokenizer.convert_tokens_to_ids(
                        item['target_action'])
                    # set the last token as eos token
                    target_seq[-1] = self.tokenizer.eos_token_id
                else:
                    raise ValueError('wrong seq2seq type')
                sample = {'doc_key': doc_key,
                          'sentence': self.tokenizer.convert_tokens_to_ids(
                              item['sentence']),
                          'target_sentence': target_sent,
                          'target_seq': target_seq,
                          'subtoken_map': item['subtoken_map'],
                          'seg_clusters': [[tuple(m) for m in c] for c in item[
                              'seg_clusters'] if len(c) >= thred],
                          'offset': item['offset']
                          }
                doc_labels[doc_id] = [[tuple(m) for m in c] for c in item[
                    'gold_clusters']]
                samples.append(sample)
        return samples, doc_labels

    def __getitem__(self, index):
        sample = self.samples[index]
        input_ids = torch.tensor(sample['sentence'], dtype=torch.long)
        if self.train_args.seq2seq_type == 'action' or \
                self.train_args.seq2seq_type == 'input_feed':
            label_ids = torch.tensor(sample['target_sentence'],
                                     dtype=torch.long)
            target_ids = torch.tensor(sample['target_seq'], dtype=torch.long)
            input_len, tgt_len = input_ids.size(0), label_ids.size(0)
            attention_mask = torch.tensor([1] * input_len, dtype=torch.long)
            src_encoding = {'input_ids': input_ids,
                            'attention_mask': attention_mask,
                            'decoder_labels': label_ids,
                            'labels': target_ids
                            }
        else:
            label_ids = torch.tensor(sample['target_seq'],
                                     dtype=torch.long)
            input_len, tgt_len = input_ids.size(0), label_ids.size(0)
            attention_mask = torch.tensor([1] * input_len, dtype=torch.long)
            src_encoding = {'input_ids': input_ids,
                            'attention_mask': attention_mask,
                            'labels': label_ids,
                            }
        return src_encoding


def get_document_predicts(doc_preds: List[List]) -> List[
    List[Tuple[int, int]]]:
    """
    Aggregate predictions for each chunk into document-level predictions.
    """
    if len(doc_preds) == 0:
        return []
    graph = nx.compose_all([nx.complete_graph(p) for p in doc_preds])

    processed_groups = []
    for component in nx.connected_components(graph):
        processed_group = []
        for start, end in sorted(component, key=lambda x: (x[0], -x[1])):
            # add this entity if it does not overlap with the previous one
            condition = not any(
                [s < start < e < end for (s, e) in processed_group])
            # if len(processed_group) == 0 or start >= processed_group[-1][1]:
            #     processed_group.append((start, end))
            if len(processed_group) == 0 or condition:
                processed_group.append((start, end))

        processed_groups.append(processed_group)

    return [[(start, end) for start, end in group] for group in
            processed_groups]


# adapted from https://github.com/lyutyuh/ASP/blob/12b80a7cacc0edf33b77b507102f583380e7e1f1/data/t5minimize_coref.py#L259
def normalize_word(word, use_br_dict=False):
    br_dict = {"-LRB-": "(", "-RRB-": ")", "-LSB-": "[", "-RSB-": "]"}
    # br_dict = {"(": "-LRB-", ")": "-RRB-", "[": "-LSB-", ']': "-RSB-"}
    # br_dict = {"(": "[", ")": "]", "-LRB-": "[", "-RRB-": "]",
    #            "-LSB-": "[", "-RSB-": "]"}

    if use_br_dict and word in br_dict:
        word = br_dict[word]
        return word
    elif word == "/." or word == "/?":
        return word[1:]
    elif word == "''" or word == "``":  # <unk> otherwise
        return "\""
    elif word == "`":  # <unk> otherwise
        return "\'"
    else:
        return word.replace('{', '(').replace('}', ')')


def parse_int_output_tokens(input_ids, output_ids,
                            special_ids, subtoken_map, tokenizer,
                            thred, is_tagging):
    rec_ids, new_id = [], -1
    ment_start_stack = []
    unmatched_clusters = defaultdict(list)
    new_output_ids = []
    if is_tagging:
        new_input_ids = [special_ids['copy'] for t in input_ids if
                         t != tokenizer.pad_token_id and t != special_ids[
                             'eos']]
        new_input_ids.append(special_ids['eos'])
    else:
        new_input_ids = [t for t in input_ids if t != tokenizer.pad_token_id]
    token_mentions = []
    for i in range(len(output_ids)):
        if output_ids[i] == tokenizer.pad_token_id:
            break
        if output_ids[i] == special_ids['mention_start']:
            new_id += 1
            ment_start_stack.append([new_id, 'name', []])
            if is_tagging:
                new_output_ids.append(output_ids[i])
        elif output_ids[i] == special_ids['mention_end']:
            new_id += 0
            if is_tagging:
                new_output_ids.append(output_ids[i])
            if len(ment_start_stack) > 0:
                item = ment_start_stack.pop()
                if item[1] == "ent":
                    unmatched_clusters[tuple(item[-1])].append(
                        (item[0], new_id))
        else:
            # a normal token
            # if output_ids[i] == special_ids['sep']:
            #     status = "ent"
            if len(ment_start_stack) > 0:
                # inside some entities
                if output_ids[i] == special_ids['sep']:
                    ment_start_stack[-1][1] = "ent"
                    if is_tagging:
                        new_output_ids.append(output_ids[i])
                else:
                    if ment_start_stack[-1][1] == 'ent':
                        ment_start_stack[-1][2].append(output_ids[i])
                        if is_tagging:
                            new_output_ids.append(output_ids[i])
                    elif ment_start_stack[-1][1] == 'name':
                        new_id += 1
                        rec_ids.append(output_ids[i])
                        if is_tagging:
                            new_output_ids.append(input_ids[new_id])
                    else:
                        raise ValueError('wrong status')
            else:
                # outside
                new_id += 1
                rec_ids.append(output_ids[i])
                if is_tagging:
                    new_output_ids.append(input_ids[new_id])
        if output_ids[i] == special_ids['mention_start']:
            new_id -= 1
    # thred = 1 if allow_singletons else 2
    # Needleman-Wunsch text alignment algorithm
    wrong_reconstruction = (rec_ids != new_input_ids)
    if wrong_reconstruction:
        print(f'new input ids {new_input_ids}')
        print(f'reconstructed ids {rec_ids}')
        print(f'out ids {output_ids}')
        print('wrong reconstruction! please debug')
        matching = global_align(new_input_ids, rec_ids)

        # update predicted entities with the positions in the original sentence
        clusters = defaultdict(list)

        for ent_id, ments in unmatched_clusters.items():
            for start, end in ments:
                new_start = None  # start in the original sequence
                new_end = None  # end in the original sequence

                for j in range(start, end + 1):
                    if j in matching:
                        if new_start is None:
                            new_start = matching[j]

                        new_end = matching[j]

                if new_start is not None:
                    # predict entity
                    clusters[ent_id].append((
                        subtoken_map[new_start], subtoken_map[new_end]))
                    token_mentions.append((new_start, new_end))
        predict_clusters = [list(set(v)) for k, v in clusters.items() if
                            len(set(v)) >= thred]
        token_mentions = list(set(token_mentions))
    else:
        clusters = [[(subtoken_map[m[0]], subtoken_map[m[1]]) for m in v] for v
                    in
                    unmatched_clusters.values()]
        predict_clusters = [list(set(v)) for v in clusters if len(set(v)) >=
                            thred]
        token_mentions = [(m[0], m[1]) for v in unmatched_clusters.values()
                          for m in v]
        token_mentions = list(set(token_mentions))
    if not is_tagging:
        new_output_ids = output_ids
    return predict_clusters, token_mentions, new_output_ids


def parse_short_target_tokens(input_ids, output_ids,
                              special_ids, subtoken_map, tokenizer,
                              align_mode, thred, split_sentence):
    #  support mark sentence, align sentence by sentence
    rec_ids, new_id = [], -1
    ment_start_stack = []
    unmatched_clusters = defaultdict(list)
    new_input_ids = [t for t in input_ids if t != tokenizer.pad_token_id]
    for i in range(len(output_ids)):
        if output_ids[i] == tokenizer.pad_token_id:
            break
        if output_ids[i] == special_ids['mention_start']:
            ment_start_stack.append([new_id + 1, 'name', []])
        elif output_ids[i] == special_ids['mention_end']:
            if len(ment_start_stack) > 0:
                item = ment_start_stack.pop()
                if item[1] == "ent":
                    unmatched_clusters[tuple(item[-1])].append(
                        (item[0], new_id))
        else:
            # a normal token
            if len(ment_start_stack) > 0:
                # inside some entities
                if output_ids[i] == special_ids['sep']:
                    ment_start_stack[-1][1] = "ent"
                else:
                    if ment_start_stack[-1][1] == 'ent':
                        ment_start_stack[-1][2].append(output_ids[i])
                    elif ment_start_stack[-1][1] == 'name':
                        new_id += 1
                        rec_ids.append(output_ids[i])
                    else:
                        raise ValueError('wrong status')

            else:
                # outside
                new_id += 1
                rec_ids.append(output_ids[i])
        # mapping.append(new_id)
    # thred = 1 if allow_singletons else 2
    # Affine global text alignment algorithm
    if split_sentence:
        input_sents = split_list(
            new_input_ids, special_ids['sentence_start'], True)
        out_sents = split_list(rec_ids, special_ids['sentence_start'], True)
        try:
            assert len(input_sents) == len(out_sents)
            aligned_input_ids, aligned_rec_ids, matching = [], [], {}
            input_offset, out_offset = 0, 0
            for input_sent, out_sent in zip(input_sents, out_sents):
                aligned_input_sent, aligned_out_sent, sent_match = \
                    affine_global_align(input_sent, out_sent,
                                        special_ids['copy'],
                                        align_mode)
                aligned_input_ids.extend(aligned_input_sent)
                aligned_rec_ids.extend(aligned_out_sent)
                matching.update(
                    {k + out_offset: v + input_offset for k, v in
                     sent_match.items()})
                input_offset += len(input_sent)
                out_offset += len(out_sent)
        except AssertionError:
            print(f'input sents and out sents different length '
                  f'{len(input_sents)} vs {len(out_sents)}, have to use '
                  f'global alignment')
            aligned_input_ids, aligned_rec_ids, matching = affine_global_align(
                new_input_ids, rec_ids, special_ids['copy'], align_mode)
    else:
        aligned_input_ids, aligned_rec_ids, matching = affine_global_align(
            new_input_ids, rec_ids, special_ids['copy'], align_mode)
    # update predicted entities with the positions in the original sentence
    clusters = defaultdict(list)

    for ent_id, ments in unmatched_clusters.items():
        for start, end in ments:
            new_start = None  # start in the original sequence
            new_end = None  # end in the original sequence

            for j in range(start, end + 1):
                if j in matching:
                    if new_start is None:
                        new_start = matching[j]

                    new_end = matching[j]

            if new_start is not None:
                # predict entity
                clusters[ent_id].append((
                    subtoken_map[new_start], subtoken_map[new_end]))
    predict_clusters = [list(set(v)) for k, v in clusters.items() if
                        len(set(v)) >= thred]
    return predict_clusters, aligned_input_ids, aligned_rec_ids


def parse_nonint_output_tokens(input_ids, output_ids,
                               special_ids, subtoken_map,
                               tokenizer,
                               add_mention_end,
                               thred):
    rec_ids, new_id = [], -1
    ment_start_stack = []
    unmatched_clusters = defaultdict(list)
    new_input_ids = [t for t in input_ids if t != tokenizer.pad_token_id]
    token_mentions = []
    for i in range(len(output_ids)):
        if output_ids[i] == tokenizer.pad_token_id:
            break
        if output_ids[i] == special_ids['mention_start']:
            new_id += 1
            ment_start_stack.append(new_id)
        elif output_ids[i] == special_ids['mention_end']:
            assert add_mention_end
            assert output_ids[i + 1] in special_ids['cluster_ids_to_num']
            cid = special_ids['cluster_ids_to_num'][output_ids[i + 1]]
            if len(ment_start_stack) > 0:
                item = ment_start_stack.pop()
                unmatched_clusters[cid].append((item, new_id))
        elif output_ids[i] in special_ids['cluster_ids_to_num']:
            if not add_mention_end:
                cid = special_ids['cluster_ids_to_num'][output_ids[i]]
                if len(ment_start_stack) > 0:
                    item = ment_start_stack.pop()
                    unmatched_clusters[cid].append((item, new_id))
        else:
            new_id += 1
            rec_ids.append(output_ids[i])
        if output_ids[i] == special_ids['mention_start']:
            new_id -= 1
    # Needleman-Wunsch text alignment algorithm
    wrong_reconstruction = (rec_ids != new_input_ids)
    # thred = 1 if allow_singletons else 2
    if wrong_reconstruction:
        print(f'new input ids {new_input_ids}')
        print(f'reconstructed ids {rec_ids}')
        print(f'out ids {output_ids}')
        print('wrong reconstruction! please debug')
        matching = global_align(new_input_ids, rec_ids)

        # update predicted entities with the positions in the original sentence
        clusters = defaultdict(list)

        for ent_id, ments in unmatched_clusters.items():
            for start, end in ments:
                new_start = None  # start in the original sequence
                new_end = None  # end in the original sequence

                for j in range(start, end + 1):
                    if j in matching:
                        if new_start is None:
                            new_start = matching[j]

                        new_end = matching[j]

                if new_start is not None:
                    # predict entity
                    clusters[ent_id].append((
                        subtoken_map[new_start], subtoken_map[new_end]))
                    token_mentions.append((new_start, new_end))
        predict_clusters = [list(set(v)) for k, v in clusters.items() if
                            len(set(v)) >= thred]
        token_mentions = list(set(token_mentions))
    else:
        clusters = [[(subtoken_map[m[0]], subtoken_map[m[1]]) for m in v] for v
                    in
                    unmatched_clusters.values()]
        predict_clusters = [list(set(v)) for v in clusters if len(set(v)) >=
                            thred]
        token_mentions = [(m[0], m[1]) for v in unmatched_clusters.values()
                          for m in v]
        token_mentions = list(set(token_mentions))
    return predict_clusters, token_mentions, output_ids


@dataclass
class ConstrainedDataCollator:
    """
    Data collator that will dynamically pad the inputs received, as well as the labels.

    Args:
        tokenizer ([`PreTrainedTokenizer`] or [`PreTrainedTokenizerFast`]):
            The tokenizer used for encoding the data.
        model ([`PreTrainedModel`]):
            The model that is being trained. If set and has the *prepare_decoder_input_ids_from_labels*, use it to
            prepare the *decoder_input_ids*

            This is useful when using *label_smoothing* to avoid calculating loss twice.
        padding (`bool`, `str` or [`~utils.PaddingStrategy`], *optional*, defaults to `True`):
            Select a strategy to pad the returned sequences (according to the model's padding side and padding index)
            among:

            - `True` or `'longest'`: Pad to the longest sequence in the batch (or no padding if only a single sequence
              is provided).
            - `'max_length'`: Pad to a maximum length specified with the argument `max_length` or to the maximum
              acceptable input length for the model if that argument is not provided.
            - `False` or `'do_not_pad'` (default): No padding (i.e., can output a batch with sequences of different
              lengths).
        max_length (`int`, *optional*):
            Maximum length of the returned list and optionally padding length (see above).
        pad_to_multiple_of (`int`, *optional*):
            If set will pad the sequence to a multiple of the provided value.

            This is especially useful to enable the use of Tensor Cores on NVIDIA hardware with compute capability >=
            7.5 (Volta).
        label_pad_token_id (`int`, *optional*, defaults to -100):
            The id to use when padding the labels (-100 will be automatically ignored by PyTorch loss functions).
        return_tensors (`str`):
            The type of Tensor to return. Allowable values are "np", "pt" and "tf".
    """

    tokenizer: PreTrainedTokenizerBase
    model: Optional[Any] = None
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    label_pad_token_id: int = -100
    return_tensors: str = "pt"

    def __call__(self, features, return_tensors=None):
        import numpy as np

        if return_tensors is None:
            return_tensors = self.return_tensors
        labels = [feature["labels"] for
                  feature in features] if "labels" in features[
            0].keys() else None
        decoder_labels = [feature["decoder_labels"] for
                          feature in features] if "decoder_labels" in features[
            0].keys() else None
        # We have to pad the labels before calling `tokenizer.pad` as this method won't pad them and needs them of the
        # same length to return tensors.
        if labels is not None:
            assert decoder_labels is not None
            max_label_length = max(len(l) for l in labels)
            if self.pad_to_multiple_of is not None:
                max_label_length = (
                        (max_label_length + self.pad_to_multiple_of - 1)
                        // self.pad_to_multiple_of
                        * self.pad_to_multiple_of
                )

            padding_side = self.tokenizer.padding_side
            for feature in features:
                remainder = [self.label_pad_token_id] * (
                        max_label_length - len(feature["labels"]))
                if isinstance(feature["labels"], list):
                    feature["labels"] = (
                        feature[
                            "labels"] + remainder if padding_side == "right"
                        else remainder + feature["labels"]
                    )
                    feature["decoder_labels"] = (
                        feature[
                            "decoder_labels"] + remainder if padding_side ==
                                                             "right"
                        else remainder + feature["decoder_labels"]
                    )
                elif padding_side == "right":
                    feature["labels"] = np.concatenate(
                        [feature["labels"], remainder]).astype(np.int64)
                    feature["decoder_labels"] = np.concatenate(
                        [feature["decoder_labels"], remainder]).astype(np.int64)
                else:
                    feature["labels"] = np.concatenate(
                        [remainder, feature["labels"]]).astype(np.int64)
                    feature["decoder_labels"] = np.concatenate(
                        [remainder, feature["decoder_labels"]]).astype(np.int64)

        features = self.tokenizer.pad(
            features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=return_tensors,
        )

        # prepare decoder_input_ids
        if (
                labels is not None
                and self.model is not None
                and hasattr(self.model, "prepare_decoder_input_ids_from_labels")
        ):
            decoder_input_ids = self.model.prepare_decoder_input_ids_from_labels(
                labels=features["decoder_labels"])
            features["decoder_input_ids"] = decoder_input_ids
            if self.model.is_input_feed:
                decoder_input_actions = \
                    self.model.prepare_decoder_input_ids_from_labels(
                        labels=features["labels"])
                features["decoder_input_actions"] = decoder_input_actions
        del features["decoder_labels"]
        return features
