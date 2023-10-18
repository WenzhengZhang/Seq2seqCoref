import os
import json
import re
from collections import defaultdict
from metrics import CorefAllMetrics
from typing import Dict
from data import get_document_predicts, SPECIAL_IDS, parse_short_target_tokens
import argparse
from transformers import T5Tokenizer
from preprocess import SPEAKER_START, SPEAKER_END, MENTION_START, \
    MENTION_END, COPY
from preprocess_mark_sentence import SPECIAL_IDS as MARK_SPECIAL_IDS
from preprocess_mark_sentence import SENTENCE_START, SENTENCE_END


def load_data(data_dir, tokenizer):
    def load_split(split):
        max_len = 4096
        data_path = os.path.join(
            data_dir,
            f'{split}.t5-small.english.{max_len}.jsonlines')
        samples = []
        doc_labels = {}
        with open(data_path, 'r') as f:
            for line in f:
                item = json.loads(line)
                doc_key = item['doc_key']
                doc_id = re.sub(r'_\d+$', '', doc_key)
                target_seq = tokenizer.convert_tokens_to_ids(
                    item['target_short_sentence'])
                sample = {'doc_key': doc_key,
                          'sentence': tokenizer.convert_tokens_to_ids(
                              item['sentence']),
                          'target_seq': target_seq,
                          'subtoken_map': item['subtoken_map'],
                          'seg_clusters': [[tuple(m) for m in c] for c in item[
                              'seg_clusters'] if len(c) >= 2],
                          'offset': item['offset']
                          }
                doc_labels[doc_id] = [[tuple(m) for m in c] for c in item[
                    'gold_clusters']]
                samples.append(sample)
        return samples, doc_labels

    samples_dev, dev_labels = load_split('dev')
    samples_test, test_labels = load_split('test')
    return samples_dev, samples_test, dev_labels, test_labels


def oracle_align(doc_labels, samples, tokenizer, align_mode, mark_sentence) -> \
        Dict:
    documents_to_chunk_data = defaultdict(list)
    documents_to_chunk_gold = defaultdict(list)
    predictions = {}
    golds = {}
    last_doc_id = re.sub(r'_\d+$', '', samples[0]['doc_key'])
    for sample in samples:
        doc_key = sample['doc_key']
        doc_id = re.sub(r'_\d+$', '', doc_key)
        # require convert to ids first
        input_ids = sample['sentence']
        subtoken_map = sample['subtoken_map']
        offset = sample['offset']
        # remove bos
        predict_ids = sample['target_seq']
        gold_data = sample['seg_clusters']
        special_ids = MARK_SPECIAL_IDS if mark_sentence else SPECIAL_IDS
        pred_data, aligned_input_ids, aligned_pred_ids = \
            parse_short_target_tokens(input_ids, predict_ids,
                                      special_ids, subtoken_map,
                                      tokenizer,
                                      align_mode, 2, mark_sentence)
        # list of (m1,m2)

        documents_to_chunk_data[doc_id].extend(pred_data)
        documents_to_chunk_gold[doc_id].extend(gold_data)
        if doc_id != last_doc_id:
            predictions[last_doc_id] = get_document_predicts(
                documents_to_chunk_data[
                    last_doc_id])
            golds[last_doc_id] = get_document_predicts(
                documents_to_chunk_gold[
                    last_doc_id])
        last_doc_id = doc_id
    # final one
    predictions[last_doc_id] = get_document_predicts(
        documents_to_chunk_data[last_doc_id]
    )
    golds[last_doc_id] = get_document_predicts(
        documents_to_chunk_gold[last_doc_id]
    )
    # print(predictions)
    predictions_list = []
    labels_list = []
    golds_list = []
    for document_id, doc_label in doc_labels.items():
        predictions_list.append(predictions[document_id])
        labels_list.append(doc_label)
        golds_list.append(golds[document_id])

    metrics = CorefAllMetrics().get_all_metrics(labels_list,
                                                predictions_list)
    metrics_golds = CorefAllMetrics().get_all_metrics(golds_list,
                                                      predictions_list)
    label_results = {
        f'{metric_name}_{x}': v
        for metric_name, metric_values in metrics['micro'].items()
        for x, v in metric_values.items()
    }
    gold_results = {
        f'gold_{metric_name}_{x}': v
        for metric_name, metric_values in metrics_golds['micro'].items()
        for x, v in metric_values.items()
    }
    results = {**label_results, **gold_results}
    return results


def main(args):
    print('load data')
    tokenizer = T5Tokenizer.from_pretrained("t5-small", model_max_length=4096)
    tokenizer.add_tokens([SPEAKER_START, SPEAKER_END,
                          MENTION_START, MENTION_END, COPY])
    if args.mark_sentence:
        tokenizer.add_tokens([SENTENCE_START, SENTENCE_END])
    samples_dev, samples_test, dev_labels, test_labels = load_data(
        args.data_dir, tokenizer)
    print('check dev')
    results_dev = oracle_align(dev_labels, samples_dev, tokenizer,
                               args.align_mode, args.mark_sentence)
    print('dev results')
    print(results_dev)
    print('check test')
    results_test = oracle_align(test_labels, samples_test, tokenizer,
                                args.align_mode, args.mark_sentence)

    print('test results')
    print(results_test)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, help='data directory')
    parser.add_argument('--align_mode', type=str, default='l',
                        help='align mode')
    parser.add_argument('--mark_sentence', action='store_true')
    args = parser.parse_args()
    main(args)
