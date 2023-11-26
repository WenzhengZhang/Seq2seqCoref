from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import re
import os
import sys
import json
import copy
import collections
import logging
from typing import Optional, Tuple, Any, Dict, Iterable, List
from collections import defaultdict
import numpy as np
import argparse

logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
    datefmt='%m/%d/%Y %H:%M:%S',
    level=logging.INFO
)
logger = logging.getLogger(__file__)

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__),
                                                '../')))
import utils
import conll
from constants import mark_sent_tokenizer as tokenizer
from constants import SPEAKER_START, SPEAKER_END, SENTENCE_START, \
    SENTENCE_END, MENTION_START, MENTION_END, SEP_TOKEN, COPY
from constants import MARK_SPECIAL_IDS as SPECIAL_IDS


class DocumentState(object):
    def __init__(self, key):
        self.doc_key = key
        self.sentence_end = []
        self.token_end = []
        self.tokens = []
        self.subtokens = []
        self.info = []
        self.segments = []
        self.subtoken_map = []
        self.segment_subtoken_map = []
        self.sentence_map = []
        self.segment_sentence_map = []
        self.pronouns = []
        self.clusters = collections.defaultdict(list)
        self.coref_stacks = collections.defaultdict(list)
        self.segment_info = []
        self.offsets = []

    def finalize(self):
        # populate clusters
        # fix the bug here after finish the processor experiments
        mention_to_seg_id = defaultdict(list)
        word_clusters = defaultdict(list)
        word_coref_stack = defaultdict(list)
        word_ment_to_seg_id = defaultdict(list)
        seg_idx_stack = defaultdict(list)
        for seg_idx, segment in enumerate(self.segment_info):
            # keeping all segments
            # modify mention_to_seg_index
            seg_subtoken_map = self.segment_subtoken_map[seg_idx]
            seg_offset = self.offsets[seg_idx]
            first_subtoken_index = seg_offset - 1
            for i, tok_info in enumerate(segment):
                first_subtoken_index += 1
                coref = tok_info[-2] if tok_info is not None else '-'
                if coref != "-":
                    last_subtoken_index = first_subtoken_index + \
                                          tok_info[-1] - 1
                    for part in coref.split("|"):
                        if part[0] == "(":
                            if part[-1] == ")":
                                cluster_id = int(part[1:-1])
                                ment = (first_subtoken_index,
                                        last_subtoken_index)
                                self.clusters[cluster_id].append(ment)
                                word_clusters[cluster_id].append(
                                    (seg_subtoken_map[ment[0] - seg_offset],
                                     seg_subtoken_map[ment[1] - seg_offset]))
                                mention_to_seg_id[
                                    ment].append(seg_idx)
                                word_ment_to_seg_id[
                                    (seg_subtoken_map[ment[0] - seg_offset],
                                     seg_subtoken_map[ment[1] - seg_offset])
                                ].append(seg_idx)

                            else:
                                cluster_id = int(part[1:])
                                if (first_subtoken_index not in
                                    self.coref_stacks[cluster_id]) or (
                                        self.coref_stacks[cluster_id][-1] ==
                                        first_subtoken_index and
                                        seg_idx_stack[cluster_id][-1] ==
                                        seg_idx):
                                    self.coref_stacks[cluster_id].append(
                                        first_subtoken_index)
                                    word_coref_stack[cluster_id].append(
                                        seg_subtoken_map[
                                            first_subtoken_index - seg_offset]
                                    )
                                    seg_idx_stack[cluster_id].append(seg_idx)
                                elif self.coref_stacks[cluster_id][-1] == \
                                        first_subtoken_index and \
                                        seg_idx_stack[cluster_id][
                                            -1] != seg_idx:
                                    seg_idx_stack[cluster_id][-1] = seg_idx
                        else:
                            cluster_id = int(part[:-1])
                            if len(self.coref_stacks[cluster_id]) > 0 and \
                                    self.coref_stacks[cluster_id][
                                        -1] < last_subtoken_index:
                                start = self.coref_stacks[cluster_id].pop()
                                word_start = word_coref_stack[cluster_id].pop()
                                m_seg_id = seg_idx_stack[cluster_id].pop()
                                ment = (start, last_subtoken_index)
                                self.clusters[cluster_id].append(ment)
                                word_clusters[cluster_id].append(
                                    (word_start,
                                     seg_subtoken_map[ment[1] - seg_offset])
                                )
                                if seg_idx == m_seg_id:
                                    mention_to_seg_id[ment].append(seg_idx)
                                    word_ment_to_seg_id[
                                        (word_start,
                                         seg_subtoken_map[ment[1] - seg_offset])
                                    ].append(seg_idx)

        # merge clusters
        merged_clusters = []
        for c1 in self.clusters.values():
            existing = None
            for m in c1:
                for c2 in merged_clusters:
                    if m in c2:
                        existing = c2
                        break
                if existing is not None:
                    break
            if existing is not None:
                logger.info("Merging clusters (shouldn't happen very often.)")
                existing.update(c1)
            else:
                merged_clusters.append(set(c1))

        # merge gold clusters
        gold_clusters = []
        for c1 in word_clusters.values():
            gold_exist = None
            for m in c1:
                for c2 in gold_clusters:
                    if m in c2:
                        gold_exist = c2
                        break
                if gold_exist is not None:
                    break
            if gold_exist is not None:
                logger.info(
                    "Merging gold clusters (shouldn't happen very often.)")
                gold_exist.update(c1)
            else:
                gold_clusters.append(set(c1))
        gold_clusters = [list(c) for c in gold_clusters]
        merged_clusters = [list(c) for c in merged_clusters]
        # print(merged_clusters)
        # print(gold_clusters)
        all_seg_clusters = get_seg_clusters(merged_clusters,
                                            mention_to_seg_id,
                                            len(self.segment_info))
        all_word_seg_clusters = get_seg_clusters(gold_clusters,
                                                 word_ment_to_seg_id,
                                                 len(self.segment_info))
        # print(len(self.segment_info))
        # print(len(self.segment_info[0]))
        # print(len(self.segment_info[1]))
        # print(all_seg_clusters)
        cluster_indices = get_mention_to_cid(all_seg_clusters)
        docs = []
        num_words = len(utils.flat_lists(self.segments))
        num_segments = len(self.segments)

        subtoken_map = self.segment_subtoken_map
        assert num_words == len(utils.flat_lists(self.segment_subtoken_map))

        sentence_map = self.segment_sentence_map
        assert num_words == len(utils.flat_lists(sentence_map))

        # all_mentions = utils.flat_lists(merged_clusters)
        all_mentions = list(mention_to_seg_id.keys())
        assert len(all_mentions) == len(set(all_mentions))
        sentences = self.segments

        # inserting <m> and </m> into target sequences for all mentions
        target_sentences = m_star_target_sequences(
            all_mentions, self.segments,
            MENTION_START, MENTION_END, SEP_TOKEN,
            mention_to_seg_id, cluster_indices,
            self.offsets
        )
        target_short_seqs = []
        for target_seq in target_sentences:
            target_short = trim_target_sequence(target_seq,
                                                MENTION_START, MENTION_END,
                                                SENTENCE_START, SENTENCE_END
                                                )
            target_short_seqs.append(target_short)

        target_maps = get_target_map(target_sentences,
                                     tokenizer.tokenize(MENTION_END)[0],
                                     tokenizer.tokenize(SEP_TOKEN)[0])
        target_tags = get_target_tags(target_sentences, target_maps,
                                      tokenizer.tokenize(MENTION_START)[0],
                                      tokenizer.tokenize(SEP_TOKEN)[0],
                                      tokenizer.tokenize(COPY)[0])
        # add gold clusters info into docs
        for i in range(num_segments):
            docs.append({
                "doc_key": f'{self.doc_key}_{i}',
                "offset": self.offsets[i],
                "sentence": sentences[i],
                "target_sentence": target_sentences[i],
                # "target_map": target_maps[i],
                "target_action": target_tags[i],
                "target_short_sentence": target_short_seqs[i],
                "subtoken_map": subtoken_map[i],
                "gold_clusters": gold_clusters,
                "seg_clusters": all_word_seg_clusters[i],
                "gold_token_clusters": merged_clusters
            })

        return docs


def get_seg_clusters(merged_clusters,
                     mention_to_seg_id,
                     num_segs):
    all_seg_clusters = []
    for seg_id in range(num_segs):
        seg_clusters = []
        for c in merged_clusters:
            seg_cluster = []
            for m in c:
                m_sids = mention_to_seg_id[tuple(m)]
                if seg_id in m_sids:
                    seg_cluster.append(m)
            if len(seg_cluster) >= 1:
                seg_clusters.append(seg_cluster)
        all_seg_clusters.append(seg_clusters)
    return all_seg_clusters


def get_mention_to_cid(all_seg_clusters):
    # k: old group idx  v: sorted group idx

    def get_seg_mention_to_gid(groups):
        mention_to_gid = {}
        first_mentions = [min(g, key=lambda m: (m[1], -m[0])) for g in groups]
        assert len(first_mentions) == len(groups)
        sorted_ids = sorted(list(range(len(first_mentions))),
                            key=lambda k: (first_mentions[k][1],
                                           -first_mentions[k][0]))
        gid_map = {j: i for i, j in enumerate(sorted_ids)}
        for i, g in enumerate(groups):
            gid = gid_map[i]
            for m in g:
                mention_to_gid[tuple(m)] = gid
        return mention_to_gid

    all_seg_ment2cid = []
    for seg_clusters in all_seg_clusters:
        ment2cid = get_seg_mention_to_gid(seg_clusters)
        all_seg_ment2cid.append(ment2cid)
    return all_seg_ment2cid


def m_star_target_sequences(
        mentions: List[Tuple[int, int]],
        sequences: List[List[str]],
        m_special_start: str,
        m_special_end: str,
        m_sep: str,
        mention_to_seg_id: Dict[tuple, list],
        cluster_indices: List[Dict],
        offsets: List
):
    """
        Get a sequence of target sentences with <m> and <\m> inserted.
        mentions: list of mentions, e.g. [(0, 0), (2, 3), (4, 4)] format: [start, end] (inclusive)
        sequences: list of sequences, e.g. [['I', 'have', 'a', 'cat'], ['I', 'have', 'a', 'dog']]
        m_special_start: special token for starting bracket
        m_special_end: special token for ending bracket
        mention_to_seg_id: dict, mapping mention to its segment id
    """
    m_startings, m_endings = zip(*mentions) if len(mentions) > 0 else ([], [])
    all_m_cids = []
    all_m_sids = []
    for m in mentions:
        m_sids = mention_to_seg_id[tuple(m)]
        m_seg_cids = [cluster_indices[m_sid] for m_sid in m_sids]
        m_cids = [m_seg_cid[tuple(m)] for m_seg_cid in m_seg_cids]
        all_m_cids.append(m_cids)
        all_m_sids.append(m_sids)
    # later segment comes first
    end_pos = [(m_sid, x + 1, -1,
                -m_startings[i],
                m_cid) for i, x in enumerate(
        m_endings) for m_sid, m_cid in zip(all_m_sids[i], all_m_cids[i])]
    start_pos = [(m_sid, x, 1, -m_endings[i],
                  m_cid) for i, x in enumerate(
        m_startings) for m_sid, m_cid in zip(all_m_sids[i], all_m_cids[i])]
    # insert from right to left, so that the calculated positions are not changed
    sorted_pos = sorted(end_pos + start_pos, reverse=True)
    target_sequences = copy.deepcopy(sequences)
    # offset of each segment
    # prev_loc, prev_token, prev_seg_idx = -1, None, -1
    for x in sorted_pos:
        seg_idx = x[0]
        offset = offsets[seg_idx]
        if x[2] > 0:
            # start
            assert x[2] == 1
            target_sequences[seg_idx].insert(x[1] - offset, m_special_start)
        else:
            # end
            end_inserts = tokenizer.tokenize(
                m_sep) + tokenizer.tokenize(str(x[-1])) + [m_special_end]
            for e in reversed(end_inserts):
                target_sequences[seg_idx].insert(x[1] - offset, e)
    return target_sequences


def trim_target_sequence(target_seq,
                         m_special_start,
                         m_special_end,
                         sentence_start,
                         sentence_end
                         ):
    out_seq = []
    ment_stack = []
    # use sentence map to add periods
    for idx, s in enumerate(target_seq):
        if s == m_special_start:
            out_seq.append(s)
            ment_stack.append(idx)
        elif len(ment_stack) > 0:
            out_seq.append(s)
            if s == m_special_end:
                ment_stack.pop()
        elif s == sentence_start or s == sentence_end:
            out_seq.append(s)
    out_seq.append('</s>')
    return out_seq


def get_target_tags(target_sequences,
                    target_maps,
                    m_special_start,
                    m_sep, m_copy):
    # 1 for inside entity 0 for outside entity
    target_tags = []
    for target_sequence, target_map in zip(target_sequences, target_maps):
        target_tag = np.array(target_sequence)
        tgt_map = np.array(target_map, dtype=bool)
        tag_map = (target_tag != m_special_start) & (
                target_tag != m_sep) & (
                      ~tgt_map)
        target_tag[tag_map] = m_copy
        target_tags.append(target_tag.tolist())
    return target_tags


def get_target_map(target_sequences, m_special_end, m_sep):
    # 1 for inside entity 0 for outside entity
    target_maps = []
    for target_sequence in target_sequences:
        target_map = []
        status = 'o'
        for t in target_sequence:
            if status == 'o':
                target_map.append(0)
            else:
                target_map.append(1)
            if t == m_sep:
                status = 'i'
            elif t == m_special_end:
                status = 'o'
        assert len(target_map) == len(target_sequence)
        target_maps.append(target_map)
    return target_maps


def normalize_word(word, language):
    br_dict = {"-LRB-": "(", "-RRB-": ")", "-LSB-": "[", "-RSB-": "]"}

    if language == "arabic":
        word = word[:word.find("#")]

    if word in br_dict:
        word = br_dict[word]
        return word
    elif word == "/." or word == "/?":
        return word[1:]
    elif word == "''" or word == "``":  # <unk> otherwise
        return "\""
    elif word == "`":  # <unk> otherwise
        return "\'"
    else:
        return word


# first try to satisfy constraints1, and if not possible, constraints2.
def split_into_segments(
        document_state,
        max_segment_len,
        stride,
        constraints1,
        constraints2,
        is_train
):
    # introduce stride
    # get offset info
    current = 0
    offsets = []
    if not is_train and len(document_state.subtokens) < max_segment_len:
        stride = len(document_state.subtokens)
    while current < len(document_state.subtokens):
        offsets.append(current)
        end = min(current + max_segment_len - 1 - 1,
                  len(document_state.subtokens) - 1)

        while end >= current and not constraints1[end]:
            end -= 1

        if end < current:
            end = min(current + max_segment_len - 1 - 1,
                      len(document_state.subtokens) - 1)
            while end >= current and not constraints2[end]:
                end -= 1
            if end < current:
                raise Exception("Can't find valid segment")

        document_state.segments.append(
            document_state.subtokens[current:end + 1] + [
                '</s>'])
        sent_map = document_state.sentence_map[current:end + 1]
        document_state.segment_sentence_map.append(sent_map + [sent_map[-1]])
        subtoken_map = document_state.subtoken_map[current:end + 1]
        document_state.segment_subtoken_map.append(
            subtoken_map + [subtoken_map[-1]])
        document_state.segment_info.append(
            document_state.info[current:end + 1] + [None])
        # current = end + 1
        next_cur = min(current + stride, len(document_state.subtokens))
        while next_cur > current and not constraints1[next_cur - 1]:
            next_cur -= 1
        if next_cur < current + 1:
            next_cur = min(current + stride, len(document_state.subtokens))
            while next_cur > current and not constraints2[next_cur - 1]:
                next_cur -= 1
            if next_cur < current + 1:
                raise Exception("Can't find valid stride")
        current = next_cur
    document_state.offsets = offsets
    return


def get_doc_sentence_map(sentence_end):
    current = 0
    sent_map = []
    for i, s in enumerate(sentence_end):
        sent_map.append(current)
        current += int(s)
    return sent_map


def get_document(
        document_lines, tokenizer, language, segment_len, stride, is_train
):
    document_state = DocumentState(document_lines[0])
    word_idx = -1

    current_speaker = None
    after_hyphen = False
    doc_lines = document_lines[1]
    last_sent_idx = None
    sent_idx = 0
    for line in doc_lines:
        row = line.split()
        sentence_end = len(row) == 0
        if not sentence_end:
            assert len(row) >= 12
            speaker_orthography = row[9].replace("_", " ").replace("#",
                                                                   " ").strip()
            if last_sent_idx is None or sent_idx != last_sent_idx:
                word_idx += 1
                document_state.token_end.append(True)
                document_state.subtokens.append(SENTENCE_START)
                info = None
                document_state.info.append(info)
                document_state.sentence_end.append(False)
                document_state.subtoken_map.append(word_idx)
                last_sent_idx = sent_idx

            if current_speaker is None or current_speaker != speaker_orthography:
                # insert speaker
                word_idx += 1
                current_speaker = speaker_orthography
                speaker_text = tokenizer.tokenize(current_speaker)
                document_state.tokens.append(current_speaker)
                if len(speaker_text) == 0:
                    speaker_token_end = [True, True]
                else:
                    speaker_token_end = [True] + [False] * (len(
                        speaker_text) - 1) + [True, True]
                assert len(speaker_token_end) == len([SPEAKER_START] +
                                                     speaker_text + [
                                                         SPEAKER_END])
                document_state.token_end += speaker_token_end
                for sidx, subtoken in enumerate(
                        [SPEAKER_START] + speaker_text + [SPEAKER_END]):
                    document_state.subtokens.append(subtoken)
                    info = None
                    document_state.info.append(info)
                    document_state.sentence_end.append(False)
                    document_state.subtoken_map.append(word_idx)

            word_idx += 1
            word = normalize_word(row[3], language)

            if is_punctuation(word):
                subtokens = tokenizer.tokenize(word)[1:]  # skipping '_'
            elif after_hyphen:
                subtokens = tokenizer.tokenize("-" + word)  # skipping '_'
                if subtokens[1] == "-":
                    subtokens = subtokens[2:]
                else:
                    subtokens = subtokens[1:]
                after_hyphen = False
            else:
                subtokens = tokenizer.tokenize(word)

            if row[4] == "HYPH":
                after_hyphen = True

            document_state.tokens.append(word)
            if len(subtokens) > 0:
                document_state.token_end += [False] * (len(subtokens) - 1) + [
                    True]

            for sidx, subtoken in enumerate(subtokens):
                document_state.subtokens.append(subtoken)
                info = None if sidx != 0 else (row + [len(subtokens)])
                document_state.info.append(info)
                document_state.sentence_end.append(False)
                document_state.subtoken_map.append(word_idx)
        else:
            word_idx += 1
            document_state.token_end.append(True)
            document_state.subtokens.append(SENTENCE_END)
            document_state.info.append(None)
            document_state.sentence_end.append(True)
            document_state.subtoken_map.append(word_idx)
            sent_idx += 1
            # document_state.sentence_end[-1] = True

    constraints1 = (
        document_state.sentence_end
        if language != "arabic"
        else document_state.token_end
    )
    document_state.sentence_map = get_doc_sentence_map(
        document_state.sentence_end)
    split_into_segments(
        document_state, segment_len, stride, constraints1,
        document_state.token_end, is_train
    )

    stats[f"max_seg_len"] = max(
        stats["max_seg_len"], max([len(s) for s in document_state.segments])
    )
    stats[f"max_num_seg"] = max(
        len(document_state.segments), stats[f"max_num_seg"]
    )
    document = document_state.finalize()
    return document


def is_punctuation(c):
    if (
            c in {".", ",", "?", "!", ";",
                  ":", "'s", "'m", "'ve", "n't", "'ll",
                  ")", "]", "}", "-"}
    ):
        return True
    return False


def is_special(c):
    if (
            c in {"<pad>", "</s>", "<unk>"}
    ):
        return True
    return False


def accumu(lis):
    total = 0
    for x in lis:
        yield total
        total += x


def minimize_partition(
        name, language, extension, stats, tokenizer, seg_len, stride,
        input_dir,
        output_dir, is_train
):
    input_path = "{}/{}.{}.{}".format(input_dir, name, language, extension)
    output_path = "{}/{}.t5-small.{}.{}.jsonlines".format(output_dir, name,
                                                          language, seg_len)

    count = 0
    logger.info("Minimizing {}".format(input_path))
    documents = []
    with open(input_path, "r") as input_file:
        for line in input_file.readlines():
            begin_document_match = re.match(conll.BEGIN_DOCUMENT_REGEX, line)
            if begin_document_match:
                doc_key = conll.get_doc_key(
                    begin_document_match.group(1), begin_document_match.group(2)
                )
                documents.append((doc_key, []))
            elif line.startswith("#end document"):
                continue
            else:
                documents[-1][1].append(line)

    datasets, max_target_len = [], 0
    max_input_len = 0
    max_num_clusters = 0
    max_seg_clusters = 0
    for document_lines in documents:
        # max_input_len = max(max_input_len,
        #                     len([x for x in document_lines[1] if
        #                          len(x) > 2]))
        document = get_document(document_lines, tokenizer, language, seg_len,
                                stride, is_train)
        for doc in document:
            max_input_len = max(
                [max_input_len] + [len(doc['sentence'])])
            max_target_len = max(
                [max_target_len] + [len(doc['target_sentence'])])
            max_num_clusters = max(
                [max_num_clusters] + [len(doc['gold_clusters'])])
            max_seg_clusters = max(
                [max_num_clusters] + [len(doc['seg_clusters'])])
            datasets.append(doc)
            count += 1
    with open(output_path, 'w') as f:
        for d in datasets:
            f.write('%s\n' % json.dumps(d))
    # json.dump(datasets, open(output_path, "w"))
    logger.info(
        f"Maximum input sequence length: {max_input_len}, Maximum target sequence length: {max_target_len}")
    logger.info(f'Maximum num gold clusters: {max_num_clusters}')
    logger.info(f'Maximum num segment clusters: {max_seg_clusters}')
    logger.info("Wrote {} documents to {}".format(count, output_path))


def minimize_language(language, stats, seg_len, stride,
                      input_dir, output_dir):
    minimize_partition("dev", language, "v4_gold_conll", stats,
                       tokenizer, seg_len, stride, input_dir, output_dir,
                       False)
    minimize_partition("train", language, "v4_gold_conll", stats,
                       tokenizer, seg_len, stride, input_dir, output_dir,
                       True)
    minimize_partition("test", language, "v4_gold_conll", stats,
                       tokenizer, seg_len, stride, input_dir, output_dir,
                       False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', type=str, help='input directory')
    parser.add_argument('--output_dir', type=str, help='output directory')
    args = parser.parse_args()
    input_dir = args.input_dir
    output_dir = args.output_dir

    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)
    # stride = 1024
    # stride = 512
    # stride = 1536
    for seg_len in [4096, 2048, 1536]:
        stats = collections.defaultdict(int)
        stride = seg_len // 2
        minimize_language("english", stats, seg_len, stride, input_dir,
                          output_dir)

        logger.info("Dataset stats:")
        for k, v in stats.items():
            logger.info("{} = {}".format(k, v))
