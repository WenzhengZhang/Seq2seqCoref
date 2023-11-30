from transformers import LogitsProcessor
import torch


class ShortSeqProcessor(LogitsProcessor):

    def __init__(self, orig_inputs, special_ids):
        self.orig_inputs = orig_inputs
        self.sentence_start = special_ids['sentence_start']
        self.sentence_end = special_ids['sentence_end']
        self.mention_start = special_ids['mention_start']
        self.mention_end = special_ids['mention_end']
        self.sep = special_ids['sep']
        self.ent_ids = special_ids['integers'] + [special_ids['mention_end']]
        self.eos_id = special_ids['eos']
        self.sentence_mask = self.get_sentence_mask(orig_inputs)

    def get_sentence_mask(self, orig_inputs: torch.Tensor):
        # index from 1 instead of 0
        return (orig_inputs == self.sentence_start).cumsum(-1)

    def __call__(self, input_ids: torch.LongTensor,
                 scores: torch.FloatTensor) -> torch.FloatTensor:
        is_sent_start = (input_ids == self.sentence_start)
        is_sent_end = (input_ids == self.sentence_end)
        sent_idx = is_sent_start.sum(-1, keepdim=True)
        unclose_sent = (sent_idx.sum(-1) - is_sent_end.sum(-1)) > 0
        close_sent = (~unclose_sent)
        is_sep = (input_ids == self.sep)
        is_end = (input_ids == self.mention_end)
        is_start = (input_ids == self.mention_start)
        is_ent = (is_sep.cumsum(-1) - is_end.cumsum(-1)).bool()
        unclose_ent = (is_ent[:, -1] & unclose_sent)
        unclose_ment = (is_start.sum(-1) - is_sep.sum(-1)) > 0
        close_ent = (~unclose_ent)
        unclose_ment = (close_ent & unclose_ment & unclose_sent)
        masks = torch.ones_like(scores, dtype=torch.bool)
        masks[unclose_sent, self.sentence_end] = False
        masks[close_sent, self.sentence_start] = False
        assert scores.size(0) % self.orig_inputs.size(0) == 0
        num_beams = scores.size(0) // self.orig_inputs.size(0)
        # repeat over beams
        orig_ids = self.orig_inputs.repeat_interleave(num_beams, 0)
        sent_mask = self.sentence_mask.repeat_interleave(num_beams, 0)
        cur_sent_mask = (sent_mask != sent_idx)
        sent_ids = orig_ids.masked_fill(cur_sent_mask, self.sentence_end)
        masks[unclose_sent] = masks[unclose_sent].scatter(1, sent_ids[
            unclose_sent], False)
        masks[unclose_sent, self.sentence_start] = True
        masks[unclose_ent, torch.tensor(self.ent_ids).unsqueeze(1)] = False
        masks[close_ent, self.mention_start] = False
        masks[unclose_ment, self.sep] = False
        is_eos = (close_sent & (sent_idx.sum(-1) == sent_mask[:, -1]))
        masks[is_eos] = True
        masks[is_eos, self.eos_id] = False
        scores.masked_fill_(masks, -float('inf'))
        return scores


class IntProcessor(LogitsProcessor):

    def __init__(self, orig_inputs, special_ids, seq2seq_type):
        """

        :param orig_inputs: original input_ids
        :param special_ids: dict with keys:[mention_start, mention_end, sep,
        integers]
        """
        self.orig_inputs = orig_inputs
        self.seq2seq_type = seq2seq_type
        self.special_ids = special_ids
        self.mention_start = special_ids['mention_start']
        self.mention_end = special_ids['mention_end']
        self.sep = special_ids['sep']
        self.ent_ids = special_ids['integers'] + [special_ids['mention_end']]
        self.specials = [self.mention_start, self.sep] + self.ent_ids
        if self.seq2seq_type == 'action' or self.seq2seq_type == 'tagging' or \
                self.seq2seq_type == 'input_feed':
            self.copy_id = special_ids['copy']
            self.specials.append(self.copy_id)
        self.eos_id = special_ids['eos']

    def __call__(self, input_ids: torch.LongTensor,
                 scores: torch.FloatTensor) -> torch.FloatTensor:
        """

        :param input_ids: BC x l
        :param scores:  BC x V
        :return:
        """
        # input_ids : B x L
        is_sep = (input_ids == self.sep)
        is_end = (input_ids == self.mention_end)
        is_start = (input_ids == self.mention_start)
        is_ent = (is_sep.cumsum(-1) - is_end.cumsum(-1)).bool()
        is_copy = ((~is_start) & (~is_ent) & (~is_end))
        unclose_ent = is_ent[:, -1]
        unclose_ment = (is_start.sum(-1) - is_sep.sum(-1)) > 0
        unclose_ment = ((~unclose_ent) & unclose_ment)
        # -1 for <pad> at begining
        num_copied = is_copy.sum(-1) - 1
        masks = torch.ones_like(scores, dtype=torch.bool)
        close_ent = (~unclose_ent)
        num_copied = num_copied.clamp(max=self.orig_inputs.size(1) - 1)
        # unclosed ent only allows to generate cluster ids or end mention id
        masks[unclose_ent, torch.tensor(self.ent_ids).unsqueeze(1)] = False
        masks[close_ent, self.mention_start] = False
        masks[unclose_ment, self.sep] = False
        # get next copy id
        assert scores.size(0) % self.orig_inputs.size(0) == 0
        num_beams = scores.size(0) // self.orig_inputs.size(0)
        # repeat over beams
        orig_ids = self.orig_inputs.repeat_interleave(num_beams, 0)
        next_ids = orig_ids[torch.arange(scores.size(0)), num_copied]
        if self.seq2seq_type == 'tagging':
            masks[close_ent, self.copy_id] = False
        else:
            if self.seq2seq_type == 'action' or self.seq2seq_type == \
                    'input_feed':
                scores[close_ent, next_ids[close_ent]] = scores[close_ent,
                                                                self.copy_id]
            masks[close_ent, next_ids[close_ent]] = False
        is_eos = (close_ent & (next_ids == self.eos_id))
        masks[is_eos, torch.tensor(self.specials).unsqueeze(1)] = True
        masks[is_eos, self.eos_id] = False
        scores.masked_fill_(masks, -float('inf'))
        return scores


class NonIntProcessor(LogitsProcessor):

    def __init__(self, orig_inputs, special_ids,
                 seq2seq_type,
                 add_mention_end):
        """

        :param orig_inputs: original input_ids
        :param special_ids: dict with keys:[mention_start, mention_end, sep,
        integers]
        :param add_mention_end: whether predict mention end before predict
        cluster ids
        """
        self.orig_inputs = orig_inputs
        self.special_ids = special_ids
        self.seq2seq_type = seq2seq_type
        self.mention_start = special_ids['mention_start']
        if add_mention_end:
            self.mention_end = special_ids['mention_end']
        else:
            self.mention_end = None
        self.cluster_ids = torch.tensor(special_ids['cluster_ids'],
                                        dtype=torch.long)
        self.cluster_new = special_ids['cluster_new']
        self.copy_id = special_ids['copy']
        self.eos_id = special_ids['eos']
        self.first_cluster_id = special_ids['cluster_ids'][0]
        self.last_cluster_id = special_ids['cluster_ids'][-1]
        self.add_mention_end = add_mention_end

    def __call__(self, input_ids: torch.LongTensor,
                 scores: torch.FloatTensor) -> torch.FloatTensor:
        """

        :param input_ids: BC x l
        :param scores:  BC x V
        :return:
        """
        # input_ids : B x L
        cluster_ids = self.cluster_ids.to(input_ids.device)
        range_indices = torch.arange(scores.size(0))
        is_not_cid = torch.isin(input_ids, cluster_ids, invert=True)
        is_not_start = (input_ids != self.mention_start)
        if self.add_mention_end:
            is_not_end = (input_ids != self.mention_end)
            unclosed_ent = (input_ids[:, -1] == self.mention_end)
            close_ent = (~unclosed_ent)
            is_copy = (is_not_start & is_not_end & is_not_cid)
        else:
            is_not_end = is_not_cid
            is_copy = (is_not_start & is_not_end)
        unclosed_ment = (is_not_start.sum(-1) - is_not_end.sum(-1)) < 0
        if self.add_mention_end:
            unclosed_ment = (close_ent & unclosed_ment)
        # -1 for <pad> at begining
        num_copied = is_copy.sum(-1) - 1
        masks = torch.ones_like(scores, dtype=torch.bool)
        num_copied = num_copied.clamp(max=self.orig_inputs.size(1) - 1)
        # unclosed ent only allows to generate cluster ids or end mention id
        # masks[:, self.specials] = False
        if self.add_mention_end:
            masks[close_ent, self.mention_start] = False
            masks[unclosed_ment, self.mention_end] = False
        else:
            masks[:, self.mention_start] = False
        # notice: make sure </mk> and </mk+1> are next to each other in vocab
        cluster_input_ids = input_ids.masked_fill(
            is_not_cid,
            self.first_cluster_id - 1)
        next_cids = cluster_input_ids.amax(-1) + 1
        if self.add_mention_end:
            has_prev_ends = (unclosed_ent & (next_cids > self.first_cluster_id))
            masks[unclosed_ent, next_cids[unclosed_ent]] = False
        else:
            has_prev_ends = (unclosed_ment & (next_cids >
                                              self.first_cluster_id))
            masks[unclosed_ment, next_cids[unclosed_ment]] = False

        masks[has_prev_ends] = masks[has_prev_ends].scatter(
            1, cluster_input_ids[has_prev_ends], False)
        masks[has_prev_ends, self.first_cluster_id - 1] = True
        # get next copy id
        assert scores.size(0) % self.orig_inputs.size(0) == 0
        num_beams = scores.size(0) // self.orig_inputs.size(0)
        # repeat over beams
        orig_ids = self.orig_inputs.repeat_interleave(num_beams, 0)
        next_ids = orig_ids[range_indices, num_copied]
        if self.add_mention_end:
            if self.seq2seq_type == 'action' or self.seq2seq_type == \
                    'input_feed':
                scores[close_ent, next_ids[close_ent]] = scores[close_ent,
                                                                self.copy_id]
            scores[unclosed_ent, next_cids[unclosed_ent]] = scores[
                unclosed_ent, self.cluster_new]
            masks[close_ent, next_ids[close_ent]] = False
        else:
            if self.seq2seq_type == 'action' or self.seq2seq_type == \
                    'input_feed':
                scores[range_indices, next_ids] = scores[:, self.copy_id]
            scores[unclosed_ment, next_cids[unclosed_ment]] = scores[
                unclosed_ment,
                self.cluster_new]
            masks[range_indices, next_ids] = False
        is_eos = (next_ids == self.eos_id)
        masks[is_eos] = True
        masks[is_eos, self.eos_id] = False
        scores.masked_fill_(masks, -float('inf'))
        return scores
