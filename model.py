from typing import Optional, Tuple, Dict, Union
import torch
import warnings
from copy import deepcopy
from torch.nn import CrossEntropyLoss
from transformers.modeling_outputs import (
    BaseModelOutput,
    Seq2SeqLMOutput
)

from transformers.models.t5.modeling_t5 import (
    T5_INPUTS_DOCSTRING
)
from transformers.models.t5.modeling_t5 import __HEAD_MASK_WARNING_MSG as \
    HEAD_MASK_WARNING_MSG
from transformers.models.t5.configuration_t5 import T5Config
from transformers.utils import (
    add_start_docstrings_to_model_forward,
    logging,
    replace_return_docstrings,
)
from transformers import T5ForConditionalGeneration

logger = logging.get_logger(__name__)

_CONFIG_FOR_DOC = "T5Config"
_TOKENIZER_FOR_DOC = "T5Tokenizer"
_CHECKPOINT_FOR_DOC = "t5-small"


class ConstrainedT5(T5ForConditionalGeneration):

    def __init__(self, config: T5Config, special_ids: Dict,
                 seq2seq_type: str, action_type: str,
                 add_mention_end: bool):
        super().__init__(config)
        self.mention_start = special_ids['mention_start']
        self.mention_end = special_ids.get('mention_end',None)
        self.eos_id = special_ids['eos']
        self.action_type = action_type
        self.add_mention_end = add_mention_end
        self.cluster_ids = None
        self.copy_id = special_ids['copy']
        self.seq2seq_type = seq2seq_type
        if action_type == 'integer':
            self.sep = special_ids['sep']
            self.ent_ids = special_ids['integers'] + [
                special_ids['mention_end']]
            self.specials = [self.mention_start, self.sep,
                             self.copy_id] + self.ent_ids
            # self.seq2seq_type = seq2seq_type
        else:
            self.cluster_new = special_ids['cluster_new']
            self.cluster_ids = special_ids['cluster_ids']
            self.eos_id = special_ids['eos']
            if self.add_mention_end:
                self.specials = [self.mention_start,
                                 self.mention_end,
                                 self.cluster_new,
                                 self.copy_id] + self.cluster_ids
            else:
                self.specials = [self.mention_start,
                                 self.cluster_new,
                                 self.copy_id] + self.cluster_ids
        if self.seq2seq_type == 'tagging':
            self.specials.append(self.eos_id)
        self.is_input_feed = (self.seq2seq_type == "input_feed")

    @add_start_docstrings_to_model_forward(T5_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=Seq2SeqLMOutput,
                               config_class=_CONFIG_FOR_DOC)
    def forward(
            self,
            input_ids: Optional[torch.LongTensor] = None,
            attention_mask: Optional[torch.FloatTensor] = None,
            decoder_input_ids: Optional[torch.LongTensor] = None,
            decoder_attention_mask: Optional[torch.BoolTensor] = None,
            head_mask: Optional[torch.FloatTensor] = None,
            decoder_head_mask: Optional[torch.FloatTensor] = None,
            cross_attn_head_mask: Optional[torch.Tensor] = None,
            encoder_outputs: Optional[Tuple[Tuple[torch.Tensor]]] = None,
            past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            decoder_inputs_embeds: Optional[torch.FloatTensor] = None,
            labels: Optional[torch.LongTensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
            decoder_input_actions: Optional[torch.LongTensor] = None,
            full_decoder_input_ids: Optional[torch.LongTensor] = None
    ) -> Union[Tuple[torch.FloatTensor], Seq2SeqLMOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[-100, 0, ...,
            config.vocab_size - 1]`. All labels set to `-100` are ignored (masked), the loss is only computed for
            labels in `[0, ..., config.vocab_size]`

        Returns:

        """
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        # FutureWarning: head_mask was separated into two input args - head_mask, decoder_head_mask
        if head_mask is not None and decoder_head_mask is None:
            if self.config.num_layers == self.config.num_decoder_layers:
                warnings.warn(HEAD_MASK_WARNING_MSG, FutureWarning)
                decoder_head_mask = head_mask

        # Encode if needed (training, first prediction pass)
        if encoder_outputs is None:
            # Convert encoder inputs in embeddings if needed
            encoder_outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                inputs_embeds=inputs_embeds,
                head_mask=head_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
        elif return_dict and not isinstance(encoder_outputs, BaseModelOutput):
            encoder_outputs = BaseModelOutput(
                last_hidden_state=encoder_outputs[0],
                hidden_states=encoder_outputs[1] if len(
                    encoder_outputs) > 1 else None,
                attentions=encoder_outputs[2] if len(
                    encoder_outputs) > 2 else None,
            )

        hidden_states = encoder_outputs[0]

        if self.model_parallel:
            torch.cuda.set_device(self.decoder.first_device)

        if labels is not None and decoder_input_ids is None and decoder_inputs_embeds is None:
            # get decoder inputs from shifting lm labels to the right
            decoder_input_ids = self._shift_right(labels)
        # Set device for model parallelism
        if self.is_input_feed and not self.training and decoder_input_actions is None:
            decoder_input_actions = self.input_to_actions(
                full_decoder_input_ids)
        if self.model_parallel:
            torch.cuda.set_device(self.decoder.first_device)
            hidden_states = hidden_states.to(self.decoder.first_device)
            if decoder_input_ids is not None:
                decoder_input_ids = decoder_input_ids.to(
                    self.decoder.first_device)
            if attention_mask is not None:
                attention_mask = attention_mask.to(self.decoder.first_device)
            if decoder_attention_mask is not None:
                decoder_attention_mask = decoder_attention_mask.to(
                    self.decoder.first_device)
            if self.is_input_feed and decoder_input_actions is \
                    not None:
                decoder_input_actions = decoder_input_actions.to(
                    self.decoder.first_device
                )
        if self.is_input_feed:
            decoder_token_embeds = self.decoder.embed_tokens(decoder_input_ids)
            if not self.training and past_key_values is not None:
                decoder_action_embeds = self.decoder.embed_tokens(
                    decoder_input_actions[:, -1:])
            else:
                decoder_action_embeds = self.decoder.embed_tokens(
                    decoder_input_actions)
            decoder_inputs_embeds = decoder_token_embeds / 2 + decoder_action_embeds / 2
        # Decode
        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids if not self.is_input_feed else None,
            attention_mask=decoder_attention_mask,
            inputs_embeds=decoder_inputs_embeds,
            past_key_values=past_key_values,
            encoder_hidden_states=hidden_states,
            encoder_attention_mask=attention_mask,
            head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = decoder_outputs[0]

        # Set device for model parallelism
        if self.model_parallel:
            torch.cuda.set_device(self.encoder.first_device)
            self.lm_head = self.lm_head.to(self.encoder.first_device)
            sequence_output = sequence_output.to(self.lm_head.weight.device)

        if self.config.tie_word_embeddings:
            # Rescale output before projecting on vocab
            # See https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/transformer/transformer.py#L586
            sequence_output = sequence_output * (self.model_dim ** -0.5)

        lm_logits = self.lm_head(sequence_output)
        masks = torch.ones_like(lm_logits,
                                dtype=torch.bool)
        masks[:, :, self.specials] = False
        lm_logits.masked_fill_(masks, -float('inf'))

        loss = None
        if labels is not None:
            # construct constrained mask here

            loss_fct = CrossEntropyLoss(ignore_index=-100)
            loss = loss_fct(lm_logits.view(-1, lm_logits.size(
                -1)), labels.view(-1))

        if not return_dict:
            output = (lm_logits,) + decoder_outputs[1:] + encoder_outputs
            return ((loss,) + output) if loss is not None else output

        return Seq2SeqLMOutput(
            loss=loss,
            logits=lm_logits,
            past_key_values=decoder_outputs.past_key_values,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            cross_attentions=decoder_outputs.cross_attentions,
            encoder_last_hidden_state=encoder_outputs.last_hidden_state,
            encoder_hidden_states=encoder_outputs.hidden_states,
            encoder_attentions=encoder_outputs.attentions,
        )

    def prepare_inputs_for_generation(
            self,
            input_ids,
            past=None,
            attention_mask=None,
            head_mask=None,
            decoder_head_mask=None,
            cross_attn_head_mask=None,
            use_cache=None,
            encoder_outputs=None,
            **kwargs
    ):

        # cut decoder_input_ids if past is used
        if past is not None:
            cut_input_ids = input_ids[:, -1:]
        else:
            cut_input_ids = input_ids

        return {
            "decoder_input_ids": cut_input_ids,
            "past_key_values": past,
            "encoder_outputs": encoder_outputs,
            "attention_mask": attention_mask,
            "head_mask": head_mask,
            "decoder_head_mask": decoder_head_mask,
            "cross_attn_head_mask": cross_attn_head_mask,
            "use_cache": use_cache,
            "full_decoder_input_ids": input_ids
        }

    def input_to_actions(self, input_ids: torch.LongTensor):
        # input_ids : B x L
        input_actions = deepcopy(input_ids)
        if self.action_type == 'integer':
            is_sep = (input_ids == self.sep)
            is_end = (input_ids == self.mention_end)
            is_start = (input_ids == self.mention_start)
            is_ent = (is_sep.cumsum(-1) - is_end.cumsum(-1)).bool()
            is_copy = ((~is_start) & (~is_ent) & (~is_end))
        else:
            cluster_ids = self.cluster_ids.to(input_ids.device)
            is_not_cid = torch.isin(input_ids, cluster_ids, invert=True)
            is_not_start = (input_ids != self.mention_start)
            if self.add_mention_end:
                is_not_end = (input_ids != self.mention_end)
                is_copy = (is_not_start & is_not_end & is_not_cid)
            else:
                is_copy = (is_not_start & is_not_cid)
        input_actions[:, 1:][is_copy[:, 1:]] = self.copy_id
        return input_actions
