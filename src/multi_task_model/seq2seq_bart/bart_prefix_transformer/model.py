import torch
import torch.nn as nn

from typing import Tuple, Dict, Any

from transformers.modeling_outputs import Seq2SeqLMOutput
from transformers.models.bart import (
    BartForConditionalGeneration,
    modeling_bart
)
from transformers.modeling_outputs import ModelOutput

from src.multi_task_model.seq2seq_bart.modeling_seq2seq import register_model, MyBartModel
from src.prompts.prompt_generators import TransformerPrompt


@register_model
class _Bart_Prefix_Trsf(BartForConditionalGeneration):
    """
    _Bart_Prefix is very similar to `BartForConditionalGeneration` with a few differences to
    handle the field `descriptors` in the methods below.
    """
    def __init__(self, config, **kwargs):
        super(_Bart_Prefix_Trsf, self).__init__(config)
        prompt_generator = TransformerPrompt(
            config,
            token_embeddings=self.model.shared,
        )
        self.model = MyBartModel(config, prompt_generator)

    @staticmethod
    def _expand_inputs_for_generation(
        input_ids: torch.LongTensor,
        expand_size: int = 1,
        is_encoder_decoder: bool = False,
        attention_mask: torch.LongTensor = None,
        encoder_outputs: ModelOutput = None,
        **model_kwargs,
    ) -> Tuple[torch.LongTensor, Dict[str, Any]]:
        expanded_return_idx = (
            torch.arange(input_ids.shape[0]).view(-1, 1).repeat(1, expand_size).view(-1).to(input_ids.device)
        )
        input_ids = input_ids.index_select(0, expanded_return_idx)

        if "token_type_ids" in model_kwargs:
            token_type_ids = model_kwargs["token_type_ids"]
            model_kwargs["token_type_ids"] = token_type_ids.index_select(0, expanded_return_idx)

        if "descriptors" in model_kwargs:  # new
            descriptors = model_kwargs["descriptors"]
            model_kwargs["descriptors"] = descriptors.index_select(0, expanded_return_idx)

        if "descriptors_attention_mask" in model_kwargs:  # new
            descriptors_attention_mask = model_kwargs["descriptors_attention_mask"]
            model_kwargs["descriptors_attention_mask"] = descriptors_attention_mask.index_select(0, expanded_return_idx)

        if attention_mask is not None:
            model_kwargs["attention_mask"] = attention_mask.index_select(0, expanded_return_idx)

        if is_encoder_decoder:
            assert encoder_outputs is not None
            encoder_outputs["last_hidden_state"] = encoder_outputs.last_hidden_state.index_select(
                0, expanded_return_idx.to(encoder_outputs.last_hidden_state.device)
            )
            model_kwargs["encoder_outputs"] = encoder_outputs
        return input_ids, model_kwargs

    def adjust_logits_during_generation(self, logits: torch.FloatTensor, cur_len, **kwargs) -> torch.FloatTensor:
        if cur_len == 1:
            logits[:, self.config.bos_token_id] = 1000
        else:
            logits[:, self.config.bos_token_id] = torch.finfo(logits.dtype).min
        return logits

    def prepare_inputs_for_generation(
        self,
        decoder_input_ids,
        descriptors,  # new
        descriptors_attention_mask, # new
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
            decoder_input_ids = decoder_input_ids[:, -1:]

        return {
            "input_ids": None,  # encoder_outputs is defined. input_ids not needed
            "descriptors": descriptors,  # new
            "descriptors_attention_mask": descriptors_attention_mask,
            "encoder_outputs": encoder_outputs,
            "past_key_values": past,
            "decoder_input_ids": decoder_input_ids,
            "attention_mask": attention_mask,
            "head_mask": head_mask,
            "decoder_head_mask": decoder_head_mask,
            "cross_attn_head_mask": cross_attn_head_mask,
            "use_cache": use_cache,  # change this to avoid caching (presumably for debugging)
        }

    @staticmethod
    def model_type():
        return "BART_PREFIX_TRANSFORMER"

    @classmethod
    def get_layer_regexp(cls):
        return r"layers.*\.([0-9]+)\..*"

    def forward(
        self,
        input_ids=None,
        descriptors=None,
        descriptors_attention_mask=None,
        decoder_start_token_id=None,
        attention_mask=None,
        decoder_input_ids=None,
        decoder_attention_mask=None,
        head_mask=None,
        decoder_head_mask=None,
        cross_attn_head_mask=None,
        encoder_outputs=None,
        past_key_values=None,
        inputs_embeds=None,
        decoder_inputs_embeds=None,
        labels=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        **kwargs,
    ):
        r"""
                labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
                    Labels for computing the masked language modeling loss. Indices should either be in ``[0, ...,
                    config.vocab_size]`` or -100 (see ``input_ids`` docstring). Tokens with indices set to ``-100`` are ignored
                    (masked), the loss is only computed for the tokens with labels in ``[0, ..., config.vocab_size]``.

                Returns: Seq2SeqLMOutput
                """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if labels is not None:
            if decoder_input_ids is None:
                decoder_input_ids = modeling_bart.shift_tokens_right(
                    labels, self.config.pad_token_id, self.config.decoder_start_token_id
                )

        outputs = self.model(
            input_ids,
            descriptors=descriptors,  # new
            descriptors_attention_mask=descriptors_attention_mask,  # new
            decoder_start_token_id=decoder_start_token_id,  # new
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            encoder_outputs=encoder_outputs,
            decoder_attention_mask=decoder_attention_mask,
            head_mask=head_mask,
            decoder_head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            decoder_inputs_embeds=decoder_inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        lm_logits = self.lm_head(outputs[0]) + self.final_logits_bias

        masked_lm_loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            masked_lm_loss = loss_fct(lm_logits.view(-1, self.config.vocab_size), labels.view(-1))

        if not return_dict:
            output = (lm_logits,) + outputs[1:]
            return ((masked_lm_loss,) + output) if masked_lm_loss is not None else output

        return Seq2SeqLMOutput(
            loss=masked_lm_loss,
            logits=lm_logits,
            past_key_values=outputs.past_key_values,
            decoder_hidden_states=outputs.decoder_hidden_states,
            decoder_attentions=outputs.decoder_attentions,
            cross_attentions=outputs.cross_attentions,
            encoder_last_hidden_state=outputs.encoder_last_hidden_state,
            encoder_hidden_states=outputs.encoder_hidden_states,
            encoder_attentions=outputs.encoder_attentions,
        )


