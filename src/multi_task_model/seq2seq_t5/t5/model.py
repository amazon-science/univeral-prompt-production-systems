import torch.nn as nn
from transformers.models.t5 import T5ForConditionalGeneration

from src.multi_task_model.seq2seq_t5.modeling_seq2seq import register_model, MyT5Model


@register_model
class _T5(T5ForConditionalGeneration):  # Adding a prefix since T5Seq2Seq exists in huggingface lib

    def __init__(self, config, **kwargs):
        super(_T5, self).__init__(config)
        self.model = MyT5Model(config)

    @staticmethod
    def model_type():
        return "T5"

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
        return super(_T5, self).forward(
            input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            head_mask=head_mask,
            decoder_head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            encoder_outputs=encoder_outputs,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            decoder_inputs_embeds=decoder_inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict
        )
