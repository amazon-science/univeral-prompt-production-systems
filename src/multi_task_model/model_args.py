from dataclasses import dataclass, field
from typing import List, Optional, Union


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        metadata={
            "help": "Path to pretrained model or model identifier from huggingface.co/models or "
        }
    )
    model_variation: str = field(
        default=None,
        metadata={
            "help": "Model variations include:"
                    "t5_prefix,"                              # same as prefix finetuning
                    "t5_prefix_transformer,"                  # transformer for promt gen
                    "t5_prefix_universal_transformer,"        # with recurrence
                    "t5_prefix_production_system"             # condition-rule interaction
                    "t5_prefix_universal_production_system"   # with recurrence
        },
    )
    config_name: Optional[str] = field(
        default=None,
        metadata={
            "help": "Pretrained config name or path if not the same as model_name"
        },
    )
    tokenizer_name: Optional[str] = field(
        default=None,
        metadata={
            "help": "Pretrained tokenizer name or path if not the same as model_name"
        },
    )
    length_penalty: Optional[float] = field(
        default=1.0,
        metadata={
            "help": "Exponential penalty to the length. 1.0 means no penalty. Set to values < 1.0 in order to encourage"
            " the model to generate shorter sequences, to a value > 1.0 in order to encourage the model to produce "
            "longer sequences."
        }
    )
    num_beams: Optional[int] = field(
        default=1,
        metadata={
            "help": "Number of beams to use for evaluation. This argument will be passed to ``model.generate``, "
            "which is used during ``evaluate`` and ``predict``."
        },
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={
            "help": "Where do you want to store the pretrained models downloaded from s3"
        },
    )
    freeze_encoder_layers: str = field(
        default=None,
        metadata={"help": "Freeze encoder layers. format: <start_layer>-<end_layer>"},
    )
    freeze_decoder_layers: str = field(
        default=None,
        metadata={"help": "Freeze decoder layers. format: <start_layer>-<end_layer>"},
    )
    unfreeze_up_proj: bool = field(
        default=False,
        metadata={"help": "Instead of unfreeze a whole transformer layer, we unfreeze just up projection fc2."},
    )
    unfreeze_q_proj: bool = field(
        default=False,
        metadata={"help": "Instead of unfreeze a whole transformer layer, we unfreeze just query projection."},
    )
    unfreeze_v_proj: bool = field(
        default=False,
        metadata={"help": "Instead of unfreeze a whole transformer layer, we unfreeze just query projection."},
    )
    unfreeze_attn: bool = field(
        default=False,
        metadata={"help": "Instead of unfreeze a whole transformer layer, we unfreeze just query attention layers."},
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    num_prompts: int = field(
        default=None,
        metadata={"help": "Number of prompt vectors that should be generated."},
    )
    prompt_layers: int = field(
        default=None,
        metadata={"help": "Number of prompt generator layers that should be used."},
    )
    prompt_attention_heads: int = field(
        default=None,
        metadata={"help": "Number of prompt attn heads in Transformer based prompts."},
    )
    prompt_d_model: int = field(
        default=None,  # 768
        metadata={"help": "Prompt dimensionality of the layers and the pooler layer."},
    )
    prompt_ffn_dim: int = field(
        default=None,  # 3072
        metadata={"help": "Intermediate layer size in Transformer based prompts."},
    )
    prompt_dropout: float = field(
        default=None,
        metadata={"help": "Dropout for prompt generators."}
    )
    prompt_hidden_condition: bool = field(
        default=False,
        metadata={"help": "Whether to also use hidden states to condition prompt generator."},
    )
    encoder_self_prefix_layer_ids: Optional[List[str]] = field(
        default=None,
        metadata={
            "help": "Append prompt embeddings to specific encoder layers using ids."
        },
    )
    decoder_self_prefix_layer_ids: Optional[List[str]] = field(
        default=None,
        metadata={
            "help": "Append prompt embeddings to specific decoder layers using ids."
        },
    )
    decoder_cross_prefix_layer_ids: Optional[List[str]] = field(
        default=None,
        metadata={
            "help": "Append prompt embeddings to specific decoder layers using ids."
        },
    )
    tok_k_rules: int = field(
        default=None,
        metadata={"help": "Number of top rules to be chosen."},
    )
    n_enc_recurrence: int = field(
        default=None,
        metadata={"help": "Universal prompt loops n times for encoder."},
    )
    n_dec_recurrence: int = field(
        default=None,
        metadata={"help": "Universal prompt loops n times for decoder."},
    )

    def __post_init__(self):
        if 'prefix' in self.model_variation:

            id_args_dict = {
                'encoder_self_prefix_layer_ids': self.encoder_self_prefix_layer_ids,
                'decoder_self_prefix_layer_ids': self.decoder_self_prefix_layer_ids,
                'decoder_cross_prefix_layer_ids': self.decoder_cross_prefix_layer_ids
            }
            for arg, ids in id_args_dict.items():
                ids = 'all' if ids is None else [int(i) for i in ids]
                setattr(self, arg, ids)

            if self.n_enc_recurrence is not None and self.encoder_self_prefix_layer_ids \
                    and isinstance(self.encoder_self_prefix_layer_ids, list):
                last = max(self.encoder_self_prefix_layer_ids)
                add_rec_ids = [last + 1 + i for i in range(self.n_enc_recurrence)]
                self.encoder_self_prefix_layer_ids.extend(add_rec_ids)
            if self.n_dec_recurrence is not None and self.decoder_self_prefix_layer_ids \
                and isinstance(self.decoder_self_prefix_layer_ids, list):
                last = max(self.decoder_self_prefix_layer_ids)
                add_rec_ids = [last + 1 + i for i in range(self.n_dec_recurrence)]
                self.decoder_self_prefix_layer_ids.extend(add_rec_ids)
            if self.n_dec_recurrence is not None and self.decoder_cross_prefix_layer_ids \
                and isinstance(self.decoder_cross_prefix_layer_ids, list):
                last = max(self.decoder_cross_prefix_layer_ids)
                add_rec_ids = [last + 1 + i for i in range(self.n_dec_recurrence)]
                self.decoder_cross_prefix_layer_ids.extend(add_rec_ids)

            if self.prompt_dropout is None:
                self.prompt_dropout = 0.0

            if self.prompt_d_model is None and self.prompt_attention_heads is not None:
                self.prompt_d_model = self.prompt_attention_heads * 32
            if self.prompt_ffn_dim is None and self.prompt_attention_heads is not None:
                self.prompt_ffn_dim = self.prompt_attention_heads * 128
            if self.prompt_d_model is not None and self.prompt_attention_heads is not None:
                assert self.prompt_d_model % self.prompt_attention_heads == 0, \
                    'prompt_d_model should be divisible by prompt_attention_heads'
