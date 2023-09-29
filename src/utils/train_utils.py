import json
import os
from pathlib import Path

import torch


def add2config(model_args, data_args, config):
    attributes2add_list = [
        'model_variation',
        'num_prompts',
        'prompt_dropout',
        'prompt_hidden_condition',
        'prompt_layers',
        'prompt_attention_heads',
        'prompt_d_model',
        'prompt_ffn_dim',
        'tok_k_rules',
        'unfreeze_up_proj',
        'unfreeze_q_proj',
        'unfreeze_v_proj',
        'unfreeze_attn',
        'n_enc_recurrence',
        'n_dec_recurrence'
    ]
    for att in attributes2add_list:
        if getattr(model_args, att) is not None and not hasattr(config, att):
            setattr(config, att, getattr(model_args, att))

    if data_args.max_descriptor_length is not None and not hasattr(config, 'max_descriptor_length'):
        config.max_descriptor_length = data_args.max_descriptor_length
    if data_args.max_source_length is not None and not hasattr(config, 'max_source_length'):
        config.max_source_length = data_args.max_source_length

    if model_args.encoder_self_prefix_layer_ids is not None and not hasattr(config, 'encoder_self_prefix_layer_ids'):
        if model_args.encoder_self_prefix_layer_ids == 'all':
            config.encoder_self_prefix_layer_ids = list(range(config.encoder_layers))
        else:
            config.encoder_self_prefix_layer_ids = model_args.encoder_self_prefix_layer_ids
    if model_args.decoder_self_prefix_layer_ids is not None and not hasattr(config, 'decoder_self_prefix_layer_ids'):
        if model_args.decoder_self_prefix_layer_ids == 'all':
            config.decoder_self_prefix_layer_ids = list(range(config.decoder_layers))
        else:
            config.decoder_self_prefix_layer_ids = model_args.decoder_self_prefix_layer_ids
    if model_args.decoder_cross_prefix_layer_ids is not None and not hasattr(config, 'decoder_cross_prefix_layer_ids'):
        if model_args.decoder_cross_prefix_layer_ids == 'all':
            config.decoder_cross_prefix_layer_ids = list(range(config.decoder_layers))
        else:
            config.decoder_cross_prefix_layer_ids = model_args.decoder_cross_prefix_layer_ids

    if model_args.freeze_encoder_layers is not None and not hasattr(config, 'freeze_encoder_layers'):
        start_layer, end_layer = model_args.freeze_encoder_layers.split("-")
        config.freeze_encoder_layers = list(range(int(start_layer), int(end_layer)+1))
    if model_args.freeze_decoder_layers is not None and not hasattr(config, 'freeze_decoder_layers'):
        start_layer, end_layer = model_args.freeze_decoder_layers.split("-")
        config.freeze_decoder_layers = list(range(int(start_layer), int(end_layer)+1))

    # fix decoder_start_token_id to bos bugs for Bart
    config.decoder_start_token_id = config.bos_token_id
    config.forced_bos_token_id = config.bos_token_id

    #fix vocab size bug
    if model_args.config_name is not None and \
       'bart-large-launch_xsum' in model_args.config_name and 'bart-large-launch_xsum' not in model_args.model_name_or_path:
        config.vocab_size = 50265
    return config


def batchify_data(data, curr_batch):
    for k in data.keys():
        if k in curr_batch.keys():
            curr_batch[k] = torch.cat((curr_batch[k], data[k]), dim=0)
        else:
            curr_batch[k] = data[k]
    return curr_batch


def write_samples_file(output_folder, device_str, iter_count, samples):
    existing = list(Path(output_folder).glob(f"{device_str}_iter_{iter_count}*"))
    with open(
        os.path.join(
            output_folder, f"{device_str}_iter_{iter_count}_{len(existing)}.json",
        ),
        "w",
    ) as f:
        json.dump(samples, f)
