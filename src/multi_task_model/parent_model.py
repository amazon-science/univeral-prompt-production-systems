"""
Author: Jonathan Pilault
This files allows us to load various seq2se1 models (Bart, T5) and inherit the appropriate class.
Class contains common methods.
"""

import re
import torch
from src.multi_task_model.seq2seq_bart.modeling_seq2seq import get_model_class
from src.multi_task_data.multi_task_dataset import MAX_LENGTHS

from transformers.utils import logging

logger = logging.get_logger(__name__)


def get_model(config, model_args):
    """ Factory Method Pattern """

    def _create_model_class(model_args, model_type=None):
        model_type = model_args.model_variation or model_type
        model_class = get_model_class(model_type)
        logger.info(f'MultiTaskModel inherits from %s' % model_class)
        return model_class
    ModelClass = _create_model_class(model_args, config.model_type)

    class MultiTaskModel(ModelClass):
        def __init__(
            self,
            config,
            data_args,
        ):
            super().__init__(config)

            self.data_args = data_args
            self.init_weights()

            self._dummy_inputs = self.get_dummy_inputs(
                use_descriptors=('prefix' in config.model_variation),
                add_decoder_start_token_id=('mbart' in config.model_type)
            )

        @property
        def dummy_inputs(self):
            return self._dummy_inputs

        def get_dummy_inputs(self, use_descriptors, add_decoder_start_token_id):
            pad_token = self.config.pad_token_id
            l = MAX_LENGTHS["prompt_input_h"] // 5
            input_ids = torch.tensor([[0, 6, 10, 4, 2] * l, [0, 8, 12, 2, pad_token] * l], device=self.device)
            descriptors = torch.tensor([[0, 2, 9], [0, 8, pad_token]], device=self.device)
            l = self.config.max_descriptor_length - 3
            extra_pad = torch.tensor([[pad_token] * l, [pad_token] * l], device=self.device)
            descriptors = torch.cat((descriptors, extra_pad), dim=-1)
            decoder_start_token_id = torch.tensor([[self.config.decoder_start_token_id]] * 2, device=self.device)
            dummy_inputs = {
                "attention_mask": input_ids.ne(pad_token), "input_ids": input_ids,
            }
            if use_descriptors:
                dummy_inputs["descriptors"] = descriptors
                dummy_inputs["descriptors_attention_mask"] = descriptors.ne(pad_token)
            if add_decoder_start_token_id:
                dummy_inputs["decoder_start_token_id"] = None
            return dummy_inputs

        def freeze_layers(self, start_layer, end_layer, modules, unfrozen_modules):
            for name, param in modules.named_parameters():
                requires_grad = True
                match = re.match(self.get_layer_regexp(), name)
                if match:
                    layer_number = int(match.groups()[0])
                    requires_grad = not int(start_layer) <= layer_number <= int(end_layer)
                    unfreeze_spec = self.config.unfreeze_q_proj or self.config.unfreeze_up_proj or self.config.unfreeze_attn
                    if requires_grad and unfreeze_spec:
                        unfreeze = False or any([module in match.string for module in ['norm']])
                        if self.config.unfreeze_q_proj:
                            unfreeze = unfreeze or any([module in match.string for module in ['q_proj']])
                        if self.config.unfreeze_v_proj:
                            unfreeze = unfreeze or any([module in match.string for module in ['v_proj']])
                        if self.config.unfreeze_up_proj:
                            unfreeze = unfreeze or any([module in match.string for module in ['.wo.', '.fc2.']])
                        if self.config.unfreeze_attn:
                            unfreeze = unfreeze or any([module in match.string for module in ['.self_attn.', '.encoder_attn.']])
                        requires_grad = unfreeze
                    requires_grad = requires_grad or any([module in match.string for module in unfrozen_modules])
                elif name.startswith("embed_tokens") or name.startswith("embed_positions"):
                    requires_grad = False
                param.requires_grad = requires_grad

        def freeze_model_layers(
            self,
            model_args,
            unfrozen_modules=["prompt_generator", "bias"]
        ):
            if model_args.freeze_encoder_layers is not None:
                start_layer, end_layer = model_args.freeze_encoder_layers.split("-")
                self.freeze_layers(start_layer, end_layer, self.model.encoder, unfrozen_modules)

            if model_args.freeze_decoder_layers is not None:
                start_layer, end_layer = model_args.freeze_decoder_layers.split("-")
                self.freeze_layers(start_layer, end_layer, self.model.decoder, unfrozen_modules)

            if model_args.freeze_encoder_layers is not None or model_args.freeze_decoder_layers is not None:
                self.model.shared.weight.requires_grad = False
                self.lm_head.weight.requires_grad = False
                if hasattr(self.model, 'prompt_generator'):
                    if hasattr(self.model.prompt_generator, 'token_embeddings'):
                        self.model.prompt_generator.token_embeddings.weight.requires_grad = False
                    if hasattr(self.model.prompt_generator, 'embed_positions'):
                        self.model.prompt_generator.embed_positions.weight.requires_grad = False

            for name, param in self.model.named_parameters():
                logger.info(
                    "%s - %s", name, ("Unfrozen" if param.requires_grad else "FROZEN")
                )

        def _log_params(self):
            all_parameters = list(self.parameters())
            trainable_parameters = [p for p in all_parameters if p.requires_grad]

            total_unique = {p.data_ptr(): p for p in all_parameters}.values()
            total_params = sum(p.numel() for p in total_unique)
            trainable_unique = {p.data_ptr(): p for p in trainable_parameters}.values()
            trainable_params = sum(p.numel() for p in trainable_unique)

            logger.info("total_param %d" % total_params)
            logger.info("total_trainable_param %d" % trainable_params)
            return total_params, trainable_params

        def set_input_embeddings(self, value):
            super().set_input_embeddings(value)
            if hasattr(self.model, 'prompt_generator'):
                if hasattr(self.model.prompt_generator, 'token_embeddings'):
                    self.model.prompt_generator.token_embeddings = value
                if hasattr(self.model.prompt_generator, 'embed_positions'):
                    self.model.prompt_generator.embed_positions = self.model.encoder.embed_positions

    return MultiTaskModel
