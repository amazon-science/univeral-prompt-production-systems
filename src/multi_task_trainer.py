import os
import comet_ml
from packaging import version
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple, List, Union, Any

import torch
import torch.nn as nn
from torch import autograd

from torch.utils.data.dataset import Dataset
from transformers import trainer
from transformers.training_args_seq2seq import Seq2SeqTrainingArguments
from transformers.trainer_seq2seq import Seq2SeqTrainer
from transformers.trainer_utils import PredictionOutput
from transformers.deepspeed import is_deepspeed_zero3_enabled
from transformers import (
    MBartTokenizer, MBartTokenizerFast, MBart50Tokenizer,
    MBart50TokenizerFast, M2M100Tokenizer
)

from src.utils.metrics import build_compute_metrics_fn
from src.multi_task_data.utils import tokenizer_lang_codes
from transformers.file_utils import cached_property, torch_required
from transformers.utils import logging

logger = logging.get_logger(__name__)
log_levels = logging.get_log_levels_dict().copy()
trainer_log_levels = dict(**log_levels, passive=-1)
MULTILINGUAL_TOKENIZERS = [
    MBartTokenizer,
    MBartTokenizerFast,
    MBart50Tokenizer,
    MBart50TokenizerFast,
    M2M100Tokenizer
]


if version.parse(torch.__version__) >= version.parse("1.6"):
    from torch.cuda.amp import autocast

if trainer.is_apex_available():
    amp = trainer.amp


def nop(it, *a, **k):
    return it


@dataclass
class MultiTaskTrainingArguments(Seq2SeqTrainingArguments):
    log_level: Optional[str] = field(
        default="info",
        metadata={
            "help": "Logger log level to use on the main node. Possible choices are the log levels as strings: 'debug', 'info', 'warning', 'error' and 'critical', plus a 'passive' level which doesn't set anything and lets the application set the level. Defaults to 'passive'.",
            "choices": trainer_log_levels.keys(),
        },
    )
    log_level_replica: Optional[str] = field(
        default="info",
        metadata={
            "help": "Logger log level to use on replica nodes. Same choices and defaults as ``log_level``",
            "choices": trainer_log_levels.keys(),
        },
    )
    use_lightning: bool = field(
        default=False,
        metadata={"help": "Whether to use pytorch lightning."},
    )
    lightning_checkpoint: Optional[str] = field(
        default=None,
        metadata={"help": "cpkt file to use for lightning trainer checkpoint.",},
    )
    every_n_epochs: Optional[int] = field(
        default=1,
        metadata={"help": "call checkpoint and eval every n epoch.",},
    )

    def __post_init__(self):
        super().__post_init__()
        if self.use_lightning:
            self._n_gpu = 1

    @cached_property
    @torch_required
    def _setup_devices(self) -> "torch.device":
        if self.use_lightning:
            self._n_gpu = 1
            if self.local_rank == -1:
                device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            else:
                device = torch.device("cuda", self.local_rank)
            if device.type == "cuda":
                torch.cuda.set_device(device)
            return device
        else:
            return super()._setup_devices

    @property
    def should_save(self):
        if self.use_lightning:
            return False
        else:
            return super().should_save

    @property
    @torch_required
    def process_index(self):
        if self.use_lightning:
            return 0
        else:
            return super().process_index


class MultiTaskTrainer(Seq2SeqTrainer):
    def __init__(self, tokenizer, tasks, data_args, callback, test_dataset, *args, **kwargs):
        super(MultiTaskTrainer, self).__init__(*args, **kwargs)
        self.tokenizer = tokenizer
        self.tasks = tasks
        self.data_args = data_args
        self.test_dataset = test_dataset

        if callback is not None:
            self.callback_handler.add_callback(callback)
        if 'descriptors' in self.model.dummy_inputs:
            self.use_descriptors = True
        else:
            self.use_descriptors = False

    def _get_train_sampler(self):
        if self.args.use_lightning:
            return None
        else:
            return super()._get_train_sampler()

    def _get_eval_sampler(self, eval_dataset):
        if self.args.use_lightning:
            return None
        else:
            return super()._get_eval_sampler(eval_dataset)

    def save_model(self, output_dir: Optional[str] = None):
        if isinstance(self.model.config.decoder_start_token_id, torch.Tensor):
            self.model.config.decoder_start_token_id = self.model.config.forced_bos_token_id
        super(MultiTaskTrainer, self).save_model(output_dir=output_dir or self.args.output_dir)
        if self.is_world_process_zero():
            self.tokenizer.save_pretrained(output_dir or self.args.output_dir)

    def update_multilingual_config(self, task_name):
        if isinstance(self.tokenizer, tuple(MULTILINGUAL_TOKENIZERS)):
            src_lang = task_name.split()[1]
            trg_lang = task_name.split()[3]
            self.tokenizer.src_lang = tokenizer_lang_codes[src_lang]
            self.tokenizer.trg_lang = tokenizer_lang_codes[trg_lang]
            # For multilingual translation models like mBART-50 and M2M100 we need to force the target language token
            # as the first generated token. We ask the user to explicitly provide this as --forced_bos_token argument.
            forced_bos_token_id = (
                self.tokenizer.lang_code_to_id[
                    self.data_args.forced_bos_token] if self.data_args.forced_bos_token is not None else None
            )
            self.model.config.forced_bos_token_id = forced_bos_token_id

        if isinstance(self.tokenizer, (MBartTokenizer, MBart50Tokenizer)):
            self.model.config.decoder_start_token_id = self.tokenizer.lang_code_to_id[self.tokenizer.trg_lang]
        elif isinstance(self.tokenizer, (MBartTokenizerFast, MBart50TokenizerFast)):
            self.model.config.decoder_start_token_id = self.tokenizer.convert_tokens_to_ids(self.tokenizer.trg_lang)

    def evaluate(
            self,
            eval_dataset: Optional[List[Dataset]] = None,
            ignore_keys: Optional[List[str]] = None,
            metric_key_prefix: str = "eval",
            max_length: Optional[int] = None,
            num_beams: Optional[int] = None,
    ) -> Dict[str, Dict[str, float]]:
        output = {}
        datasets = eval_dataset or self.eval_dataset
        logger.info("*** Evaluate on dev ***")

        for task_name, eval_dataset in datasets.items():
            if "translate" in task_name:
                self.update_multilingual_config(task_name)

            logger.info(task_name)
            if self.args.predict_with_generate:
                self.compute_metrics = build_compute_metrics_fn(
                    task_name, self.tokenizer
                )
            else:
                self.compute_metrics = None
            task_name = task_name.replace(" ", "_")

            eval_result = super().evaluate(
                eval_dataset=eval_dataset,
                ignore_keys=ignore_keys,
                metric_key_prefix=metric_key_prefix,
                max_length=max_length,
                num_beams=num_beams
            )
            output[task_name] = eval_result
            max_eval_samples = self.data_args.max_eval_samples \
                if self.data_args.max_eval_samples is not None else len(eval_dataset)
            eval_result["eval_%s_samples" % task_name] = min(max_eval_samples, len(eval_dataset))
            self.log_metrics("eval_%s" % task_name, eval_result)
            self.save_metrics("eval_%s" % task_name, eval_result)
        return output

    def predict(
        self,
        test_dataset: Optional[List[Dataset]] = None,
        ignore_keys: Optional[List[str]] = None,
        metric_key_prefix: str = "predict",
        max_length: Optional[int] = None,
        num_beams: Optional[int] = None,
    ) -> Dict[str, PredictionOutput]:

        output = {}
        datasets = test_dataset or self.test_dataset
        logger.info("*** Predict ***")

        for task_name, test_dataset in datasets.items():
            if "translate" in task_name:
                self.update_multilingual_config(task_name)

            logger.info(task_name)
            if self.args.predict_with_generate:
                self.compute_metrics = build_compute_metrics_fn(
                    task_name, self.tokenizer
                )
            else:
                self.compute_metrics = None
            task_name = task_name.replace(" ", "_")

            predict_results = super().predict(
                eval_dataset=test_dataset,
                ignore_keys=ignore_keys,
                metric_key_prefix=metric_key_prefix,
                max_length=max_length,
                num_beams=num_beams
            )
            output[task_name] = predict_results
            metrics = predict_results.metrics
            max_eval_samples = self.data_args.max_eval_samples \
                if self.data_args.max_eval_samples is not None else len(test_dataset)
            metrics["predict_%s_samples" % task_name] = min(max_eval_samples, len(test_dataset))
            self.log_metrics("predict_%s" % task_name, metrics)
            self.save_metrics("predict_%s" % task_name, metrics)
            if self.is_world_process_zero():
                if self.args.predict_with_generate:
                    predictions = self.tokenizer.batch_decode(
                        predict_results.predictions, skip_special_tokens=True, clean_up_tokenization_spaces=True
                    )
                    predictions = [pred.strip() for pred in predictions]
                    output_prediction_file = os.path.join(self.args.output_dir, "generated_predictions_%s.txt" % task_name)
                    with open(output_prediction_file, "w") as writer:
                        writer.write("\n".join(predictions))
        return output

    def prediction_step(
        self,
        model: nn.Module,
        inputs: Dict[str, Union[torch.Tensor, Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[List[str]] = None,
    ) -> Tuple[Optional[float], Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Perform an evaluation step on :obj:`model` using obj:`inputs`.

        Subclass and override to inject custom behavior.

        Args:
            model (:obj:`nn.Module`):
                The model to evaluate.
            inputs (:obj:`Dict[str, Union[torch.Tensor, Any]]`):
                The inputs and targets of the model.

                The dictionary will be unpacked before being fed to the model. Most models expect the targets under the
                argument :obj:`labels`. Check your model's documentation for all accepted arguments.
            prediction_loss_only (:obj:`bool`):
                Whether or not to return the loss only.

        Return:
            Tuple[Optional[float], Optional[torch.Tensor], Optional[torch.Tensor]]: A tuple with the loss, logits and
            labels (each being optional).
        """

        if not self.args.predict_with_generate or prediction_loss_only:
            return super().prediction_step(
                model, inputs, prediction_loss_only=prediction_loss_only, ignore_keys=ignore_keys
            )

        has_labels = "labels" in inputs
        inputs = self._prepare_inputs(inputs)

        # XXX: adapt synced_gpus for fairscale as well
        gen_kwargs = {
            "max_length": self._max_length if self._max_length is not None else self.model.config.max_length,
            "min_length": self.model.config.min_length,
            "num_beams": self._num_beams if self._num_beams is not None else self.model.config.num_beams,
            "length_penalty": self.model.config.length_penalty,
            "synced_gpus": True if is_deepspeed_zero3_enabled() else False,
        }
        model_kwards = {
            "input_ids": inputs["input_ids"],
            "attention_mask": inputs["attention_mask"],
        }
        if self.use_descriptors:
            model_kwards["descriptors"] = inputs["descriptors"]
            model_kwards["descriptors_attention_mask"] = inputs["descriptors_attention_mask"]
        generated_tokens = self.model.generate(
            **model_kwards,
            **gen_kwargs,
        )
        # in case the batch is shorter than max length, the output should be padded
        if generated_tokens.shape[-1] < gen_kwargs["max_length"]:
            generated_tokens = self._pad_tensors_to_max_len(generated_tokens, gen_kwargs["max_length"])

        with torch.no_grad():
            if self.use_amp:
                with autocast():
                    outputs = model(**inputs)
            else:
                outputs = model(**inputs)
            if has_labels:
                if self.label_smoother is not None:
                    loss = self.label_smoother(outputs, inputs["labels"]).mean().detach()
                else:
                    loss = (outputs["loss"] if isinstance(outputs, dict) else outputs[0]).mean().detach()
            else:
                loss = None

        if self.args.prediction_loss_only:
            return (loss, None, None)

        labels = inputs["labels"]
        if labels.shape[-1] < gen_kwargs["max_length"]:
            labels = self._pad_tensors_to_max_len(labels, gen_kwargs["max_length"])

        return (loss, generated_tokens, labels)

    def training_step(self, model: nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]]) -> torch.Tensor:
        """
        No changes from Hugging Face. This is used for debugging anomalies.
        Perform a training step on a batch of inputs.
        Subclass and override to inject custom behavior.
        Args:
            model (:obj:`nn.Module`):
                The model to train.
            inputs (:obj:`Dict[str, Union[torch.Tensor, Any]]`):
                The inputs and targets of the model.

                The dictionary will be unpacked before being fed to the model. Most models expect the targets under the
                argument :obj:`labels`. Check your model's documentation for all accepted arguments.
        Return:
            :obj:`torch.Tensor`: The tensor with training loss on this batch.
        """
        #with autograd.detect_anomaly():
        model.train()
        inputs = self._prepare_inputs(inputs)

        if self.use_amp:
            with autocast():
                loss = self.compute_loss(model, inputs)
        else:
            loss = self.compute_loss(model, inputs)

        if self.args.n_gpu > 1:
            loss = loss.mean()  # mean() to average on multi-gpu parallel training

        if self.args.gradient_accumulation_steps > 1 and not self.deepspeed:
            # deepspeed handles loss scaling by gradient_accumulation_steps in its `backward`
            loss = loss / self.args.gradient_accumulation_steps

        if self.use_amp:
            self.scaler.scale(loss).backward()
        elif self.use_apex:
            with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                scaled_loss.backward()
        elif self.deepspeed:
            # loss gets scaled under gradient_accumulation_steps in deepspeed
            loss = self.deepspeed.backward(loss)
        else:
            loss.backward()

        return loss.detach()