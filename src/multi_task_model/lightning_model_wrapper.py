import math
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union

import torch
from torch import nn
import pytorch_lightning as pl
from pytorch_lightning.trainer.states import TrainerFn
from transformers.trainer_utils import EvalPrediction
from src.utils.metrics import build_compute_metrics_fn


class LightningModelWrapper(pl.LightningModule):
    def __init__(self, hugginface_trainer):
        super().__init__()
        self.hf_trainer = hugginface_trainer
        self.tokenizer = hugginface_trainer.tokenizer
        self.output_dir = Path(hugginface_trainer.args.output_dir)
        self.model = hugginface_trainer.model
        self.config = self.model.config
        self._example_input_array = hugginface_trainer.model.dummy_inputs
        self.hf_trainer._max_length = self.hf_trainer.data_args.val_max_target_length
        self.hf_trainer._num_beams = self.model.config.num_beams
        self.metrics = {}

    def forward(self, **inputs):
        return self.model(**inputs)

    def training_step(self, *args, **kwargs) -> torch.Tensor:
        inputs = args[0]
        loss = self.hf_trainer.compute_loss(self.model, inputs)
        self.log('train_loss', loss, on_epoch=True)
        return loss

    def prediction_step(
            self,
            model: nn.Module,
            inputs: Dict[str, Union[torch.Tensor, Any]],
            prediction_loss_only: bool,
            ignore_keys: Optional[List[str]] = None,
    ) -> Dict[str, Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]]:
        loss, logits, labels = self.hf_trainer.prediction_step(model, inputs, prediction_loss_only, ignore_keys)
        return {'loss': loss, "logits": logits, "labels": labels}

    def prediction_epoch_end(self, outputs, prefix, task_name=None):
        epoch = 'epoch_%s' % self.trainer.current_epoch
        self.metrics[epoch] = {}
        if isinstance(outputs[0], list):
            losses = []
            for task_id, task_outputs in enumerate(outputs):
                loss = self.prediction_epoch_end(task_outputs, prefix, self.eval_task_name[task_id])
                losses.append(loss)
            return losses
        elif task_name is None:
            task_name = self.eval_task_name[0]

        prefix = '%s_%s_' % (prefix, task_name)
        compute_metrics = build_compute_metrics_fn(task_name, self.tokenizer)
        preds = torch.cat([x['logits'] for x in outputs]).detach().cpu().numpy()
        labels = torch.cat([x['labels'] for x in outputs]).detach().cpu().numpy()
        loss = torch.stack([x['loss'] for x in outputs]).mean()
        metrics = compute_metrics(EvalPrediction(predictions=preds, label_ids=labels))
        self.metrics[epoch][task_name] = {prefix+k: v for k, v in metrics.items()}
        self.metrics[epoch][task_name][prefix+'loss'] = float(loss)
        self.log(prefix + 'loss', loss, prog_bar=True, sync_dist=True)
        self.log_dict(dictionary=self.metrics[epoch][task_name], prog_bar=True, sync_dist=True)
        return loss

    def validation_step(self, *args, **kwargs):
        inputs = args[0]
        return self.prediction_step(self.model, inputs, self.hf_trainer.args.prediction_loss_only)

    def validation_epoch_end(self, outputs):
        return self.prediction_epoch_end(outputs, "eval")

    def test_step(self, *args, **kwargs):
        inputs = args[0]
        return self.prediction_step(self.model, inputs, self.hf_trainer.args.prediction_loss_only)

    def test_epoch_end(self, outputs):
        return self.prediction_epoch_end(outputs, "test")

    def setup(self, stage=None) -> None:
        # Get dataloader by calling it - train_dataloader() is called after setup() by default
        if stage is not TrainerFn.TESTING:
            train_dataloader = self.train_dataloader()
            n_gpu = torch.cuda.device_count()
            # Calculate total steps
            num_update_steps_per_epoch = len(train_dataloader) // self.trainer.accumulate_grad_batches // n_gpu
            num_update_steps_per_epoch = max(num_update_steps_per_epoch, 1)
            self.total_steps = math.ceil(self.trainer.max_epochs * num_update_steps_per_epoch)
            # Collect the names of each task
            self.eval_task_name = list(self.hf_trainer.eval_dataset.keys())
        else:
            self.eval_task_name = list(self.hf_trainer.test_dataset.keys())

    def configure_optimizers(self):
        self.hf_trainer.create_optimizer_and_scheduler(num_training_steps=self.total_steps)
        return {
            'optimizer': self.hf_trainer.optimizer,
            'lr_scheduler': {
                'scheduler': self.hf_trainer.lr_scheduler,
                'interval': 'step',
                'frequency': 1
            }
        }

    @pl.utilities.rank_zero_only
    def on_save_checkpoint(self, checkpoint: Dict[str, Any] = None) -> None:
        epoch = self.trainer.current_epoch if self.trainer is not None else ''
        save_path = self.output_dir.joinpath("epoch_%s" % epoch)
        self.model.save_pretrained(save_path)
        self.tokenizer.save_pretrained(save_path)

