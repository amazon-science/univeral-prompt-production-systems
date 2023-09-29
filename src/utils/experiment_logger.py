import os
import torch
import comet_ml
from pathlib import Path
from dataclasses import dataclass, field

from transformers.integrations import CometCallback
from transformers.utils import logging
from src.utils.misc import save_json, return_average_metric

import pytorch_lightning as pl
from pytorch_lightning.utilities import rank_zero_only, rank_zero_info

logger = logging.get_logger(__name__)


@dataclass
class ExperimentLoggerArguments:
    disable_exp_logger: bool = field(
        default=False, metadata={"help": "Disable the experiment logger"},
    )
    exp_workspace_name: str = field(
        default=None, metadata={"help": "The experiment workspace name"},
    )
    exp_project_name: str = field(
        default=None, metadata={"help": "The experiment project name"},
    )
    exp_logger_api_key: str = field(
        default=None, metadata={"help": "Api key"},
    )
    exp_name: str = field(
        default=None, metadata={"help": "The experiment name"},
    )


class MyCometCallback(CometCallback):
    """
    Easier set-up than parent.
    A :class:`~transformers.TrainerCallback` that sends the logs to `Comet ML <https://www.comet.ml/site/>`__.
    """

    def __init__(self, args: ExperimentLoggerArguments, previous_experiment: str = None):
        self._initialized = False
        self.log_args = args
        self.previous_experiment = previous_experiment

    def setup(self, args, state, model):
        """
        Setup the optional Comet.ml integration.

        Environment:
            COMET_MODE (:obj:`str`, `optional`):
                "OFFLINE", "ONLINE", or "DISABLED"
            COMET_PROJECT_NAME (:obj:`str`, `optional`):
                Comet.ml project name for experiments
            COMET_OFFLINE_DIRECTORY (:obj:`str`, `optional`):
                Folder to use for saving offline experiments when :obj:`COMET_MODE` is "OFFLINE"

        For a number of configurable items in the environment, see `here
        <https://www.comet.ml/docs/python-sdk/advanced/#comet-configuration-variables>`__.
        """
        experiment_kwargs = {
            "auto_metric_logging": False,
            "log_env_details": False, "log_env_host": False,
            "log_env_cpu": False, "log_env_gpu": False,
            "auto_output_logging": "simple",
            "api_key": self.log_args.exp_logger_api_key,
            "project_name": self.log_args.exp_project_name,
            "workspace": self.log_args.exp_workspace_name
        }
        self._initialized = True
        if state.is_world_process_zero:
            if self.previous_experiment is not None:
                experiment = comet_ml.ExistingExperiment(
                    previous_experiment=self.previous_experiment,
                    **experiment_kwargs
                )
                assert experiment.connection.experiment_id == self.previous_experiment
            else:
                experiment = comet_ml.Experiment(
                    **experiment_kwargs
                )
            experiment._set_model_graph(model, framework="transformers")
            experiment._log_parameters(args, prefix="args/", framework="transformers")
            experiment._log_parameters(model.config, prefix="config/", framework="transformers")
            experiment.set_name(self.log_args.exp_name)
            if not hasattr(model.config, 'experiment_id'):
                model.config.experiment_id = experiment.connection.experiment_id
            logger.info("Automatic Comet.ml online logging enabled")


class LoggingCallback(pl.Callback):
    def on_batch_end(self, trainer, pl_module):
        lrs = {f"lr_group_{i}": param["lr"] for i, param in enumerate(pl_module.trainer.optimizers[0].param_groups)}
        pl_module.logger.log_metrics(lrs)

    @rank_zero_only
    def _write_logs(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule, type_path: str, save_generations=True
    ) -> None:
        logger.info(f"***** {type_path} results at step {trainer.global_step:05d} *****")
        metrics = trainer.callback_metrics
        trainer.logger.log_metrics({k: v for k, v in metrics.items() if k not in ["log", "progress_bar", "preds"]})
        # Log results
        od = Path(pl_module.hparams.output_dir)
        if type_path == "test":
            results_file = od / "test_results.txt"
            generations_file = od / "test_generations.txt"
        else:
            # this never gets hit. I prefer not to save intermediate generations, and results are in metrics.json
            # If people want this it will be easy enough to add back.
            results_file = od / f"{type_path}_results/{trainer.global_step:05d}.txt"
            generations_file = od / f"{type_path}_generations/{trainer.global_step:05d}.txt"
            results_file.parent.mkdir(exist_ok=True)
            generations_file.parent.mkdir(exist_ok=True)
        with open(results_file, "a+") as writer:
            for key in sorted(metrics):
                if key in ["log", "progress_bar", "preds"]:
                    continue
                val = metrics[key]
                if isinstance(val, torch.Tensor):
                    val = val.item()
                msg = f"{key}: {val:.6f}\n"
                writer.write(msg)

        if not save_generations:
            return

        if "preds" in metrics:
            content = "\n".join(metrics["preds"])
            generations_file.open("w+").write(content)

    @rank_zero_only
    def on_test_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        if not hasattr(self, 'metrics'):
            self.metrics = {}
        metric = return_average_metric(trainer.callback_metrics)
        self.metrics['epoch_test'] = metric
        save_json(self.metrics, pl_module.output_dir / "test_metrics.json")
        #self._write_logs(trainer, pl_module, "test")

    @rank_zero_only
    def on_validation_end(self, trainer: pl.Trainer, pl_module):
        if not hasattr(self, 'metrics'):
            self.metrics = {}
        metric = return_average_metric(trainer.callback_metrics)
        self.metrics['epoch_%s' % trainer.current_epoch] = metric
        save_json(self.metrics, pl_module.output_dir / "eval_metrics.json")
