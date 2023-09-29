import os
import re
import sys
import glob
import torch
import logging
import warnings

import transformers.utils.logging
from transformers.utils import logging
from transformers import (
    HfArgumentParser,
    AutoTokenizer,
    AutoConfig
)
from src.multi_task_model.parent_model import get_model
from src.multi_task_trainer import MultiTaskTrainer, MultiTaskTrainingArguments
from src.multi_task_data.multi_task_data_collator import MultiTaskDataCollator
from src.multi_task_data.multi_task_dataset import MultiTaskDataset
from src.multi_task_data.utils import read_tasks_file
from src.utils.train_utils import add2config
from src.utils.experiment_logger import ExperimentLoggerArguments, LoggingCallback
from src.utils.metrics import PAD_TOKEN_ID
from src.multi_task_data.data_args import MultiTaskDataArguments
from src.multi_task_model.model_args import ModelArguments

from pytorch_lightning.plugins import DDPPlugin
from pytorch_lightning import seed_everything, Trainer
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from src.multi_task_model.lightning_model_wrapper import LightningModelWrapper
from src.multi_task_data.lightning_data_wrapper import LightningDataWrapper

logger = logging.get_logger(__name__)
warnings.filterwarnings("ignore")


def setup_logging(training_args):
    logging.enable_default_handler()
    logging.enable_explicit_format()
    logging.set_verbosity(logging.ERROR)


def check_resume(model_args, training_args):
    if (
        not training_args.overwrite_output_dir
        and not os.path.isdir(model_args.model_name_or_path)
        and os.path.isdir(training_args.output_dir)
    ):

        checkpoints_sorted = list(sorted(glob._iglob(
            pathname=os.path.join(training_args.output_dir, 'epoch_*'), recursive=True, dironly=True
        )))
        if checkpoints_sorted:
            model_args.model_name_or_path = checkpoints_sorted[-1]
        lightning_checkpoints_sorted = list(sorted(glob.glob(
            pathname=os.path.join(training_args.output_dir, "*.ckpt"), recursive=True
        )))
        if checkpoints_sorted:
            training_args.lightning_checkpoint = lightning_checkpoints_sorted[-1]


def parse_cmd_args():
    parser = HfArgumentParser(
        (
            ModelArguments,
            MultiTaskDataArguments,
            MultiTaskTrainingArguments,
            ExperimentLoggerArguments,
        )
    )

    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        model_args, data_args, training_args, exp_logger_args = parser.parse_json_file(
            json_file=os.path.abspath(sys.argv[1])
        )
    else:
        (
            model_args,
            data_args,
            training_args,
            exp_logger_args,
        ) = parser.parse_args_into_dataclasses()

    logger.info("Training/evaluation parameters %s", training_args)

    if data_args.source_prefix is None and model_args.model_name_or_path in [
        "t5-small",
        "t5-base",
        "t5-large",
        "t5-3b",
        "t5-11b",
    ]:
        logger.warning(
            "You're running a t5 model but didn't provide a source prefix, which is expected, e.g. with "
            "`--source_prefix 'translate English to German: ' `"
        )

    return model_args, data_args, training_args, exp_logger_args


def main():

    model_args, data_args, training_args, exp_logger_args = parse_cmd_args()
    assert training_args.use_lightning

    setup_logging(training_args)

    seed_everything(training_args.seed)

    tasks = read_tasks_file(data_args)

    check_resume(model_args, training_args)

    config = AutoConfig.from_pretrained(
        model_args.config_name
        if model_args.config_name
        else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        task_specific_params=None,
        num_beams=model_args.num_beams,
        max_length=data_args.max_target_length,
        min_length=data_args.min_target_length,
        length_penalty=model_args.length_penalty,
    )
    config = add2config(model_args, data_args, config)

    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name
        if model_args.tokenizer_name
        else model_args.model_name_or_path,
        use_fast=model_args.use_fast_tokenizer,
        cache_dir=model_args.cache_dir
    )

    model = get_model(config, model_args).from_pretrained(
        model_args.model_name_or_path,
        data_args,
        config=config,
        cache_dir=model_args.cache_dir,
    )
    model.resize_token_embeddings(len(tokenizer))
    model.freeze_model_layers(model_args)
    logger.info(model)

    data = MultiTaskDataset(data_args, training_args, tokenizer, tasks)

    data_collator = MultiTaskDataCollator(
        tokenizer,
        model=model,
        label_pad_token_id=PAD_TOKEN_ID,
        pad_to_multiple_of=8 if training_args.fp16 else None,
    )
    trainer = MultiTaskTrainer(
        model=model,
        args=training_args,
        tokenizer=tokenizer,
        data_collator=data_collator,
        tasks=tasks,
        data_args=data_args,
        callback=None,
        train_dataset=data.train_dataset if training_args.do_train else None,
        eval_dataset=data.val_dataset if training_args.do_eval else None,
        test_dataset=data.test_dataset if training_args.do_predict else None,
    )

    n_gpu = torch.cuda.device_count()

    lightning_model = LightningModelWrapper(trainer)
    lightning_data = LightningDataWrapper(trainer)

    checkpoint_callback = ModelCheckpoint(
        filename='epoch_{epoch}',
        auto_insert_metric_name=False,
        dirpath=training_args.output_dir,
        save_on_train_epoch_end=True,
        every_n_epochs=training_args.every_n_epochs
    )
    logging_callback = LoggingCallback()
    lr_monitor = LearningRateMonitor(logging_interval='step')

    lightning_trainer = Trainer(
        accelerator='ddp' if n_gpu > 1 else None,
        accumulate_grad_batches=training_args.gradient_accumulation_steps,
        checkpoint_callback=True,
        callbacks=[logging_callback, checkpoint_callback, lr_monitor],
        default_root_dir=training_args.output_dir,
        max_epochs=training_args.num_train_epochs,
        gpus=n_gpu,
        check_val_every_n_epoch=training_args.every_n_epochs,
        gradient_clip_val=training_args.max_grad_norm,
        precision=16 if training_args.fp16 else 32,
        amp_level=training_args.fp16_opt_level,
        resume_from_checkpoint=training_args.lightning_checkpoint,
        plugins=DDPPlugin(find_unused_parameters=False)
    )

    if training_args.do_train:
        lightning_trainer.fit(
            model=lightning_model,
            datamodule=lightning_data
        )

    if training_args.do_predict:
        if training_args.lightning_checkpoint is not None:
            check_resume(model_args, training_args)
            lightning_model.hparams.test_checkpoint = training_args.lightning_checkpoint
            lightning_model = lightning_model.load_from_checkpoint(
                lightning_trainer.resume_from_checkpoint,
                hugginface_trainer=trainer
            )
        lightning_trainer.test(
            model=lightning_model,
            ckpt_path='None',
            dataloaders=lightning_data.test_dataloader()
        )


if __name__ == "__main__":
    main()
