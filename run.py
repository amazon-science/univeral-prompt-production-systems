import re
import os
import sys
import logging
import comet_ml
import warnings
from pathlib import Path

from transformers.utils import logging
from transformers import (
    HfArgumentParser,
    set_seed,
    AutoTokenizer,
    AutoConfig
)
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR

from src.multi_task_model.parent_model import get_model
from src.multi_task_trainer import MultiTaskTrainer, MultiTaskTrainingArguments
from src.multi_task_data.multi_task_data_collator import MultiTaskDataCollator
from src.multi_task_data.multi_task_dataset import MultiTaskDataset
from src.multi_task_data.utils import read_tasks_file
from src.utils.train_utils import add2config
from src.utils.experiment_logger import ExperimentLoggerArguments
from src.utils.experiment_logger import MyCometCallback
from src.utils.metrics import PAD_TOKEN_ID
from src.multi_task_data.data_args import MultiTaskDataArguments
from src.multi_task_model.model_args import ModelArguments

logger = logging.get_logger(__name__)
warnings.filterwarnings("ignore")


def setup_logging(training_args):
    logging.enable_default_handler()
    logging.enable_explicit_format()
    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
        training_args.local_rank,
        training_args.device,
        training_args.n_gpu,
        bool(training_args.local_rank != -1),
        training_args.fp16,
    )
    logging.set_verbosity(logging.ERROR)


def get_callback(exp_logger_args, previous_experiment):
    if not exp_logger_args.disable_exp_logger:
        callback = MyCometCallback(exp_logger_args, previous_experiment)
    else:
        os.environ['COMET_DISABLE_AUTO_LOGGING'] = '1'
        callback = None
    return callback


def check_resume(model_args, training_args):
    if (
        not training_args.overwrite_output_dir
        and not os.path.isdir(model_args.model_name_or_path)
        and os.path.isdir(training_args.output_dir)
    ):

        ordering_and_checkpoint_path = []

        glob_checkpoints = [
            str(x)
            for x in Path(training_args.output_dir).glob(f"{PREFIX_CHECKPOINT_DIR}-*")
        ]

        for path in glob_checkpoints:
            regex_match = re.match(f".*{PREFIX_CHECKPOINT_DIR}-([0-9]+)", path)
            if regex_match and regex_match.groups():
                ordering_and_checkpoint_path.append(
                    (int(regex_match.groups()[0]), path)
                )

        checkpoints_sorted = sorted(ordering_and_checkpoint_path)
        checkpoints_sorted = [checkpoint[1] for checkpoint in checkpoints_sorted]
        if checkpoints_sorted:
            model_args.model_name_or_path = checkpoints_sorted[-1]


def do_train(trainer, training_args, model_args):
    resume_from_checkpoint = model_args.model_name_or_path \
        if model_args.model_name_or_path and os.path.isdir(model_args.model_name_or_path) else None
    train_result = trainer.train(
        resume_from_checkpoint=resume_from_checkpoint
    )
    latest_checkpoint_name = os.path.join(
        training_args.output_dir, f"{PREFIX_CHECKPOINT_DIR}-{trainer.state.global_step}"
    )
    logger.info(f"Using {latest_checkpoint_name} to save model.")
    trainer.save_model(output_dir=latest_checkpoint_name)
    trainer.log_metrics("train", train_result.metrics)
    trainer.save_metrics("train", train_result.metrics)
    trainer.save_state()


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

    return model_args, data_args, training_args, exp_logger_args


def main():

    model_args, data_args, training_args, exp_logger_args = parse_cmd_args()
    assert not training_args.use_lightning

    setup_logging(training_args)

    set_seed(training_args.seed)

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
        length_penalty=model_args.length_penalty
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
    model._log_params()
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
        callback=get_callback(exp_logger_args, getattr(config, "experiment_id", None)),
        train_dataset=data.train_dataset if training_args.do_train else None,
        eval_dataset=data.val_dataset if training_args.do_eval else None,
        test_dataset=data.test_dataset if training_args.do_predict else None,
    )

    if training_args.do_train:
        do_train(trainer, training_args, model_args)

    if not training_args.do_train and (training_args.do_eval or training_args.do_predict):
        resume_from_checkpoint = model_args.model_name_or_path \
            if model_args.model_name_or_path and os.path.isdir(model_args.model_name_or_path) else None

        if resume_from_checkpoint:
            trainer.model = trainer.model.from_pretrained(resume_from_checkpoint, config=config)
            trainer.model = trainer.model.to(trainer.args.device)

    if training_args.do_eval:
        output = trainer.evaluate(
            max_length=data_args.val_max_target_length, num_beams=model_args.num_beams
        )

    if training_args.do_predict:
        output = trainer.predict(
            max_length=data_args.val_max_target_length, num_beams=model_args.num_beams
        )


def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    main()
