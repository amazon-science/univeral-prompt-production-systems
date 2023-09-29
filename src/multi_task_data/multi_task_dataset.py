"""
Author: Jonathan Pilault
This files allows us to create multi-task training, val, test datasets.
Dynamically chooses process function using get_preprocess_function depending on the dataset.
"""

import glob
import itertools
from typing import Optional, Union, List

import logging
import datasets

from datasets import load_dataset, interleave_datasets, set_caching_enabled
from transformers import PreTrainedTokenizer
from transformers import (
    MBartTokenizer, MBartTokenizerFast, MBart50Tokenizer,
    MBart50TokenizerFast, M2M100Tokenizer
)

from src.multi_task_trainer import MultiTaskTrainingArguments
from src.multi_task_data.data_args import MultiTaskDataArguments
from src.multi_task_data.utils import Task, Split, find_language_pair, scan_actions

datasets.logging.get_verbosity = lambda: logging.NOTSET
set_caching_enabled(False)

MULTILINGUAL_TOKENIZERS = [
    MBartTokenizer,
    MBartTokenizerFast,
    MBart50Tokenizer,
    MBart50TokenizerFast,
    M2M100Tokenizer
]

MAX_LENGTHS = {
    "sources": 3,
    "domains": 5,
    "prompt_input_h": 768  # condition w/ part of the hidden state, it's faster and works well
}


class MultiTaskDataset(object):
    def __init__(
        self,
        data_args: MultiTaskDataArguments,
        training_args: MultiTaskTrainingArguments,
        tokenizer: PreTrainedTokenizer,
        tasks: List[Task],
        cache_dir:  Optional[str] = None
    ):
        self.data_args = data_args
        self.training_args = training_args
        self.tokenizer = tokenizer
        self.tasks = tasks
        self.modes = list(set(itertools.chain.from_iterable(task.modes for task in self.tasks)))
        self.current_task = None
        datasets = [
            self.get_dataset(
                task, task_id, cache_dir
            ) for task_id, task in enumerate(tasks)
        ]
        if Split.train in self.modes:
            self.train_dataset = interleave_datasets([dt[Split.train] for dt in datasets
                                                      if Split.train in dt])
        if Split.val in self.modes:
            self.val_dataset = {task.task_dict_name.lower(): dt[Split.val] for dt, task in zip(datasets, tasks)
                                if Split.val in dt}
        if Split.test in self.modes:
            self.test_dataset = {task.task_dict_name.lower(): dt[Split.test] for dt, task in zip(datasets, tasks)
                                 if Split.test in dt}

    def get_dataset(self, task, task_id, cache_dir):
        split = [m.value for m in task.modes]
        if task.load_name is not None:
            # Downloading and loading a dataset from the hub.
            lang1, lang2 = find_language_pair(task.source_lang, task.target_lang)
            raw_datasets = load_dataset(
                task.load_name, name=task.config_name,
                lang1=lang1, lang2=lang2,
                cache_dir=cache_dir,
            )
            if 'val' not in raw_datasets and 'test' not in raw_datasets:
                train_testvalid = raw_datasets['train'].train_test_split(test_size=0.1)
                raw_datasets['train'] = train_testvalid['train']
                test_valid = train_testvalid["test"].train_test_split(test_size=0.5)
                raw_datasets['val'] = test_valid['train']
                raw_datasets['test'] = test_valid['test']
        else:
            data_files = {}
            for file_path in glob.glob(task.data_dir + '/*.' + task.extension):
                data_files[file_path.split("/")[-1].split(".")[0]] = file_path
            raw_datasets = load_dataset(task.extension, data_files=data_files, cache_dir=cache_dir, split=split)
            raw_datasets = {name: dt for dt, name in zip(raw_datasets, split)}

        self.column_names = raw_datasets[split[0]].column_names
        self.prefix = self.data_args.source_prefix if self.data_args.source_prefix is not None else ""  # for T5

        datasets = {}
        for mode in task.modes:
            if mode.name not in raw_datasets:
                raise ValueError("--do_%s requires a %s dataset" % (mode.name, mode.name))
            dataset = raw_datasets[mode.name]
            if mode == Split.train and self.data_args.max_train_samples is not None:
                # We will select sample from whole data if agument is specified
                dataset = dataset.select(range(self.data_args.max_train_samples))
            elif mode == Split.val and self.data_args.max_eval_samples is not None:
                # We will select sample from whole data if agument is specified
                dataset = dataset.select(range(self.data_args.max_eval_samples))
            elif mode == Split.test and self.data_args.max_predict_samples is not None:
                # We will select sample from whole data if agument is specified
                dataset = dataset.select(range(self.data_args.max_predict_samples))

            self.current_task = task

            if isinstance(self.tokenizer, tuple(MULTILINGUAL_TOKENIZERS)):
                assert task.target_lang is not None and task.source_lang is not None, (
                    f"{self.tokenizer.__class__.__name__} is a multilingual tokenizer which requires "
                    f"--source_lang and --target_lang arguments."
                )
                self.tokenizer.src_lang = task.source_tokenizer_code
                self.tokenizer.tgt_lang = task.target_tokenizer_code
            dataset = dataset.map(
                self.get_preprocess_function(task),
                batched=True,
                num_proc=self.data_args.preprocessing_num_workers,
                remove_columns=self.column_names,
                load_from_cache_file=not self.data_args.overwrite_cache,
                desc="Running tokenizer on every text in %s dataset" % mode.name
            )
            datasets[mode] = dataset
        return datasets

    def get_preprocess_function(self, task):
        if task.task_name.value in ["ABSTRACTIVE_SUMMARIZATION", "TOPIC_ABSTRACTIVE_SUMMARIZATION",
                                    "ABSTRACTIVE_PARAPHRASE", "EXTRACTIVE_SUMMARIZATION",
                                    "TOPIC_EXTRACTIVE_SUMMARIZATION"]:
            return self.summarization_preprocess
        elif task.task_name.value in ["EXTRACTIVE_ANSWERING"]:
            return self.question_answering_preprocess
        elif task.task_name.value in ["ENTITY_ANSWERING"]:
            return self.ner_preprocess
        elif task.task_name.value in ["TRANSLATE"]:
            return self.translation_preprocess
        elif task.task_name.value in ["GENERATE_RULES"]:
            return self.scan_preprocess
        else:
            raise Exception("Task has no preprocess method.")

    def translation_preprocess(self, examples):
        padding = "max_length" if self.data_args.pad_to_max_length else False
        dataset_columns = self.current_task.dataset_columns
        translation_column = dataset_columns[0]

        inputs = [ex[self.current_task.source_lang] for ex in examples[translation_column]]
        targets = [ex[self.current_task.target_lang] for ex in examples[translation_column]]
        domains = ['']
        sources = ['']

        inputs = [self.prefix + inp for inp in inputs]
        model_inputs = self.tokenizer(inputs, max_length=self.data_args.max_source_length, padding=padding, truncation=True)
        if isinstance(self.tokenizer, (MBartTokenizer, MBartTokenizerFast, MBart50Tokenizer, MBart50TokenizerFast)):
            assert self.current_task.target_tokenizer_code in self.tokenizer.additional_special_tokens
            assert self.current_task.source_tokenizer_code in self.tokenizer.additional_special_tokens
            if isinstance(self.tokenizer, (MBartTokenizer, MBart50Tokenizer)):
                decoder_start_token_id = self.tokenizer.lang_code_to_id[self.current_task.target_tokenizer_code]
            else:
                decoder_start_token_id = self.tokenizer.convert_tokens_to_ids(self.current_task.target_tokenizer_code)
            model_inputs['decoder_start_token_id'] = [decoder_start_token_id for _ in range(len(model_inputs["input_ids"]))]
        return self.preprocess_common(model_inputs, targets, padding, domains, sources)

    def question_answering_preprocess(self, examples):
        padding = "max_length" if self.data_args.pad_to_max_length else False
        dataset_columns = self.current_task.dataset_columns
        text_column = dataset_columns[0]
        question_column = dataset_columns[1]
        answer_column = dataset_columns[2]
        domains_column = dataset_columns[3]
        sources_column = dataset_columns[4] if self.current_task.source_name is None else None

        inputs = examples[text_column]
        questions = examples[question_column]
        targets = examples[answer_column]
        domains = examples[domains_column]
        sources = examples[sources_column] if sources_column else [self.current_task.source_name]
        inputs = [self.prefix + inp for inp in inputs]
        model_inputs = self.tokenizer(questions, inputs, max_length=self.data_args.max_source_length, padding=padding, truncation=True)
        return self.preprocess_common(model_inputs, targets, padding, domains, sources)

    def ner_preprocess(self, examples):
        padding = "max_length" if self.data_args.pad_to_max_length else False
        dataset_columns = self.current_task.dataset_columns
        text_column = dataset_columns[0]
        question_column = dataset_columns[1]
        answer_column = dataset_columns[2]
        domains_column = dataset_columns[3]
        sources_column = dataset_columns[4] if self.current_task.source_name is None else None

        inputs = examples[text_column]
        questions = examples[question_column]
        targets = examples[answer_column]
        domains = examples[domains_column]
        sources = examples[sources_column] if sources_column else [self.current_task.source_name]

        inputs_ = []
        questions_ = []
        targets_ = []
        domains_ = []
        for i, q, t, d in zip(inputs, questions, targets, domains):
            for q_, t_ in zip(q.split('<s>'), t.split('<s>')):
                inputs_.append(i)
                questions_.append(q_)
                targets_.append(t_)
                domains_.append(d)

        inputs = [self.prefix + inp for inp in inputs]
        model_inputs = self.tokenizer(questions, inputs, max_length=self.data_args.max_source_length, padding=padding, truncation=True)
        return self.preprocess_common(model_inputs, targets, padding, domains, sources)

    def summarization_preprocess(self, examples):
        padding = "max_length" if self.data_args.pad_to_max_length else False
        dataset_columns = self.current_task.dataset_columns
        text_column = dataset_columns[0]
        summary_column = dataset_columns[1]
        domains_column = dataset_columns[2]
        sources_column = dataset_columns[3] if self.current_task.source_name is None else None

        inputs = examples[text_column]
        targets = examples[summary_column]
        domains = examples[domains_column] if not self.data_args.remove_domains else [""]
        sources = examples[sources_column] if sources_column else [self.current_task.source_name]
        inputs = [self.prefix + inp for inp in inputs]
        model_inputs = self.tokenizer(inputs, max_length=self.data_args.max_source_length, padding=padding, truncation=True)
        return self.preprocess_common(model_inputs, targets, padding, domains, sources)

    def scan_preprocess(self, examples):
        padding = "max_length" if self.data_args.pad_to_max_length else False
        dataset_columns = self.current_task.dataset_columns
        command_column = dataset_columns[0]
        action_column = dataset_columns[1]

        inputs = examples[command_column]
        outputs = examples[action_column]

        conjunctions_actions = []
        directions = []
        for command in inputs:
            num_act = sum([1 for c in command.split() if c.lower in scan_actions])
            if "and" in command:
                conjunctions_actions.append("and %s" % num_act)
            elif "after" in command:
                conjunctions_actions.append("after %s" % num_act)
            else:
                conjunctions_actions.append("%s" % num_act)

            if "opposite" in command:
                directions.append("opposite")
            elif "around" in command:
                directions.append("around")
            else:
                directions.append(" ")

        inputs = [self.prefix + inp for inp in inputs]
        model_inputs = self.tokenizer(inputs, max_length=self.data_args.max_source_length, padding=padding, truncation=True)
        return self.preprocess_common(model_inputs, outputs, padding, directions, conjunctions_actions)

    def preprocess_common(self, model_inputs, targets, padding, domains, sources):
        num_examples = len(model_inputs["input_ids"])
        if len(sources) == 1:
            sources = [sources[0] for _ in range(num_examples)]
        if len(domains) == 1:
            domains = [domains[0] for _ in range(num_examples)]
        task_description = ' '.join(self.current_task.task_description.lower().split("_"))
        descriptors = [task_description for _ in range(num_examples)]
        # Setup the tokenizer for targets
        with self.tokenizer.as_target_tokenizer():
            labels = self.tokenizer(targets, max_length=self.data_args.max_target_length, padding=padding, truncation=True)
        # If we are padding here, replace all tokenizer.pad_token_id in the labels by -100 when we want to ignore
        # padding in the loss.
        if padding == "max_length" and self.data_args.ignore_pad_token_for_loss:
            labels["input_ids"] = [
                [(l if l != self.tokenizer.pad_token_id else -100) for l in label] for label in labels["input_ids"]
            ]
        model_inputs["labels"] = labels["input_ids"]

        sources = self.tokenizer(sources, max_length=MAX_LENGTHS['sources'],
                                 padding=padding, add_special_tokens=False, truncation=True)
        domains = self.tokenizer(domains, max_length=MAX_LENGTHS['domains'],
                                 padding=padding, add_special_tokens=False, truncation=True)
        descriptors = self.tokenizer(descriptors, max_length=self.data_args.max_descriptor_length,
                                     padding=padding, add_special_tokens=False, truncation=True)

        model_inputs["descriptors"] = [source + domain + description for source, domain, description
                                       in zip(sources["input_ids"], domains["input_ids"], descriptors["input_ids"])]
        model_inputs["descriptors_attention_mask"] = [source + domain + description for source, domain, description
                                       in zip(sources["attention_mask"], domains["attention_mask"], descriptors["attention_mask"])]
        return model_inputs