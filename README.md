<h1 align="center">
<p>On Conditional and Compositional Language Model Differentiable Prompting --- PRompt Production System (PRoPS) code</p>
</h1>

<p align="center">
    <a>
        <img alt="Python" src="https://img.shields.io/badge/Python-3.7-blue">
    </a>
    <a>
        <img alt="Python" src="https://img.shields.io/badge/Pytorch-1.9-blue">
    </a>
    <a>
        <img alt="Python" src="https://img.shields.io/badge/Release-1.0.0-blue">
    </a>
       <a>
        <img alt="Python" src="https://img.shields.io/badge/License-MIT-blue">
    </a>
</p>


The source code uses the [huggingface implementation](https://github.com/huggingface/transformers) of transformers adapted for multitask training. The code in this repo replicates results of the paper `On Conditional and Compositional Language Model Differentiable Prompting`. Our paper was accepted at IJCAI 2023 (https://arxiv.org/abs/2307.01446).


## Requirements

- Python 3.7
- Pytorch 1.9
- Huggingface transformers 4.9.1 

Note: Newer versions of the requirements should work, but was not tested.

### Using a virual environment 

```bash
# Create a virtual environment
python3.7 -m venv prompt_xp
source prompt_xp/bin/activate

# Install the requirements
pip install requirements.txt
# If you are using an environment that have torch already installed use "requirements.txt"
```

### Run files
There are two run scripts that are equivalen:
1. `run.py` that runs the huggingface trainer
2. `run_lightning.py` that uses the huggingface trainer to set-up the data and model but uses the lightning trainer
We found the lightning model to be faster


### Run scripts
All run scripts can be found in the `.\scripts\launch_{cnndm, xsum, scan, multi_nmt}` directories. For example:

```bash
chmod +x lightning_bartlarge_prefix_upps_xsum_summ.sh
./lightning_bartlarge_prefix_upps_xsum_summ.sh
```

To run the out-of-topic `xsum_news` dataset, you need to replace 

```
--tasks_file_path ./task_files/xsum_trg_summ.yml
```

by 

```
--tasks_file_path ./task_files/xsum_trg_summ_news.yml
``` 

in the launch scripts.

All scripts with the word `inter` in the namenclature are used to pretrain the prompt generator on intermediate tasks. 
See `\task_files\cnndm_inter_all.yml` and `\task_files\xsum_inter_all.yml` for the list of tasks used. 
Make sure to change your `--output_dir`.

Once trained, this model is used for 0, 5, 50, 100, 200, 500 shot experiments using `--max_train_samples`.
Note that I increase `--max_source_length` to 768. Make sure to add path of intermediate task pretrained model checkpoint in `--model_name_or_path`.
See `lightning_bartlarge_xsum_summ_low_res.sh` for an example.
For an example of how to test the intermediate task finetuned model, see `/scripts/launch_cnndm/lightning_bartlarge_prefix_cnndm_summ_test.sh`.
All run scripts assume that the data is in `/home/Datasets` and that the models/log files are saved in `/home/Models`. 
If that's not the case, you will need to manually change paths in task_files folder for `yml` files in the 'task_files' directory and in the `sh` files in the `scripts` directory.

### Downloading Datasets

Datasets below require additional preprocessing (see preprocessing section). You can skip the next two sections and find the preprocessed data links in the "Preprocessed Data" section

| Type | Link |
| ------ | ------ |
| XSUM and XSUM-news | [lisa_colab/xsum_data](https://worksheets.codalab.org/bundles/0x58f85171b43f4e61bf411c35faab369d)|
| Topic CNN-DM  | [Amazon Alexa S3 link](https://liuca-backup.s3.us-east-2.amazonaws.com/khalil/public-dataset/subtopic_first.tar) |
| Extractive CNN-DM | [Microsoft download](https://www.microsoft.com/en-us/research/project/newsqa-dataset/download/)|
| NewsQA | [Google Drive](https://drive.google.com/uc?id=1SsenAqbK1wmvd_1oWgAT1fRtLcA9rglS) |
| SCAN | [Github link](https://github.com/brendenlake/SCAN) |


Translation and semantic parsing datasets can directly accessed via the [Huggingface datahub](https://huggingface.co/datasets).

### Preprocessing Data
Scripts to preprocess data can be found in `./scripts/preprocessing`. 
You will need to download domain information or article types (sports, business, politics, ...) from here: 
TODO: please make this link public

Preprocessing files mainly put all data in a `csv` format while putting together the meta data (article label, news outlet). 
However, for the new `xsum_ner, xsum_paraphrase and xsum_extractive` datasets that we used to test task composition, further data processing was done:

* `xsum_ner` requires extracting most common entities, which is used as an output, and the corresponding entity label, used as an input.
* `xsum_paraphrase` requires a pretrained translation model. We randomly extract passages from the input article and perform backtranslation (en>ar>fr>zh>en) to create abstractive passages.
* `xsum_extractive` requires using BertScores to extract the most relevant sentence.

### Preprocessed Data
All preprocessed datasets can be found here: 
TODO: please make this link public.

### Usage 
```
usage: run.py [-h] --model_name_or_path MODEL_NAME_OR_PATH
              [--model_variation MODEL_VARIATION] [--config_name CONFIG_NAME]
              [--tokenizer_name TOKENIZER_NAME]
              [--length_penalty LENGTH_PENALTY] [--num_beams NUM_BEAMS]
              [--cache_dir CACHE_DIR]
              [--freeze_encoder_layers FREEZE_ENCODER_LAYERS]
              [--freeze_decoder_layers FREEZE_DECODER_LAYERS]
              [--unfreeze_up_proj [UNFREEZE_UP_PROJ]]
              [--unfreeze_q_proj [UNFREEZE_Q_PROJ]]
              [--unfreeze_v_proj [UNFREEZE_V_PROJ]]
              [--unfreeze_attn [UNFREEZE_ATTN]] [--no_use_fast_tokenizer]
              [--use_fast_tokenizer [USE_FAST_TOKENIZER]]
              [--num_prompts NUM_PROMPTS] [--prompt_layers PROMPT_LAYERS]
              [--prompt_attention_heads PROMPT_ATTENTION_HEADS]
              [--prompt_d_model PROMPT_D_MODEL]
              [--prompt_ffn_dim PROMPT_FFN_DIM]
              [--prompt_dropout PROMPT_DROPOUT]
              [--prompt_hidden_condition [PROMPT_HIDDEN_CONDITION]]
              [--encoder_self_prefix_layer_ids ENCODER_SELF_PREFIX_LAYER_IDS [ENCODER_SELF_PREFIX_LAYER_IDS ...]]
              [--decoder_self_prefix_layer_ids DECODER_SELF_PREFIX_LAYER_IDS [DECODER_SELF_PREFIX_LAYER_IDS ...]]
              [--decoder_cross_prefix_layer_ids DECODER_CROSS_PREFIX_LAYER_IDS [DECODER_CROSS_PREFIX_LAYER_IDS ...]]
              [--tok_k_rules TOK_K_RULES]
              [--n_enc_recurrence N_ENC_RECURRENCE]
              [--n_dec_recurrence N_DEC_RECURRENCE]
              [--dataset_name DATASET_NAME]
              [--dataset_config_name DATASET_CONFIG_NAME]
              [--tasks_file_path TASKS_FILE_PATH] [--text_column TEXT_COLUMN]
              [--summary_column SUMMARY_COLUMN]
              [--overwrite_cache [OVERWRITE_CACHE]]
              [--preprocessing_num_workers PREPROCESSING_NUM_WORKERS]
              [--max_source_length MAX_SOURCE_LENGTH]
              [--max_target_length MAX_TARGET_LENGTH]
              [--min_target_length MIN_TARGET_LENGTH]
              [--max_descriptor_length MAX_DESCRIPTOR_LENGTH]
              [--val_max_target_length VAL_MAX_TARGET_LENGTH]
              [--pad_to_max_length [PAD_TO_MAX_LENGTH]]
              [--forced_bos_token FORCED_BOS_TOKEN]
              [--max_train_samples MAX_TRAIN_SAMPLES]
              [--max_eval_samples MAX_EVAL_SAMPLES]
              [--max_predict_samples MAX_PREDICT_SAMPLES]
              [--no_ignore_pad_token_for_loss]
              [--ignore_pad_token_for_loss [IGNORE_PAD_TOKEN_FOR_LOSS]]
              [--source_prefix SOURCE_PREFIX]
              [--remove_domains [REMOVE_DOMAINS]] --output_dir OUTPUT_DIR
              [--overwrite_output_dir [OVERWRITE_OUTPUT_DIR]]
              [--do_train [DO_TRAIN]] [--do_eval [DO_EVAL]]
              [--do_predict [DO_PREDICT]]
              [--evaluation_strategy {no,steps,epoch}]
              [--prediction_loss_only [PREDICTION_LOSS_ONLY]]
              [--per_device_train_batch_size PER_DEVICE_TRAIN_BATCH_SIZE]
              [--per_device_eval_batch_size PER_DEVICE_EVAL_BATCH_SIZE]
              [--per_gpu_train_batch_size PER_GPU_TRAIN_BATCH_SIZE]
              [--per_gpu_eval_batch_size PER_GPU_EVAL_BATCH_SIZE]
              [--gradient_accumulation_steps GRADIENT_ACCUMULATION_STEPS]
              [--eval_accumulation_steps EVAL_ACCUMULATION_STEPS]
              [--learning_rate LEARNING_RATE] [--weight_decay WEIGHT_DECAY]
              [--adam_beta1 ADAM_BETA1] [--adam_beta2 ADAM_BETA2]
              [--adam_epsilon ADAM_EPSILON] [--max_grad_norm MAX_GRAD_NORM]
              [--num_train_epochs NUM_TRAIN_EPOCHS] [--max_steps MAX_STEPS]
              [--lr_scheduler_type {linear,cosine,cosine_with_restarts,polynomial,constant,constant_with_warmup}]
              [--warmup_ratio WARMUP_RATIO] [--warmup_steps WARMUP_STEPS]
              [--log_level {debug,info,warning,error,critical,passive}]
              [--log_level_replica {debug,info,warning,error,critical,passive}]
              [--no_log_on_each_node] [--log_on_each_node [LOG_ON_EACH_NODE]]
              [--logging_dir LOGGING_DIR]
              [--logging_strategy {no,steps,epoch}]
              [--logging_first_step [LOGGING_FIRST_STEP]]
              [--logging_steps LOGGING_STEPS]
              [--save_strategy {no,steps,epoch}] [--save_steps SAVE_STEPS]
              [--save_total_limit SAVE_TOTAL_LIMIT]
              [--save_on_each_node [SAVE_ON_EACH_NODE]] [--no_cuda [NO_CUDA]]
              [--seed SEED] [--fp16 [FP16]] [--fp16_opt_level FP16_OPT_LEVEL]
              [--fp16_backend {auto,amp,apex}]
              [--fp16_full_eval [FP16_FULL_EVAL]] [--local_rank LOCAL_RANK]
              [--tpu_num_cores TPU_NUM_CORES]
              [--tpu_metrics_debug [TPU_METRICS_DEBUG]] [--debug DEBUG]
              [--dataloader_drop_last [DATALOADER_DROP_LAST]]
              [--eval_steps EVAL_STEPS]
              [--dataloader_num_workers DATALOADER_NUM_WORKERS]
              [--past_index PAST_INDEX] [--run_name RUN_NAME]
              [--disable_tqdm DISABLE_TQDM] [--no_remove_unused_columns]
              [--remove_unused_columns [REMOVE_UNUSED_COLUMNS]]
              [--label_names LABEL_NAMES [LABEL_NAMES ...]]
              [--load_best_model_at_end [LOAD_BEST_MODEL_AT_END]]
              [--metric_for_best_model METRIC_FOR_BEST_MODEL]
              [--greater_is_better GREATER_IS_BETTER]
              [--ignore_data_skip [IGNORE_DATA_SKIP]]
              [--sharded_ddp SHARDED_DDP] [--deepspeed DEEPSPEED]
              [--label_smoothing_factor LABEL_SMOOTHING_FACTOR]
              [--adafactor [ADAFACTOR]] [--group_by_length [GROUP_BY_LENGTH]]
              [--length_column_name LENGTH_COLUMN_NAME]
              [--report_to REPORT_TO [REPORT_TO ...]]
              [--ddp_find_unused_parameters DDP_FIND_UNUSED_PARAMETERS]
              [--no_dataloader_pin_memory]
              [--dataloader_pin_memory [DATALOADER_PIN_MEMORY]]
              [--no_skip_memory_metrics]
              [--skip_memory_metrics [SKIP_MEMORY_METRICS]]
              [--use_legacy_prediction_loop [USE_LEGACY_PREDICTION_LOOP]]
              [--push_to_hub [PUSH_TO_HUB]]
              [--resume_from_checkpoint RESUME_FROM_CHECKPOINT]
              [--push_to_hub_model_id PUSH_TO_HUB_MODEL_ID]
              [--push_to_hub_organization PUSH_TO_HUB_ORGANIZATION]
              [--push_to_hub_token PUSH_TO_HUB_TOKEN]
              [--mp_parameters MP_PARAMETERS]
              [--sortish_sampler [SORTISH_SAMPLER]]
              [--predict_with_generate [PREDICT_WITH_GENERATE]]
              [--use_lightning [USE_LIGHTNING]]
              [--lightning_checkpoint LIGHTNING_CHECKPOINT]
              [--every_n_epochs EVERY_N_EPOCHS]
              [--disable_exp_logger [DISABLE_EXP_LOGGER]]
              [--exp_workspace_name EXP_WORKSPACE_NAME]
              [--exp_project_name EXP_PROJECT_NAME]
              [--exp_logger_api_key EXP_LOGGER_API_KEY] [--exp_name EXP_NAME]

optional arguments:
  -h, --help            show this help message and exit
 --model_name_or_path MODEL_NAME_OR_PATH
                        Path to pretrained model or model identifier from: CA-
                        MTL-base, CA-MTL-large, bert-base-cased bert-base-
                        uncased, bert-large-cased, bert-large-uncased
  --data_dir DATA_DIR   The input data dir. Should contain the .tsv files (or
                        other data files) for the task.
  --tasks_file_path     yml file containing task(s) information
                        The task file that contains the tasks to train on. If
                        None all tasks will be used
  --overwrite_cache     Overwrite the cached training and evaluation sets
  --max_seq_length MAX_SEQ_LENGTH
                        The maximum total input sequence length after
                        tokenization. Sequences longer than this will be
                        truncated, sequences shorter will be padded.
  --output_dir OUTPUT_DIR
                        The output directory where the model predictions and
                        checkpoints will be written.
  --overwrite_output_dir
                        Overwrite the content of the output directory.Use this
                        to continue training if output_dir points to a
                        checkpoint directory.
  --do_train            Whether to run training.
  --do_eval             Whether to run eval on the dev set.
  --do_predict          Whether to run predictions on the test set.
  --evaluate_during_training
                        Run evaluation during training at each logging step.
  --per_device_train_batch_size PER_DEVICE_TRAIN_BATCH_SIZE
                        Batch size per GPU/TPU core/CPU for training.
  --per_device_eval_batch_size PER_DEVICE_EVAL_BATCH_SIZE
                        Batch size per GPU/TPU core/CPU for evaluation.
  --per_gpu_train_batch_size PER_GPU_TRAIN_BATCH_SIZE
                        Deprecated, the use of `--per_device_train_batch_size`
                        is preferred. Batch size per GPU/TPU core/CPU for
                        training.
  --per_gpu_eval_batch_size PER_GPU_EVAL_BATCH_SIZE
                        Deprecated, the use of `--per_device_eval_batch_size`
                        is preferred.Batch size per GPU/TPU core/CPU for
                        evaluation.
  --gradient_accumulation_steps GRADIENT_ACCUMULATION_STEPS
                        Number of updates steps to accumulate before
                        performing a backward/update pass.
  --learning_rate LEARNING_RATE
                        The initial learning rate for Adam.
  --weight_decay WEIGHT_DECAY
                        Weight decay if we apply some.
  --adam_epsilon ADAM_EPSILON
                        Epsilon for Adam optimizer.
  --max_grad_norm MAX_GRAD_NORM
                        Max gradient norm.
  --num_train_epochs NUM_TRAIN_EPOCHS
                        Total number of training epochs to perform.
  --max_steps MAX_STEPS
                        If > 0: set total number of training steps to perform.
                        Override num_train_epochs.
  --warmup_steps WARMUP_STEPS
                        Linear warmup over warmup_steps.
  --logging_dir LOGGING_DIR
                        Tensorboard log dir.
  --logging_first_step  Log and eval the first global_step
  --logging_steps LOGGING_STEPS
                        Log every X updates steps.
  --save_steps SAVE_STEPS
                        Save checkpoint every X updates steps.
  --save_total_limit SAVE_TOTAL_LIMIT
                        Limit the total amount of checkpoints.Deletes the
                        older checkpoints in the output_dir. Default is
                        unlimited checkpoints
  --no_cuda             Do not use CUDA even when it is available
  --seed SEED           random seed for initialization
  --fp16                Whether to use 16-bit (mixed) precision (through
                        NVIDIA apex) instead of 32-bit
  --fp16_opt_level FP16_OPT_LEVEL
                        For fp16: Apex AMP optimization level selected in
                        ['O0', 'O1', 'O2', and 'O3'].See details at
                        https://nvidia.github.io/apex/amp.html
  --local_rank LOCAL_RANK
                        For distributed training: local_rank
  --tpu_num_cores TPU_NUM_CORES
                        TPU: Number of TPU cores (automatically passed by
                        launcher script)
```

Since our code is based on the [huggingface implementation](https://github.com/huggingface/transformers). 
All parameters are described in their [documentation](https://huggingface.co/transformers/main_classes/trainer.html?highlight=trainer).


## How do I cite?
```
@inproceedings{ijcai2023p0460,
  title     = {On Conditional and Compositional Language Model Differentiable Prompting},
  author    = {Pilault, Jonathan and Liu, Can and Bansal, Mohit and Dreyer, Markus},
  booktitle = {Proceedings of the Thirty-Second International Joint Conference on
               Artificial Intelligence, {IJCAI-23}},
  publisher = {International Joint Conferences on Artificial Intelligence Organization},
  editor    = {Edith Elkind},
  pages     = {4136--4144},
  year      = {2023},
  month     = {8},
  note      = {Main Track},
  doi       = {10.24963/ijcai.2023/460},
  url       = {https://doi.org/10.24963/ijcai.2023/460},
}
```

## Contact and Contribution
For any question or request, please create a Github issue in this repository.
