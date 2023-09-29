#!/bin/sh
# using 4 16G GPUs
cd ../.. && /home/ubuntu/anaconda3/envs/pytorch_p36/bin/python3 -u run_lightning.py --do_train --do_eval --use_lightning \
      --model_variation bart --model_name_or_path /home/ubuntu/Models/prompt_rule_gen/bartlarge_xsum_inter/epoch_5 \
      --config_name facebook/bart-large-xsum --tokenizer_name facebook/bart-large-xsum \
      --output_dir /home/ubuntu/Models/prompt_rule_gen/bartlarge_xsum_summ --report_to none \
      --tasks_file_path ./task_files/xsum_trg_summ.yml --num_train_epochs 5 --disable_exp_logger \
      --evaluation_strategy epoch --learning_rate 3e-5 --seed 1 \
      --preprocessing_num_workers 16 --max_source_length 768 --max_target_length 62 \
      --num_beams 3 --val_max_target_length 62 --max_eval_samples 500 --length_penalty 1 \
      --predict_with_generate --pad_to_max_length --ddp_find_unused_parameters False --warmup_ratio 0.01 \
      --per_device_train_batch_size 2 --per_device_eval_batch_size 4 --gradient_accumulation_steps 16 --fp16 \
      --label_smoothing_factor 0.1 --weight_decay 0.01 --max_grad_norm 0.1 --lr_scheduler_type polynomial
