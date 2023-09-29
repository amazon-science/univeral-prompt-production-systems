#!/bin/sh
cd ../.. && /home/ubuntu/anaconda3/envs/pytorch_p36/bin/python3 -u run_lightning.py --do_train --do_eval --use_lightning \
      --model_variation bart_prefix --model_name_or_path facebook/bart-large --output_dir /home/ubuntu/Models/prompt_rule_gen/bartlarge_prefix_xsum_summ \
      --tasks_file_path ./task_files/xsum_trg_summ.yml --num_train_epochs 30 --disable_exp_logger --learning_rate 5e-5 \
      --evaluation_strategy epoch --freeze_encoder_layers 0-11 --freeze_decoder_layers 0-11 --save_strategy epoch --save_total_limit 1 \
      --preprocessing_num_workers 16 --max_source_length 768 --max_target_length 60 \
      --max_descriptor_length 10 --seed 101 --num_beams 6  --val_max_target_length 62 --max_eval_samples 1000 --length_penalty 2 \
      --predict_with_generate --pad_to_max_length --ddp_find_unused_parameters False --warmup_ratio 0 \
      --num_prompts 200 --prompt_layers 2 --per_device_train_batch_size 8 --per_device_eval_batch_size 8 --gradient_accumulation_steps 2 --fp16
