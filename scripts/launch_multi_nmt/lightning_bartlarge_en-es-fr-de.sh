#!/bin/sh
cd ../.. && /home/ubuntu/anaconda3/envs/pytorch_p36/bin/python3 -u run_lightning.py --do_train --do_eval --use_lightning \
	--model_variation bart --model_name_or_path facebook/mbart-large-cc25 \
	--output_dir /home/ubuntu/Models/prompt_rule_gen/bartlarge_en-es_fr-de --report_to none \
	--tasks_file_path ./task_files/nmt_en-es-fr-de_parallel.yml --num_train_epochs 150 --disable_exp_logger --learning_rate 3e-5 \
	--freeze_encoder_layers 13-14 --freeze_decoder_layers 13-14 --save_total_limit 1 --every_n_epochs 25 \
	--preprocessing_num_workers 16 --max_source_length 70 --max_target_length 70 --max_eval_samples 1000 --max_train_samples 500 \
	--seed 1 --num_beams 3 --val_max_target_length 70 --length_penalty 1 \
	--predict_with_generate --pad_to_max_length --ddp_find_unused_parameters False --warmup_ratio 0.1 \
	--per_device_train_batch_size 8 --per_device_eval_batch_size 16 --gradient_accumulation_steps 3 --fp16 \
	--label_smoothing_factor 0.1 --max_grad_norm 0.1 --lr_scheduler_type polynomial

