#!/bin/sh
cd ../.. && /home/ubuntu/anaconda3/envs/pytorch_p36/bin/python3 -u run_lightning.py --do_predict --use_lightning \
	--model_variation bart_prefix --model_name_or_path /home/ubuntu/Models/prompt_rule_gen/bartlarge_prefix_inter_cnndm/epoch_3 \
	--output_dir /home/ubuntu/Models/prompt_rule_gen/bartlarge_prefix_inter_cnndm --report_to none \
	--tasks_file_path ./task_files/cnndm_topics_summ.yml --num_train_epochs 5 --disable_exp_logger --learning_rate 3e-5 \
	--freeze_encoder_layers 0-11 --freeze_decoder_layers 0-11 --save_total_limit 1 \
	--preprocessing_num_workers 16 --max_source_length 512 --max_target_length 62 \
	--max_descriptor_length 10 --seed 1 --num_beams 6 --val_max_target_length 62 --max_eval_samples 1000 --length_penalty 1 \
	--predict_with_generate --pad_to_max_length --ddp_find_unused_parameters False --warmup_ratio 0.1 \
	--num_prompts 100 --prompt_layers 1 --per_device_train_batch_size 8 --per_device_eval_batch_size 24 --gradient_accumulation_steps 3 --fp16 \
	--label_smoothing_factor 0.2 --weight_decay 0.01 --max_grad_norm 0.1 --lr_scheduler_type polynomial
