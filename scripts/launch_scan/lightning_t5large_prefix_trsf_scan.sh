#!/bin/sh
cd ../.. && /home/ubuntu/anaconda3/envs/pytorch_p36/bin/python3 -u run_lightning.py --do_train --do_eval --use_lightning \
        --model_variation t5_prefix_transformer --model_name_or_path facebook/bart-large \
        --config_name facebook/bart-large-xsum --tokenizer_name facebook/bart-large-xsum \
        --output_dir /home/ubuntu/Models/prompt_rule_gen/bartlarge_prefix_trsf_xsum_summ --report_to none \
        --tasks_file_path ./task_files/xsum_trg_summ.yml --num_train_epochs 30 --disable_exp_logger --learning_rate 3e-5 \
        --freeze_encoder_layers 0-10 --freeze_decoder_layers 0-10 --save_total_limit 1 \
        --preprocessing_num_workers 16 --max_source_length 50 --max_target_length 200 \
        --max_descriptor_length 10 --seed 1 --num_beams 3 --val_max_target_length 200 --max_eval_samples 1000 --length_penalty 1 \
        --predict_with_generate --pad_to_max_length --ddp_find_unused_parameters False --warmup_ratio 0.0 \
        --num_prompts 100 --prompt_layers 1 --per_device_train_batch_size 3 --per_device_eval_batch_size 12 --gradient_accumulation_steps 8 --fp16 \
        --prompt_hidden_condition --prompt_attention_heads 16 --tok_k_rules 4 --prompt_dropout 0.1 \
        --prompt_d_model 1024 --prompt_ffn_dim 4096 \
        --label_smoothing_factor 0.2 --weight_decay 0.01 --max_grad_norm 0.1 --lr_scheduler_type polynomial \
        --encoder_self_prefix_layer_ids 0 1 10 11 --decoder_self_prefix_layer_ids 0 1 10 11 --decoder_cross_prefix_layer_ids 0 1 10 11