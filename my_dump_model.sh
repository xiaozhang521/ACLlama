#! /bin/bash

NCCL_P2P_DISABLE=1 \
NCCL_IB_DISABLE=1 \
CUDA_VISIBLE_DEVICES=6 \
# torchrun \
# --nproc_per_node 1 \
# --nnodes 1 \
# --node_rank 0 \
# --master_addr localhost \
# --master_port 6601 \
# dump_model.py \
# --audio_model_name_or_path "/data/s50042884/huggingface_model/whisper-large-v3" \
# --text_model_name_or_path "/data/s50042884/my_code/ACLlama" \
# --data_path "/data/s50042884/huggingface_model/libri_train_update.json" \
# --fp16 True \
# --output_dir "../ACLlama_output/ACLlama_lora" \
# --num_train_epochs 20 \
# --per_device_train_batch_size 2 \
# --per_device_eval_batch_size 1 \
# --gradient_accumulation_steps 8 \
# --evaluation_strategy "no" \
# --save_strategy "steps" \
# --save_steps 100 \
# --save_total_limit 1 \
# --learning_rate 1e-5 \
# --weight_decay 0.1 \
# --adam_beta2 0.95 \
# --warmup_ratio 0.01 \
# --lr_scheduler_type "cosine" \
# --logging_steps 1 \
# --report_to "none" \
# --model_max_length 512 \
# --gradient_checkpointing True \
# --deepspeed "./config/ds_config_zero2.json" \
# --use_lora
# #--lazy_preprocess True \
# #--text_model_name_or_path "/mnt/user/zhangyuhao/LLM/llama3-instruct/llama3_1-8B/" \


python3 dump_model.py \
--audio_model_name_or_path "/data/s50042884/huggingface_model/whisper-large-v3" \
--text_model_name_or_path "/data/s50042884/huggingface_model/Llama-3.2-3B" \
--data_path "/data/s50042884/huggingface_model/libri_train_update.json" \
--fp16 True \
--output_dir "../ACLlama_output/ACLlama_lora" \
--num_train_epochs 20 \
--per_device_train_batch_size 2 \
--per_device_eval_batch_size 1 \
--gradient_accumulation_steps 8 \
--evaluation_strategy "no" \
--save_strategy "steps" \
--save_steps 100 \
--save_total_limit 1 \
--learning_rate 1e-5 \
--weight_decay 0.1 \
--adam_beta2 0.95 \
--warmup_ratio 0.01 \
--lr_scheduler_type "cosine" \
--logging_steps 1 \
--report_to "none" \
--model_max_length 512 \
--gradient_checkpointing True \
--deepspeed "./config/ds_config_zero2.json" \
--use_lora
#--lazy_preprocess True \
#--text_model_name_or_path "/mnt/user/zhangyuhao/LLM/llama3-instruct/llama3_1-8B/" \
