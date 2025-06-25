#! /bin/bash

device=(0,1,2,3,4,5,6,7)
gpu_num=8
# device=(0)
# gpu_num=1


export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
# export TORCH_COMPILE=0
# export DISABLE_TORCH_COMPILE=1
# export NCCL_DEBUG=INFO
# export TORCH_DISTRIBUTED_DEBUG=DETAIL

# output_tag="../ACLlama_output/ACLlama_lora_finetune"
# output_tag="../ACLlama_output/ACLlama_lora_finetune_add_contrastive_loss"
# output_tag="../ACLlama_output/ACLlama_lora_finetune_add_contrastive_loss_v1"
# output_tag="../ACLlama_output/ACLlama_lora_finetune_add_clip_contrastive_loss"
# output_tag="../ACLlama_output/ACLlama_lora_finetune_add_clip_contrastive_loss_audio_caption_300epoch"
# output_tag="../ACLlama_output/ACLlama_lora_finetune_add_clip_contrastive_loss_audio_caption_300epoch_large_batch"
# output_tag="../ACLlama_output/ACLlama_encoder_stage2"
# output_tag="../ACLlama_output/ACLlama_encoder_stage2_from_contrastive_asr_loss_base_stage2"
# output_tag="../ACLlama_output/ACLlama_encoder_stage1_from_contrastive_asr_loss_base_stage2"
# output_tag="../ACLlama_output/ACLlama_encoder_stage1_50epoch"
output_tag="../ACLlama_output/ACLlama_encoder_stage2_resume"
# output_tag="../ACLlama_output/ACLlama_encoder_stage1_wo_asr_head_copy"

if [[ ! -e ${output_tag} ]]; then
    mkdir -p ${output_tag}
fi
code_save_path=$output_tag"/code_save"
if [[ ! -e ${code_save_path} ]]; then
    mkdir -p ${save_dir}
fi

export CUDA_VISIBLE_DEVICES=${device[@]}
cmd="torchrun
    --nproc_per_node 8
    --nnodes 1
    --node_rank 0
    --master_addr localhost
    --master_port 6881
    finetune_acllama.py
    --audio_model_name_or_path "/data/s50042884/huggingface_model/whisper-large-v3"
    --text_model_name_or_path "../ACLlama_output/ACLlama_model_ori_zhang"
    --data_path "/data/s50042884/my_code/data/libri_train_update.json"
    --output_dir ${output_tag}
    --num_train_epochs 30
    --fp16 True
    --per_device_train_batch_size 16
    --per_device_eval_batch_size 1
    --gradient_accumulation_steps 8
    --evaluation_strategy "no"
    --save_strategy "steps"
    --save_steps 400
    --save_total_limit 100
    --learning_rate 3e-5
    --weight_decay 0.1
    --adam_beta2 0.95
    --warmup_ratio 0.01
    --lr_scheduler_type "cosine"
    --logging_steps 1
    --report_to "none"
    --model_max_length 512
    --gradient_checkpointing True
    --deepspeed "./config/ds_config_zero2.json"
    --use_lora"

    # --per_device_train_batch_size 32 \
    # --per_device_eval_batch_size 1 \
    # --gradient_accumulation_steps 8 \
    # --data_path "/data/s50042884/my_code/data/audio_caps_formatted.json"
    # --deepspeed "./config/ds_config_zero2.json" \
    # --fp16 True \
    # --bf16 True \
    # --num_train_epochs 40 \

script_path=$(realpath "$0")
script_dir=$(dirname "$(realpath "$0")")
cp ${script_path} ${save_dir}/
cp ./finetune_acllama.py ${code_save_path}
cp ./ACLlama_el.py ${code_save_path}
cp ./dump_model.py ${code_save_path}
cp ./my_dump_model.sh ${code_save_path}

timestamp=$(date +"%Y%m%d_%H%M%S")
save_cmd="${output_tag}/train_${timestamp}.log"
echo $cmd
eval $cmd 2>&1 | tee $save_cmd

# python3 finetune_acllama.py \
#     --audio_model_name_or_path "/data/s50042884/huggingface_model/whisper-large-v3" \
#     --text_model_name_or_path "/data/s50042884/my_code/ACLlama_output/ACLlama_lora" \
#     --data_path "/data/s50042884/huggingface_model/libri_train_update.json" \
#     --fp16 True \
#     --output_dir "../ACLlama_output/ACLlama_lora" \
#     --num_train_epochs 40 \
#     --per_device_train_batch_size 4 \
#     --per_device_eval_batch_size 1 \
#     --gradient_accumulation_steps 64 \
#     --evaluation_strategy "no" \
#     --save_strategy "steps" \
#     --save_steps 100 \
#     --save_total_limit 1 \
#     --learning_rate 1e-5 \
#     --weight_decay 0.1 \
#     --adam_beta2 0.95 \
#     --warmup_ratio 0.01 \
#     --lr_scheduler_type "cosine" \
#     --logging_steps 1 \
#     --report_to "none" \
#     --model_max_length 512 \
#     --gradient_checkpointing True \
#     --deepspeed "./config/ds_config_zero2.json" \
#     --use_lora