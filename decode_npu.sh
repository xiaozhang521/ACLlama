#! /bin/bash

device=(0,1,2,3,4,5,6,7)
gpu_num=8
# device=(0)
# gpu_num=1


export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
export TORCH_COMPILE=0
export DISABLE_TORCH_COMPILE=1
export NCCL_DEBUG=INFO
export TORCH_DISTRIBUTED_DEBUG=DETAIL

output_model_root="/data/s50042884/my_code/ACLlama_output/"
output_model_path=${output_model_root}"whisper-large-v3-and-Llama-3.2-3B-align/"

export ASCEND_VISIBLE_DEVICES=${device[@]}
export ASCEND_DEVICE_ID=${device[@]}
cmd="python llama3.1_peft_lora_predict_npu.py
    --eval_data "/data/s50042884/huggingface_model/libri_test.json"
    --audio_tower "/data/s50042884/huggingface_model/whisper-large-v3"
    --base_model_path ${output_model_root}"/ACLlama_lora/"
    --peft_model_id ${output_model_path}"/checkpoint-2/"
    --clean_out_path ${output_model_path}"/test_clean.txt"
    --other_out_path ${output_model_path}"/test_other.txt"
    --num_threads ${gpu_num}"

save_cmd="${output_model_path}/inference.log"
echo $cmd
eval $cmd 2>&1 | tee $save_cmd

# python llama3.1_peft_lora_predict.py \
#     --eval_data "/data/s50042884/huggingface_model/libri_test.json" \
#     --audio_tower "/data/s50042884/huggingface_model/whisper-large-v3" \
#     --base_model_path ${output_model_root}"/ACLlama_lora/" \
#     --peft_model_id ${output_model_path}"/checkpoint-30/" \
#     --clean_out_path ${output_model_path}"/test_clean.txt" \
#     --other_out_path ${output_model_path}"/test_other.txt" \
#     --num_threads ${gpu_num}


    # --eval_data "/data/s50042884/huggingface_model/libri_test_clean.json" \
    # --eval_data "/data/s50042884/huggingface_model/libri_test_other.json" \
