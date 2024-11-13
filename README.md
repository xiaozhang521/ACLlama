# ACLlama

ACLlama (ACoustic Llama) is a project that extends the Llama model for acoustic processing tasks, providing scripts for both training and inference.

## Directory Structure

- `config/`: Contains configuration files for training and inference.
- `scripts/`: Includes auxiliary scripts such as data preprocessing tools.
- `ACLlama.py`: Defines the main model architecture.
- `convert.py`: Script for converting model formats from .bin to safe.
- `dump_model.py`: Exports model parameters.
- `finetune_acllama.py`: Script for fine-tuning the model.
- `llama3.1_peft_lora_predict.py`: Script for model inference.

## Environment Dependencies

Before using this project, ensure the following Python libraries are installed:
#### requirement.txt: 
```txt
torch==2.2.0
transformers==4.43.1
deepspeed==0.9.4
accelerate==0.34.2
peft==0.5.0
numpy==1.24.4
jinja2==3.1.3
pydantic==1.10.6
datasets==2.18.0
bitsandbytes==0.43.0
```

#### Install command
```bash
pip install protobuf==3.20.1  -i https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple
pip uninstall -y transformer-engine
pip uninstall -y pydantic
pip install -r requirements.txt  -i https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple
pip install flash_attn-2.6.3+cu123torch2.2cxx11abiFALSE-cp310-cp310-linux_x86_64.whl
apt install -y screen
```

## Model Training

Model Training
To fine-tune the ACLlama model, execute the finetune_acllama.py script with the following command:

```bash
NCCL_P2P_DISABLE=1 \
NCCL_IB_DISABLE=1 \
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
torchrun \
    --nproc_per_node 8 \
    --nnodes 1 \
    --node_rank 0 \
    --master_addr localhost \
    --master_port 6601 \
    ../finetune_acllama.py \
    --audio_model_name_or_path "/path/to/whisper-v3" \
    --text_model_name_or_path "/path/to/ACLlama" \
    --data_path "../data/speech_libritrain.json" \
    --fp16 True \
    --output_dir "../output/ACLlama_lora" \
    --num_train_epochs 40 \
    --per_device_train_batch_size 32 \
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
    --deepspeed "../config/ds_config_zero2.json" \
    --use_lora
```
**Parameter Explanation:**

- `NCCL_P2P_DISABLE=1` and `NCCL_IB_DISABLE=1`: Environment variables to configure NCCL settings, potentially improving communication performance.
- `CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7`: Specifies which GPUs to use.
- `torchrun`: Utility to launch distributed training.
  - `--nproc_per_node`: Number of processes per node (equal to the number of GPUs).
  - `--nnodes`: Number of nodes participating in training.
  - `--node_rank`: Rank of the node.
  - `--master_addr`: Address of the master node.
  - `--master_port`: Port of the master node.
- `../finetune_acllama.py`: Path to the training script.
  - `--audio_model_name_or_path`: Path to the pretrained audio model.
  - `--text_model_name_or_path`: Path to the pretrained text model.
  - `--data_path`: Path to the training data.
  - `--fp16 True`: Enables 16-bit floating-point precision for training.
  - `--output_dir`: Directory to save the fine-tuned model.
  - `--num_train_epochs`: Number of training epochs.
  - `--per_device_train_batch_size`: Batch size per device during training.
  - `--per_device_eval_batch_size`: Batch size per device during evaluation.
  - `--gradient_accumulation_steps`: Number of steps to accumulate gradients before updating.
  - `--evaluation_strategy`: Evaluation strategy; "no" means no evaluation during training.
  - `--save_strategy`: Strategy to save checkpoints; "steps" means saving at regular steps.
  - `--save_steps`: Number of steps between checkpoint saves.
  - `--save_total_limit`: Maximum number of checkpoints to keep.
  - `--learning_rate`: Learning rate for training.
  - `--weight_decay`: Weight decay for optimization.
  - `--adam_beta2`: Beta2 parameter for the Adam optimizer.
  - `--warmup_ratio`: Ratio of warmup steps to total training steps.
  - `--lr_scheduler_type`: Type of learning rate scheduler.
  - `--logging_steps`: Number of steps between logging outputs.
  - `--report_to`: Reporting tool; "none" means no reporting.
  - `--model_max_length`: Maximum sequence length for the model.
  - `--gradient_checkpointing`: Enables gradient checkpointing to save memory.
  - `--deepspeed`: Path to the DeepSpeed configuration file.
  - `--use_lora`: Enables the use of LoRA (Low-Rank Adaptation) for parameter-efficient fine-tuning.
## Model Inference

To perform inference with the fine-tuned model, run the `llama3.1_peft_lora_predict.py` script:

```bash
python llama3.1_peft_lora_predict.py \
    --input_audio_file <path_to_audio_file> \
    --llm_model <path_to_llm_model> \
    --adapter_size <adapter_size> \
    --audio_tower <path_to_audio_tower_model> \
    --llm_type <llm_type> \
    --temperature <generation_temperature> \
    --max_new_tokens <max_new_tokens_to_generate>
```

**Parameter Explanation:**

- `--input_audio_file`: Path to the input audio file.
- `--llm_model`: Path to the large language model.
- `--adapter_size`: Size of the adapter.
- `--audio_tower`: Path to the audio tower model.
- `--llm_type`: Type of the large language model.
- `--temperature`: Sampling temperature for generation.
- `--max_new_tokens`: Maximum number of new tokens to generate.

