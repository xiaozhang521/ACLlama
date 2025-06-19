import os
import json
from peft import PeftModel, PeftConfig, LoraModel, LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, AutoConfig, WhisperProcessor
import librosa
import argparse
# from datasets import load_dataset
from ACLlama_el import ACLlamaForCausalLM
import torch
import random
from tqdm import tqdm
import torch.multiprocessing as mp

########
import torch_npu
from torch_npu.contrib import transfer_to_npu
########

os.environ["TOKENIZERS_PARALLELISM"] = "true"


class BasicSetting:
    def __init__(self):
        # self.devices = ["cuda:0", "cuda:1", "cuda:2", "cuda:3"]
        self.devices = ["npu:0", "npu:1", "npu:2", "npu:3", "npu:4", "npu:5", "npu:6",  "npu:7"]
        self.sampling_rate = 16000
        self.audio_token_len = 1  # 1500 = 300 token x 5 compress
        self.stop = "</s>"


CONFIG = BasicSetting()


def get_result(model_inputs, model, tokenizer, audio_feat):
    model.generation_config.pad_token_id = tokenizer.eos_token_id
    output_ids = model.generate(
        **model_inputs,
        audios=audio_feat,
        max_new_tokens=512,
        eos_token_id=tokenizer.eos_token_id,
        # do_sample=False,
    )
    # print(f"output_ids is : {output_ids}")
    # exit(0)
    # print(tokenizer.batch_decode(output_ids))
    input_ids = model_inputs["input_ids"]
    input_token_len = input_ids.shape[1]
    n_diff_input_output = (input_ids != output_ids[:, :input_token_len]).sum().item()
    if n_diff_input_output > 0:
        print(f'[Warning] {n_diff_input_output} output_ids are not the same as the input_ids')
    outputs = tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)[0]

    outputs = outputs.strip()
    if outputs.endswith(CONFIG.stop):
        outputs = outputs[:-len(CONFIG.stop)]
    outputs = outputs.strip()

    return outputs


def gen_model_inputs(tokenizer, system, prompt, device, audio_placeholder_ids, begin_of_text_id, start_header_id,
                     end_header_id, eot_id, nl_tokens, _system, _user, _assistant):
    input_ids = []
    # batch 1
    input_id = []
    system = [begin_of_text_id] + [start_header_id] + _system + [end_header_id] + nl_tokens + tokenizer(
        system).input_ids + [eot_id]
    input_id += system
    # input_id += audio_placeholder_ids
    # user_input_id = [start_header_id] + _user + [end_header_id] + nl_tokens + tokenizer(prompt).input_ids + [eot_id]
    user_input_id = [start_header_id] + _user + [end_header_id] + audio_placeholder_ids + tokenizer(
        prompt).input_ids + [eot_id]
    assistant_input_id = [start_header_id] + _assistant + [end_header_id] + nl_tokens
    input_id += user_input_id
    input_id += assistant_input_id
    input_ids.append(input_id)
    input_ids = torch.tensor(input_ids, dtype=torch.int).to(device)
    attention_mask = input_ids.ne(tokenizer.pad_token_id)
    return dict(input_ids=input_ids, attention_mask=attention_mask)


def process_items(thread_id, subset, args, CONFIG, return_dict):
    device = CONFIG.devices[thread_id % len(CONFIG.devices)]  # 根据线程ID选择设备
    print(f"Thread-{thread_id} running on {device}")

    quantization_config = None
    model = ACLlamaForCausalLM.from_pretrained(
        args.base_model_path,
        device_map=None,
        torch_dtype=torch.float16,
        quantization_config=quantization_config,
    )
    
    # import glob
    # from safetensors.torch import load_file
    # shard_files = sorted(glob.glob(os.path.join(args.peft_model_id, "adapter_model-*.safetensors")))
    # if not shard_files:
    #     shard_files = sorted(glob.glob(os.path.join(args.peft_model_id, "adapter_model.safetensors")))
    # need_combined_weights = {}
    # for shard in shard_files:
    #     shard_state = load_file(shard)
    #     need_combined_weights.update(shard_state)
    # print(f"need_combined_weights is : {need_combined_weights.keys()}")
    
    # # new_sd = {}
    # # for k, v in need_combined_weights.items():
    # #     if ".lora_A.weight" in k and ".default" not in k:
    # #         k = k.replace(".lora_A.weight", ".lora_A.default.weight")
    # #     if ".lora_B.weight" in k and ".default" not in k:
    # #         k = k.replace(".lora_B.weight", ".lora_B.default.weight")
    # #     new_sd[k] = v
    # # torch.save(new_sd, os.path.join(args.peft_model_id, "adapter_model/patched.bin"))
    
    for module in model.model.audio_tower:
        module.to(device)
    torch_npu.npu.empty_cache()
    # model.model.mask_tensor = model.model.mask_tensor.to(device)
    tokenizer = AutoTokenizer.from_pretrained(args.peft_model_id)

    audio_config = model.get_model().audio_tower[0].config
    audio_config.audio_patch_token = tokenizer.get_vocab()["<audio_patch>"]
    audio_config.llm_pad_token_id = tokenizer.pad_token_id
    audio_config.audio_patch_size = CONFIG.audio_token_len

    # LoRA
    lora_config = PeftConfig.from_pretrained(args.peft_model_id)
    # model = PeftModel.from_pretrained(model, os.path.join(args.peft_model_id, "adapter_model/patched.bin"), config=lora_config, adapter_name="default").to(
    model = PeftModel.from_pretrained(model, args.peft_model_id, config=lora_config).to(
        dtype=torch.float16, device=device
    )
    torch_npu.npu.empty_cache()
    model.eval()


    DEFAULT_AUDIO_PATCH_TOKEN = "<audio_patch>"
    audio_placeholder = DEFAULT_AUDIO_PATCH_TOKEN * CONFIG.audio_token_len
    audio_placeholder = "\n" + audio_placeholder
    audio_placeholder_ids = tokenizer(audio_placeholder).input_ids

    begin_of_text_id = tokenizer.get_vocab()["<|begin_of_text|>"]
    start_header_id = tokenizer.get_vocab()["<|start_header_id|>"]
    end_header_id = tokenizer.get_vocab()["<|end_header_id|>"]
    eot_id = tokenizer.get_vocab()["<|eot_id|>"]
    nl_tokens = tokenizer('\n').input_ids
    _system = tokenizer('system').input_ids
    _user = tokenizer('user').input_ids
    _assistant = tokenizer('assistant').input_ids

    # Whisper
    audio_processor = WhisperProcessor.from_pretrained(args.audio_tower, torch_dtype=torch.float16)

    # prefix
    prompt = "What does the person say?"
    # prompt = "What is the content of the recording?"
    # system = "You are a pirate chatbot who always responds in pirate speak!"
    system = "You are a helpful language and speech assistant. You are able to understand the speech content that the user provides, and assist the user with a variety of tasks using natural language."
    model_inputs = gen_model_inputs(tokenizer, system, prompt, device, audio_placeholder_ids, begin_of_text_id,
                                    start_header_id, end_header_id, eot_id, nl_tokens, _system, _user, _assistant)

    thread_results = {"clean": [], "other": []}

    for idx, i in tqdm(subset, desc=f"Thread-{thread_id} processing"):
        cur_input_audio_file = i["conversations"][0]["audio"]

        if "test-other" in cur_input_audio_file:
            category = "other"
        elif "test-clean" in cur_input_audio_file:
            category = "clean"
        else:
            print(f"Unrecognized audio file: {cur_input_audio_file}")
            raise ValueError("Unrecognized audio file category")

        audio, _ = librosa.load(cur_input_audio_file, sr=CONFIG.sampling_rate)
        audio_feat = audio_processor(
            audio, sampling_rate=CONFIG.sampling_rate, return_tensors="pt"
        ).input_features
        audio_feat = audio_feat.to(device, dtype=torch.float16)

        base_model_response = get_result(model_inputs, model, tokenizer, audio_feat)

        result_ = (
                base_model_response.replace("The person says: ", "").strip()
                + " ||| "
                + i["conversations"][1]["value"].replace("The person says: ", "").strip()
        )
        
        thread_results[category].append(result_)

    return_dict[thread_id] = thread_results


def main(args):
    open(args.clean_out_path, "w").close()
    open(args.other_out_path, "w").close()

    with open(args.eval_data, "r") as fo:
        items = json.load(fo)

    # 数据分块 并附带每条数据原始索引
    if args.num_threads > 0:
        chunk_size = len(items) // args.num_threads + (1 if len(items) % args.num_threads else 0)
    else:
        chunk_size = len(items)

    subsets = [
        [(idx, items[idx]) for idx in range(i, min(i + chunk_size, len(items)))]
        for i in range(0, len(items), chunk_size)
    ]

    manager = mp.Manager()
    return_dict = manager.dict()  # 存储每个线程的结果
    processes = []

    for thread_id in range(args.num_threads):
        p = mp.Process(
            target=process_items,
            args=(thread_id, subsets[thread_id], args, CONFIG, return_dict),
        )
        processes.append(p)
        p.start()

    for p in processes:
        p.join()

    with open(args.clean_out_path, "a") as clean_out, open(args.other_out_path, "a") as other_out:
        for thread_id in range(args.num_threads):
            thread_results = return_dict[thread_id]
            for line in thread_results["clean"]:
                clean_out.write(line + "\n")
            for line in thread_results["other"]:
                other_out.write(line + "\n")

    print("Processing completed!")


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)

    parser = argparse.ArgumentParser()
    parser.add_argument('--adapter_size', type=int, default=1280)
    parser.add_argument('--audio_tower', type=str, default='/huyujin/LLMs/whisper-v3')
    parser.add_argument('--temperature', type=float, default=0.2)
    parser.add_argument('--max_new_tokens', type=int, default=1024)
    parser.add_argument('--base_model_path', type=str,
                        default="/huyujin/ACLlama/ACLlama")
    parser.add_argument('--peft_model_id', type=str,
                        default="/huyujin/ACLlama/output/ACLlama_lora_mt/checkpoint-28000")
    parser.add_argument('--eval_data', type=str, default="/huyujin/ACLlama/data/libri_test.json")
    parser.add_argument('--clean_out_path', type=str,
                        default="/huyujin/ACLlama/clean_out_ACLlama_3k")
    parser.add_argument('--other_out_path', type=str,
                        default="/huyujin/ACLlama/other_out_ACLlama_3k")
    parser.add_argument('--num_threads', type=int, default=4)

    args = parser.parse_args()

    main(args)

