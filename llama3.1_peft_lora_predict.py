import os
import json
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from peft import PeftModel, PeftConfig, LoraModel, LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, AutoConfig,WhisperProcessor
import librosa
import argparse

from datasets import load_dataset
from ACLlama import ACLlamaForCausalLM
import torch
import random


class BasicSetting:
    def __init__(self):
        self.device = "cuda"
        self.sampling_rate = 16000
        self.audio_token_len = 375
        self.stop = "</s>"

CONFIG = BasicSetting()
# base_model_path = "./model_hub/qwen/Qwen1___5-72B-Chat"
# peft_model_id = "./output/qwen1.5_72B_lora/checkpoint-80/"

#base_model_path = "/mnt/user/zhangyuhao/LLM/llama3-instruct/llama3_1-8B"
base_model_path = "/wangbenyou/zhangyuhao/llms/ACLlama2/ACLlama"
peft_model_id = "/wangbenyou/zhangyuhao/llms/ACLlama2/output/ACLlama_lora_libri_check_save/checkpoint-900"
#input_audio_file= "/mnt/user/bufan/speech_data/speech_wav/LibriSpeech/LibriSpeech/test-clean/6829/68769/6829-68769-0026.flac"
#input_audio_file= "/mnt/user/bufan/speech_data/speech_wav/LibriSpeech/LibriSpeech/test-clean/6829/68769/6829-68769-0046.flac"
input_audio_file= "/wangbenyou/zhangyuhao/data/LibriSpeech/dev-other/4570/102353/4570-102353-0007.flac"
#input_audio_file= "/mnt/user/zhangyuhao/data/LibriSpeech/dev-other/4570/102353/4570-102353-0005.flac"

quantization_config = None

def get_result(model_inputs, model, tokenizer, audio_feat):

    #output_ids = model.generate(model_inputs["input_ids"], audios=audio_feat, do_sample=True, temperature=0.2, max_new_tokens=512)
    model.generation_config.pad_token_id = tokenizer.eos_token_id
    output_ids = model.generate(
        **model_inputs,
        audios=audio_feat,
        max_new_tokens=512,
        eos_token_id=tokenizer.eos_token_id,
        #do_sample=False,
    )
    #print(tokenizer.batch_decode(output_ids))
    input_ids=model_inputs["input_ids"]
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

def gen_model_inputs(tokenizer, system, prompt):
    DEFAULT_AUDIO_PATCH_TOKEN = "<audio_patch>"
    audio_placeholder = DEFAULT_AUDIO_PATCH_TOKEN * CONFIG.audio_token_len
    audio_placeholder = "\n"+audio_placeholder
    audio_placeholder_ids = tokenizer(audio_placeholder).input_ids

    begin_of_text_id = tokenizer.get_vocab()["<|begin_of_text|>"]
    start_header_id = tokenizer.get_vocab()["<|start_header_id|>"]
    end_header_id = tokenizer.get_vocab()["<|end_header_id|>"]
    eot_id = tokenizer.get_vocab()["<|eot_id|>"]
    nl_tokens = tokenizer('\n').input_ids
    _system = tokenizer('system').input_ids
    _user = tokenizer('user').input_ids
    _assistant = tokenizer('assistant').input_ids

    input_ids = []
    #batch 1
    input_id = []
    system = [begin_of_text_id] + [start_header_id] + _system + [end_header_id] + nl_tokens + tokenizer(system).input_ids + [eot_id]
    input_id += system
    #input_id += audio_placeholder_ids
    #user_input_id = [start_header_id] + _user + [end_header_id] + nl_tokens + tokenizer(prompt).input_ids + [eot_id]
    user_input_id = [start_header_id] + _user + [end_header_id] + audio_placeholder_ids + tokenizer(prompt).input_ids + [eot_id]
    assistant_input_id = [start_header_id] + _assistant + [end_header_id] + nl_tokens
    input_id += user_input_id
    input_id += assistant_input_id
    #print("input_id", input_id)
    #print(target)
    #print(tokenizer.decode(input_id))
    #print(len(input_id), len(target))
    #input_id += [tokenizer.pad_token_id] * (max_len - len(input_id))
    input_ids.append(input_id)
    input_ids = torch.tensor(input_ids, dtype=torch.int).to(CONFIG.device)
    attention_mask=input_ids.ne(tokenizer.pad_token_id)
    return dict(input_ids=input_ids, attention_mask=attention_mask)


def main(args):
    model = ACLlamaForCausalLM.from_pretrained(base_model_path,
                                               device_map="cuda",
                                               torch_dtype=torch.float16,
                                               quantization_config=quantization_config)
    print(model) 
    tokenizer = AutoTokenizer.from_pretrained(peft_model_id)
    audio_config = model.get_model().audio_tower[0].config
    audio_config.audio_patch_token = tokenizer.get_vocab()["<audio_patch>"]

    lora_config = PeftConfig.from_pretrained(peft_model_id)

    #combined_weights = torch.load(peft_model_id + "/base_model.bin", map_location=f"cuda")
    #need_combined_weights = {}
    #for item in combined_weights.keys():
    #    if "lora" not in item:
    #        need_combined_weights[item.replace("base_model.model.", "")] = combined_weights[item]
    #model.load_state_dict(need_combined_weights, strict=True)

    #my_target_modules = []
    #for id, (name, param) in enumerate(model.named_modules()):
    #    if 'model' in name and ('q_proj' in name or 'v_proj' in name):
    #        my_target_modules.append(name)
    #lora_config = LoraConfig(r=64, lora_alpha=16, target_modules=my_target_modules, lora_dropout=0.05, bias="none")
    #lora_config.target_modules=my_target_modules
    model = PeftModel.from_pretrained(model, peft_model_id, config=lora_config).to(dtype=torch.float16).to('cuda')
    print(model) 
    model.eval()
    
    prompt = "What does the person say?"
    
    system="You are a pirate chatbot who always responds in pirate speak!"
    model_inputs = gen_model_inputs(tokenizer, system, prompt)
    #model_inputs2 = tokenizer([text], return_tensors="pt").to(CONFIG.device)
    #print(tokenizer.decode(model_inputs["input_ids"][0]))
    
    fo = open("data/speech_libritrain.json","r")
    items = json.load(fo)
    for i in items:
        cur_input_audio_file = i["conversations"][0]["audio"]
        audio_processor = WhisperProcessor.from_pretrained(args.audio_tower, torch_dtype=torch.float16)
        audio, _ = librosa.load(cur_input_audio_file, sr=CONFIG.sampling_rate)
        audio_feat = audio_processor(audio, sampling_rate=CONFIG.sampling_rate, return_tensors="pt").input_features
        audio_feat = audio_feat.unsqueeze(0).unsqueeze(0).to(CONFIG.device, dtype=torch.float16)
        base_model_response = get_result(model_inputs, model, tokenizer, audio_feat)
        print("--------------------------------------")
        print(base_model_response)
        print(i["conversations"][1]["value"])
        input()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_audio_file', type=str, default='/mnt/user/bufan/speech_data/speech_wav/LibriSpeech/LibriSpeech/test-clean/6829/68769/6829-68769-0026.flac')
    parser.add_argument('--llm_model', type=str, default='/mnt/user/zhangyuhao/LLM/ACLlama2/ACLlama')
    parser.add_argument('--adapter_size', type=int, default=1280)
    parser.add_argument('--audio_tower', type=str, default='/wangbenyou/zhangyuhao/llms/whisper-v3')
    parser.add_argument('--llm_type', type=str, default='llama3')
    parser.add_argument('--temperature', type=float, default=0.2)
    parser.add_argument('--max_new_tokens', type=int, default=1024)
    args = parser.parse_args()
    main(args)


