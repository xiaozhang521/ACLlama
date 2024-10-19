from safetensors.torch import save_file
import torch
import sys
model_weights = torch.load(sys.argv[1]+'/base_model.bin')
new_model_weights = {}
for key in model_weights.keys():
    if "lora" in key:
        new_model_weights[key] = model_weights[key]
save_file(new_model_weights, sys.argv[1]+'/adater_model.safetensors')
