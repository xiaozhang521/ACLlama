from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss

from transformers import AutoConfig, AutoModelForCausalLM, \
                         LlamaConfig, LlamaModel, LlamaForCausalLM

from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from transformers import (
    WhisperProcessor,
    WhisperModel,
)



class ACLlamaConfig(LlamaConfig):
    model_type = "ACLlama"
    


def load_whisper(audio_tower_name):
    model = WhisperModel.from_pretrained(
            audio_tower_name,torch_dtype=torch.float16, low_cpu_mem_usage=True).to('cuda')
    model.config.forced_decoder_ids = None
    return model



class ACLlamaModel(LlamaModel):
    config_class = ACLlamaConfig

    def __init__(self, config: LlamaConfig):
        super(ACLlamaModel, self).__init__(config)

        if hasattr(config, "audio_tower"):
            self.audio_tower = [load_whisper(config.audio_tower)]

        if hasattr(config, "adapter_size"):
            #self.down_sampler = Conv1dSubsampler(config.adapter_size, config.hidden_size // 2, config.hidden_size // 2, [5])
            #self.conv1 = nn.Conv1d(1280, config.hidden_size//2, kernel_size=3, stride=2, padding=1)
            #self.conv2 = nn.Conv1d(4096, 4096, kernel_size=3, stride=2, padding=1)
            self.mm_projector1 = nn.Linear(config.adapter_size*4 , config.hidden_size)
            self.relu = nn.ReLU()
            self.mm_projector2 = nn.Linear(config.hidden_size , config.hidden_size)
            self.post_init()

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        audios: Optional[torch.FloatTensor] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPast]:

        # HACK: replace back original embeddings for LLaAA pretraining
        orig_embeds_params = getattr(self, 'orig_embeds_params', None)

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        audio_tower = getattr(self, 'audio_tower', None)
        if audio_tower is not None and (input_ids.shape[1] != 1 or self.training) and audios is not None:
            audio_tower = audio_tower[0]  # HACK: for FSDP
            with torch.no_grad():
                bs_audio_features = []
                for audios_list in audios:
                    if len(audios_list) == 0:
                        dummy_audio_feature = torch.zeros(self.config.audio_token_len, self.config.adapter_size, device=inputs_embeds.device, dtype=inputs_embeds.dtype)
                        audio_features = [dummy_audio_feature]
                    else:
                        audio_features = []
                        for audio in audios_list:
                            #decoder_input_ids = torch.ones((1, self.config.audio_token_len)) * audio_tower.config.decoder_start_token_id
                            #decoder_input_ids = decoder_input_ids.to(audio.device).to(torch.long)
                            #audio_feature = audio_tower(audio, decoder_input_ids=decoder_input_ids).last_hidden_state
                            audio_feature = audio_tower.encoder(audio).last_hidden_state
                            audio_features.append(audio_feature)
                    bs_audio_features.append(audio_features)

            audio_config = audio_tower.config
            new_input_embeds = []
            for cur_input_ids, cur_input_embeds, cur_audio_features in zip(input_ids, inputs_embeds, bs_audio_features):
                if (cur_input_ids == audio_config.audio_patch_token).sum() == 0:
                    # multimodal LLM, but the current sample is not multimodal, for using both language and audio data
                    dummy_audio_features = self.mm_projector(cur_audio_features[0])
                    cur_input_embeds = cur_input_embeds + (0. * dummy_audio_features).sum()
                    new_input_embeds.append(cur_input_embeds)
                    continue
                audio_start_tokens = torch.where(cur_input_ids == audio_config.audio_patch_token)[0][:1]
                if len(audio_start_tokens) != len(cur_audio_features):
                    raise ValueError(f"The number of audio start tokens ({len(audio_start_tokens)}) and audio features ({len(cur_audio_features)}) should be the same.")
                for audio_start_token_pos, cur_audio_feature in zip(audio_start_tokens, cur_audio_features):
                    #print(cur_audio_feature[0][0])
                    cur_audio_feature = cur_audio_feature.view(cur_audio_feature.shape[0], cur_audio_feature.shape[1]//4, 4 * cur_audio_feature.shape[2])

                    #cur_audio_feature = nn.functional.gelu(self.conv1(cur_audio_feature.transpose(1,2)/3)).transpose(1,2)
                    #print(cur_audio_feature.transpose(1,2)[0][0])
                    #cur_audio_feature = nn.functional.gelu(self.conv2(cur_audio_feature)).transpose(1,2)
                    #print(cur_audio_feature[0][0])
                    cur_audio_feature = self.mm_projector1(cur_audio_feature)
                    cur_audio_feature = self.relu(cur_audio_feature)
                    cur_audio_feature = self.mm_projector2(cur_audio_feature)[0]
                    cur_audio_feature = cur_audio_feature.to(device=cur_input_embeds.device)
                    num_patches = cur_audio_feature.shape[0]
                    if orig_embeds_params is not None:
                        cur_new_input_embeds = torch.cat(
                            (cur_input_embeds[:audio_start_token_pos].detach(),
                             cur_input_embeds[audio_start_token_pos:audio_start_token_pos+1],
                             cur_audio_feature,
                             cur_input_embeds[audio_start_token_pos + num_patches + 1:audio_start_token_pos + num_patches + 2],
                             cur_input_embeds[audio_start_token_pos + num_patches + 2:].detach()), dim=0)
                    else:
                        cur_new_input_embeds = torch.cat((
                            cur_input_embeds[:audio_start_token_pos],
                            cur_audio_feature,
                            cur_input_embeds[audio_start_token_pos + num_patches:]), dim=0)
                new_input_embeds.append(cur_new_input_embeds)

            inputs_embeds = torch.stack(new_input_embeds, dim=0)

        return super(ACLlamaModel, self).forward(
            input_ids=None, attention_mask=attention_mask, past_key_values=past_key_values,
            inputs_embeds=inputs_embeds, use_cache=use_cache,
            output_attentions=output_attentions, output_hidden_states=output_hidden_states,
            return_dict=return_dict
        )


class ACLlamaForCausalLM(LlamaForCausalLM):
    config_class = ACLlamaConfig

    def __init__(self, config):
        super(LlamaForCausalLM, self).__init__(config)
        self.model = ACLlamaModel(config)

        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def get_model(self):
        return self.model

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        audios: Optional[torch.FloatTensor] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            audios=audios
        )

        hidden_states = outputs[0]
        logits = self.lm_head(hidden_states)


        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model/pipeline parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )




AutoConfig.register("ACLlama", ACLlamaConfig)
AutoModelForCausalLM.register(ACLlamaConfig, ACLlamaForCausalLM)
