from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss, CTCLoss


from transformers import AutoConfig, AutoModelForCausalLM, \
                         LlamaConfig, LlamaModel, LlamaForCausalLM
from transformers.trainer_pt_utils import LabelSmoother
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from transformers import (
    WhisperProcessor,
    WhisperModel,
)

IGNORE_TOKEN_ID = LabelSmoother.ignore_index


class ACLlamaConfig(LlamaConfig):
    model_type = "ACLlama"
    


def load_whisper(audio_tower_name):
    model = WhisperModel.from_pretrained(
            audio_tower_name,torch_dtype=torch.float16, low_cpu_mem_usage=True).to('cuda')
    model.config.forced_decoder_ids = None
    return model

class LookBackModule(nn.Module):
    def __init__(self, cfg: LlamaConfig):
        super().__init__()
        self.encoder_attn = nn.MultiheadAttention(
            cfg.hidden_size,
            cfg.num_attention_heads,
            dropout=0.1,
            batch_first=True
        )
        self.atten_layer_norm = nn.LayerNorm(cfg.hidden_size)
        #self.fc1 = nn.Linear(cfg.encoder_embed_dim, cfg.encoder_ffn_embed_dim)
        #self.fc2 = nn.Linear(cfg.encoder_ffn_embed_dim, cfg.encoder_embed_dim)
        #self.activation_fn = nn.SiLU() #utils.get_activation_fn(activation="swish")
        #self.ffn_layer_norm = LayerNorm(cfg.encoder_embed_dim)
        #self.lb_dropout = nn.Dropout(0.1)

    def forward(self, x, wav_feature, bf_shrink_padding_mask):

        residual = x
        x, _ = self.encoder_attn(
            query=x,
            key=wav_feature,
            value=wav_feature,
            key_padding_mask=bf_shrink_padding_mask,
            #attn_mask=padding_mask,
        )
        x += residual
        #x = self.lb_dropout(x)
        x = self.atten_layer_norm(x)
        #residual = x
        #x = self.fc1(x)
        #x = self.activation_fn(x)
        #x = self.fc2(x)
        #x += residual
        #x = self.lb_dropout(x)
        #x = self.ffn_layer_norm(x)
        return x

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
            self.mm_projector1 = nn.Linear(config.adapter_size*2 , config.hidden_size)
            #self.relu = nn.ReLU()
            #self.mm_projector2 = nn.Linear(config.hidden_size , config.hidden_size)
            asr_encoder_layer = nn.TransformerEncoderLayer(
                d_model=config.hidden_size,
                nhead=config.num_attention_heads,
                dim_feedforward=config.hidden_size*2,
                dropout=0.1,
                norm_first=True
            )
            self.lbm =  LookBackModule(config)
            self.out_norm = nn.LayerNorm(config.hidden_size)
            self.audio_feature_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
            self.asr_transformer_encoder = nn.TransformerEncoder(asr_encoder_layer, num_layers=1)

        ########
        self.avg_pooler = nn.AvgPool1d(2, stride=2)
        ########


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
            audio_list=[]
            
            audio_config = audio_tower.config
            #with torch.no_grad():
            #    audio_features = audio_tower.encoder(audios).last_hidden_state
            #for audio_feature in audio_features:
            #    audio_feature = audio_feature.unsqueeze(0)
            for audio in audios:
                with torch.no_grad():
                    audio=audio.unsqueeze(0)
                    audio_feature = audio_tower.encoder(audio).last_hidden_state

                audio_feature = audio_feature.view(audio_feature.shape[0], audio_feature.shape[1]//2, 2 * audio_feature.shape[2])
                audio_feature = self.mm_projector1(audio_feature)
                audio_feature = self.asr_transformer_encoder(audio_feature)
                audio_feature = self.out_norm(audio_feature)
                audio_list.append(audio_feature[0])

            audio_features = torch.stack(audio_list, dim=0)
 
            ######
            audio_feature_lengths = attention_mask.int().sum(dim=1)  # shape: (batch_size,)

            # print(f"inputs_embeds is : {inputs_embeds}")
            # print(f"attention_mask is : {attention_mask}")
            # print(f"audio_features is : {audio_features}")
            # print(f"audio_feature_lengths is : {audio_feature_lengths}")
            # print(f"audio_feature_lengths is : {audio_feature_lengths.max()}")
            # print(f"attention_mask is : {attention_mask.size()}")
            # print(f"inputs_embeds is : {inputs_embeds.size()}")
            # print(f"audio_features is : {audio_features.size()}")
            
            audio_features_4_loss = audio_features.clone().permute(0, 2, 1)
            while audio_features_4_loss.size(2) // 2 - 1 > audio_feature_lengths.max():
                audio_features_4_loss = self.avg_pooler(audio_features_4_loss)
            audio_features_4_loss = audio_features_4_loss.permute(0, 2, 1)
            
            # print("###############")
            # print(f"audio_feature_lengths is : {audio_feature_lengths}")
            # print(f"audio_features_4_loss is : {audio_features_4_loss}")
            # print(f"audio_features_4_loss is : {audio_features_4_loss.size()}")
            # exit(0)
            ######
 
            #audio_features = audio_features.view(audio_features.shape[0], audio_features.shape[1]//2, 2 * audio_features.shape[2])
            #audio_features = self.mm_projector1(audio_features)
            #audio_features = self.asr_transformer_encoder(audio_features)
            #audio_features = self.out_norm(audio_features)

            predict_logits = self.audio_feature_head(audio_features)

            new_input_embeds = []
            label_shift = []
            label_extend = -1
            new_input_ids = []
            tokens = predict_logits.argmax(dim=-1)
            shrink_mask = tokens.roll(1) != tokens
            shrink_mask[:,0] = True

            #for i in range(shrink_mask.shape[0]):
            #    m_length = min(torch.nonzero(shrink_mask[i])[-1] + 5, shrink_mask.shape[1])
            #    shrink_mask[i][:m_length]=torch.ones(m_length).to(shrink_mask.device)
                
            lengths = shrink_mask.long().sum(-1)
            shrink_2d = audio_features[shrink_mask]
            #num_patches = audio_features.shape[1]
            num_patches = audio_config.audio_patch_size
            l_index=0
            shrink_features = []
            for v, audio_feature, mask in zip(lengths, audio_features, ~shrink_mask):
                shrink_feature = shrink_2d[l_index:l_index+v]
                shrink_feature = self.lbm(shrink_feature, audio_feature, bf_shrink_padding_mask=mask)
                #shrink_feature = self.lbm(shrink_feature, audio_feature, bf_shrink_padding_mask=None)
                shrink_features.append(shrink_feature)
                l_index += v
            if self.training: 
                maxn_length = lengths.max()
                label_extend = maxn_length - num_patches
                for cur_input_ids, cur_input_embeds, shrink_feature in zip(input_ids, inputs_embeds, shrink_features):
                    pad_ids = torch.full(size=(maxn_length,), fill_value=audio_config.llm_pad_token_id, dtype=torch.long).to(attention_mask.device)
                    pad_embeds = self.embed_tokens(pad_ids)
                    v = shrink_feature.shape[0]
                    audio_start_token_pos = torch.where(cur_input_ids == audio_config.audio_patch_token)[0][:1]
                    cur_new_input_id = torch.cat((cur_input_ids[:audio_start_token_pos], cur_input_ids[audio_start_token_pos: audio_start_token_pos+1].repeat(v), cur_input_ids[audio_start_token_pos + num_patches:], pad_ids[:maxn_length - v]), dim=0)
                    cur_new_input_embeds = torch.cat((
                    cur_input_embeds[:audio_start_token_pos],
                    shrink_feature,
                    cur_input_embeds[audio_start_token_pos + num_patches:],pad_embeds[:maxn_length-v]), dim=0)
                    new_input_embeds.append(cur_new_input_embeds)
                    new_input_ids.append(cur_new_input_id)
                    label_shift.append(v - num_patches)
                    
                input_ids = torch.stack(new_input_ids, dim=0)
                attention_mask=input_ids.ne(audio_config.llm_pad_token_id)
                inputs_embeds = torch.stack(new_input_embeds, dim=0)
            else:
                for cur_input_ids, cur_input_embeds, shrink_feature in zip(input_ids, inputs_embeds, shrink_features):
                    v = shrink_feature.shape[0]

                    audio_start_token_pos = torch.where(cur_input_ids == audio_config.audio_patch_token)[0][:1]
                    cur_new_input_id = torch.cat((cur_input_ids[:audio_start_token_pos],cur_input_ids[audio_start_token_pos: audio_start_token_pos+1].repeat(v), cur_input_ids[audio_start_token_pos + num_patches:]),dim=0)
                    cur_new_input_embeds = torch.cat((
                    cur_input_embeds[:audio_start_token_pos],
                    shrink_feature,
                    cur_input_embeds[audio_start_token_pos + num_patches:]), dim=0)
                    new_input_embeds.append(cur_new_input_embeds)
                    new_input_ids.append(cur_new_input_id)
                input_ids = torch.stack(new_input_ids, dim=0)
                attention_mask=input_ids.ne(audio_config.llm_pad_token_id)
                inputs_embeds = torch.stack(new_input_embeds, dim=0)
            #else:
            #    for cur_input_ids, cur_input_embeds, cur_audio_features in zip(input_ids, inputs_embeds, audio_features):
            #        #num_patches = cur_audio_features.shape[0]
            #        num_patches = audio_config.audio_patch_size
            #        audio_start_token_pos = torch.where(cur_input_ids == audio_config.audio_patch_token)[0][:1]
            #        #if len(audio_start_tokens) != len(cur_audio_features):
            #        #    raise ValueError(f"The number of audio start tokens ({len(audio_start_tokens)}) and audio features ({len(cur_audio_features)}) should be the same.")
            #        cur_new_input_embeds = torch.cat((
            #            cur_input_embeds[:audio_start_token_pos],
            #            cur_audio_features,
            #            cur_input_embeds[audio_start_token_pos + num_patches:]), dim=0)
            #        new_input_embeds.append(cur_new_input_embeds)

            #    inputs_embeds = torch.stack(new_input_embeds, dim=0)

        return_state=super(ACLlamaModel, self).forward(
            input_ids=None, attention_mask=attention_mask, past_key_values=past_key_values,
            inputs_embeds=inputs_embeds, use_cache=use_cache,
            output_attentions=output_attentions, output_hidden_states=output_hidden_states,
            return_dict=return_dict
        )
        if self.training:
            return_state["audio_features"] =  predict_logits
            return_state["label_shift"] = label_shift
            return_state["label_extend"] = label_extend
        return_state = {"audio_features":predict_logits}
        
        #########
        return_state_update = {"audio_feature_lengths": audio_feature_lengths, "audio_features_4_loss": audio_features_4_loss, "inputs_embeds": inputs_embeds}
        return_state.update(return_state_update)
        #########
        
        return return_state 


class ACLlamaForCausalLM(LlamaForCausalLM):
    config_class = ACLlamaConfig

    def __init__(self, config):
        super(LlamaForCausalLM, self).__init__(config)
        self.model = ACLlamaModel(config)

        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()
        
        ########
        self.similarity_function = nn.CosineSimilarity(dim=-1)
        ########

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
        asr_targets: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        audios: Optional[torch.FloatTensor] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        ######
        input_ids_neg: Optional[torch.LongTensor] = None,
        labels_neg: Optional[torch.LongTensor] = None,
        attention_mask_neg: Optional[torch.Tensor] = None,
        audios_neg: Optional[torch.FloatTensor] = None,
        asr_targets_neg: Optional[torch.LongTensor] = None,
        ######
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # print(f"input_ids is : {input_ids}")
        # print(f"attention_mask is : {attention_mask}")
        # print(f"labels is : {labels}")
        # print(f"audios is : {audios}")
        # print(f"attention_mask is : {attention_mask.size()}")
        # print(f"labels is : {labels.size()}")
        # print(f"input_ids is : {input_ids.size()}")
        # print(f"audios is : {audios.size()}")

                
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
        
        #######
        audio_feature_lengths = outputs.pop("audio_feature_lengths")
        audio_features_4_loss = outputs.pop("audio_features_4_loss")
        inputs_embeds = outputs.pop("inputs_embeds")
        # print(f"audio_feature_lengths is : {audio_feature_lengths}")
        # print(f"audio_features_4_loss is : {audio_features_4_loss.size()}")
        # print(f"inputs_embeds is : {inputs_embeds.size()}")
        #######
        
        # hidden_states = outputs[0]
        # logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            if asr_targets is not None:
                mask_asr_targets = (asr_targets != IGNORE_TOKEN_ID)
                target_lengths = mask_asr_targets.sum(1)
                input_lengths = torch.full(size=(outputs["audio_features"].shape[0],), fill_value=outputs["audio_features"].shape[1], dtype=torch.long)
                asr_logits = outputs["audio_features"]

                loss_ctc = CTCLoss()

                log_probs = F.log_softmax(asr_logits, dim=-1).transpose(0, 1)
                #print(asr_targets.shape)
                #print(input_lengths, target_lengths)

                with torch.backends.cudnn.flags(enabled=False):
                    loss_asr = F.ctc_loss(
                        log_probs,
                        asr_targets,
                        input_lengths,
                        target_lengths,
                        blank=self.model.audio_tower[0].config.audio_patch_token,
                        reduction='mean',
                        zero_infinity=True,
                    )
            else:
                loss_asr=0

            # # Shift so that tokens < n predict n
            # shift_logits = logits[..., :-1, :].contiguous()
            # shift_labels = labels[..., 1:].contiguous()

            # if len(outputs["label_shift"]) >0:
            #     if outputs["label_extend"] != -1:
            #         new_shift_labels = torch.full(size=(shift_labels.shape[0], outputs["label_extend"]+shift_labels.shape[1]), fill_value=IGNORE_TOKEN_ID, dtype=torch.long).to(shift_labels.device)
            #         for i in range(len(outputs["label_shift"])):
            #             #extend=torch.full(size=(outputs["label_extend"][i],), fill_value=IGNORE_TOKEN_ID, dtype=torch.long).to(shift_labels[i].device)
            #             #shift_labels[i] = torch.cat((shift, shift_labels[i], extend), dim=0)
            #             #print(new_shift_labels[i].shape,outputs["label_shift"][i],len(shift_labels[i]))
            #             new_shift_labels[i][outputs["label_shift"][i]:outputs["label_shift"][i] + len(shift_labels[i])]= shift_labels[i]
            #         shift_labels = new_shift_labels
            #     else:
            #         for i in range(len(outputs["label_shift"])):
            #             shift_labels[i]= shift_labels[i].roll(-outputs["label_shift"][i])

            # loss_fct = CrossEntropyLoss()
            # # Flatten the tokens
            # shift_logits = shift_logits.view(-1, self.config.vocab_size)
            # shift_labels = shift_labels.view(-1)
                    
            # #value, index = shift_logits.topk(k=1, dim=-1)
            # #index = index.view(-1)
            # #mask = (shift_labels != -100)
            # #gold_label = torch.masked_select(shift_labels, mask)
            # #index_label = torch.masked_select(index, mask)
            # #print(gold_label.shape, gold_label[:50])
            # #print(index_label.shape, index_label[:50])

            # # Enable model/pipeline parallelism
            # shift_labels = shift_labels.to(shift_logits.device)
            # loss = loss_fct(shift_logits, shift_labels)
            # loss = loss + 0.3*loss_asr 

            loss = loss_asr 


            ########
            def get_contrastive_loss(self, encoder_out1, encoder_out2):
                def _sentence_embedding(encoder_out, padding_mask):
                    mask=(~padding_mask).int()
                    encoder_output = encoder_out.transpose(0, 1)
                    
                    #if "src_tokens" in sample["net_input"]:
                    #    src_tokens = sample["net_input"]["src_tokens"]
                    #    mask = (src_tokens != self.padding_idx)
                    encoder_embedding = (encoder_output * mask.unsqueeze(-1)).sum(dim=1) / mask.float().sum(dim=1).unsqueeze(-1)  # [batch, hidden_size]
                    return encoder_embedding
                if self.is_shrink != "": 
                    encoder_embedding1 = _sentence_embedding(encoder_out1["encoder_out"], encoder_out1["padding_mask"])  # [batch, hidden_size]
                    encoder_embedding2 = _sentence_embedding(encoder_out2["encoder_out"][0], encoder_out2["encoder_padding_mask"][0])  # [batch, hidden_size]
                    batch_size = encoder_embedding2.shape[0]
                    feature_dim = encoder_embedding2.shape[1]
                    anchor_feature = encoder_embedding1
                    contrast_feature = encoder_embedding2
                    if self.get_similarity:
                        similarity = self.similarity_function(encoder_out1["wav2vec_out"].mean(1),encoder_embedding2).mean(-1)
                        #print(encoder_out1["wav2vec_out"].mean(1).shape)
                    else: 
                        similarity = self.similarity_function(encoder_embedding1,encoder_embedding2).mean(-1)
                    anchor_dot_contrast = self.similarity_function(anchor_feature.expand((batch_size, batch_size, feature_dim)),
                                                            torch.transpose(contrast_feature.expand((batch_size, batch_size, feature_dim)), 0, 1))
                    
                    loss = -nn.LogSoftmax(0)(torch.div(anchor_dot_contrast, self.contrastive_temperature)).diag().sum()
                else:
                    encoder_embedding1 = encoder_out1["encoder_out"]
                    encoder_embedding2 = encoder_out2["encoder_out"][0]
                    batch_size = encoder_embedding2.shape[1]
                    length = encoder_embedding2.shape[0]
                    feature_dim = encoder_embedding2.shape[2]
                    similarity = self.similarity_function(encoder_embedding1.mean(-1),encoder_embedding2.mean(-1)).mean(-1)
                    anchor_dot_contrast = self.similarity_function(encoder_embedding1.expand((length, length, batch_size, feature_dim)).transpose(0,2),
                                                                encoder_embedding2.expand((length, length, batch_size, feature_dim)).transpose(0,2))
                    loss = -nn.LogSoftmax(1)(torch.div(anchor_dot_contrast, self.contrastive_temperature)).diagonal().sum()
                
                return loss, similarity
            
            # encoder_embedding1: [B, 512, 3072]
            # encoder_embedding2: [B, 187, 3072]
            # lengths: [B]
            
            # 创建 mask1: [B, 512]
            mask1 = torch.arange(inputs_embeds.size(1), device=inputs_embeds.device)[None, :] < audio_feature_lengths[:, None]
            mask1 = mask1.unsqueeze(-1)  # [B, 512, 1]

            # masked mean
            masked_sum1 = (inputs_embeds * mask1).sum(dim=1)  # [B, 3072]
            masked_mean1 = masked_sum1 / audio_feature_lengths.unsqueeze(1)     # [B, 3072]

            # 直接对 encoder_embedding2 做 mean
            mean2 = audio_features_4_loss.mean(dim=1)  # 假设它无 padding

            # print(f"loss 111 is : {loss}")

            # ######
            # # 全局余弦相似度
            # similarity = self.similarity_function(masked_mean1, mean2).mean(-1)  # [B]
            # e1 = inputs_embeds.unsqueeze(2)  # [B, 512, 1, 3072]
            # e2 = audio_features_4_loss.unsqueeze(1)  # [B, 1, 187, 3072]
            # pairwise_sim = self.similarity_function(e1, e2)  # [B, 512, 187]
            # mask1 = (torch.arange(512, device=inputs_embeds.device)[None, :] < audio_feature_lengths[:, None]).unsqueeze(-1)
            # anchor_dot_contrast = pairwise_sim * mask1  # [B, 512, 187]
            # loss = -nn.LogSoftmax(1)(torch.div(anchor_dot_contrast, 1.0)).diagonal().sum()
            # ######

            ######
            # === Step 2: 构造 global 对比相似度矩阵 ===
            # similarity_matrix: [B, B]
            similarity_matrix = self.similarity_function(
                masked_mean1.unsqueeze(1),  # [B, 1, D]
                mean2.unsqueeze(0)          # [1, B, D]
            )
            # === Step 3: InfoNCE loss ===
            logits = similarity_matrix / 1.0  # [B, B]
            log_probs = nn.LogSoftmax(dim=1)(logits)
            loss = -log_probs.diagonal().mean()
            ######

            # print(f"loss 222 is : {loss}")
            # exit(0)
            ########


        return CausalLMOutputWithPast(
           loss=loss,
           logits=outputs["audio_features"],
        )

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

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        attention_mask=None,
        inputs_embeds=None,
        cache_position=None,
        position_ids=None,
        use_cache=True,
        **kwargs,
    ):
        # If we have cache: let's slice `input_ids` through `cache_position`, to keep only the unprocessed tokens
        # Exception 1: when passing input_embeds, input_ids may be missing entries
        # Exception 2: some generation methods do special slicing of input_ids, so we don't need to do it here
        if past_key_values is not None:
            if inputs_embeds is not None:  # Exception 1
                input_ids = input_ids[:, -cache_position.shape[0] :]
            elif input_ids.shape[1] != cache_position.shape[0]:  # Default case (the "else", a no op, is Exception 2)
                input_ids = input_ids[:, cache_position]

        if attention_mask is not None and position_ids is None:
            # create position_ids on the fly for batch generation
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past_key_values:
                position_ids = position_ids[:, -input_ids.shape[1] :]

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and cache_position[0] == 0:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids.contiguous()}  # `contiguous()` needed for compilation use cases

        model_inputs.update(
            {
                "position_ids": position_ids,
                "cache_position": cache_position,
                "past_key_values": past_key_values,
                "use_cache": use_cache,
                "attention_mask": attention_mask,
            }
        )
        
        print(kwargs.keys())
        exit(0)
        
        model_inputs.update({"audios": kwargs["audios"]} if "audios" in kwargs.keys() else {})
        
        ########
        model_inputs.update({"input_ids_neg": kwargs["input_ids_neg"]} if "input_ids_neg" in kwargs.keys() else {})
        model_inputs.update({"labels_neg": kwargs["labels_neg"]} if "labels_neg" in kwargs.keys() else {})
        model_inputs.update({"attention_mask_neg": kwargs["attention_mask_neg"]} if "attention_mask_neg" in kwargs.keys() else {})
        model_inputs.update({"audios_neg": kwargs["audios_neg"]} if "audios_neg" in kwargs.keys() else {})
        model_inputs.update({"asr_targets_neg": kwargs["asr_targets_neg"]} if "asr_targets_neg" in kwargs.keys() else {})
        ########

        return model_inputs


AutoConfig.register("ACLlama", ACLlamaConfig)
AutoModelForCausalLM.register(ACLlamaConfig, ACLlamaForCausalLM)
