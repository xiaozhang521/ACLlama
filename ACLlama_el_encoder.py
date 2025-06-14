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

######
from transformers.models.whisper.modeling_whisper import WhisperAttention
from transformers.activations import ACT2FN
import numpy as np
######

IGNORE_TOKEN_ID = LabelSmoother.ignore_index

class ACLlamaConfig(LlamaConfig):
    model_type = "ACLlama"
    


def load_whisper(audio_tower_name):
    model = WhisperModel.from_pretrained(
            audio_tower_name,torch_dtype=torch.bfloat16, low_cpu_mem_usage=True).to('cuda')
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

########
# Copied from transformers.models.mbart.modeling_mbart.MBartEncoderLayer with MBart->Whisper, MBART->WHISPER
class MYEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward, dropout, rms_norm_eps=1e-03):
        super().__init__()
        self.embed_dim = d_model

        self.self_attn = WhisperAttention(
            embed_dim=self.embed_dim,
            num_heads=nhead,
            dropout=dropout,
            config=None,
        )
        self.self_attn_layer_norm = nn.LayerNorm(self.embed_dim, eps=rms_norm_eps)
        self.dropout = dropout
        self.activation_fn = ACT2FN["gelu"]
        self.activation_dropout = dropout
        self.fc1 = nn.Linear(self.embed_dim, dim_feedforward)
        self.fc2 = nn.Linear(dim_feedforward, self.embed_dim)
        self.final_layer_norm = nn.LayerNorm(self.embed_dim, eps=rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask = None,
        layer_head_mask = None,
        output_attentions = False,
    ) -> torch.Tensor:
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`torch.FloatTensor`): attention mask of size
                `(batch, 1, tgt_len, src_len)` where padding elements are indicated by very large negative values.
            layer_head_mask (`torch.FloatTensor`): mask for attention heads in a given layer of size
                `(encoder_attention_heads,)`.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
        """
        residual = hidden_states
        hidden_states = self.self_attn_layer_norm(hidden_states)
        
        # print(f"hidden_states is : {hidden_states}")
        # print(f"attention_mask is : {attention_mask}")
        
        hidden_states, attn_weights, _ = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            layer_head_mask=layer_head_mask,
            output_attentions=output_attentions,
        )
        
        # print(f"hidden_states after attn is : {hidden_states}")

        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        hidden_states = residual + hidden_states

        # print(f"hidden_states before ffn  ln is : {hidden_states}")
        # print("before norm:", hidden_states.min(), hidden_states.max(), hidden_states.mean())

        residual = hidden_states
        hidden_states = self.final_layer_norm(hidden_states)
        
        # print(f"hidden_states before ffn is : {hidden_states}")

        hidden_states = self.activation_fn(self.fc1(hidden_states))
        hidden_states = nn.functional.dropout(hidden_states, p=self.activation_dropout, training=self.training)
        hidden_states = self.fc2(hidden_states)
        
        # print(f"hidden_states after ffn is : {hidden_states}")

        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        hidden_states = residual + hidden_states

        if hidden_states.dtype == torch.bfloat16 and (
            torch.isinf(hidden_states).any() or torch.isnan(hidden_states).any()
        ):
            clamp_value = torch.finfo(hidden_states.dtype).max - 1000
            hidden_states = torch.clamp(hidden_states, min=-clamp_value, max=clamp_value)

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (attn_weights,)

        return outputs
########

class ACLlamaModel(LlamaModel):
    config_class = ACLlamaConfig

    def __init__(self, config: LlamaConfig):
        super(ACLlamaModel, self).__init__(config)

        # if hasattr(config, "audio_tower"):
        #     self.audio_tower = [load_whisper(config.audio_tower)]
        self.audio_tower = WhisperModel.from_pretrained(
        config.audio_tower, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True).to('cuda')
        self.audio_tower.config.forced_decoder_ids = None

        if hasattr(config, "adapter_size"):
            #self.down_sampler = Conv1dSubsampler(config.adapter_size, config.hidden_size // 2, config.hidden_size // 2, [5])
            #self.conv1 = nn.Conv1d(1280, config.hidden_size//2, kernel_size=3, stride=2, padding=1)
            #self.conv2 = nn.Conv1d(4096, 4096, kernel_size=3, stride=2, padding=1)
            self.mm_projector1 = nn.Linear(config.adapter_size*2 , config.hidden_size)
            # self.mm_projector1 = nn.Sequential(
            #     nn.Linear(config.adapter_size * 2, config.hidden_size),
            #     nn.LayerNorm(config.hidden_size, eps=1e-5),
            #     nn.Tanh()
            # )
            #self.relu = nn.ReLU()
            #self.mm_projector2 = nn.Linear(config.hidden_size , config.hidden_size)
            # asr_encoder_layer = nn.TransformerEncoderLayer(
            #     d_model=config.hidden_size,
            #     nhead=config.num_attention_heads,
            #     dim_feedforward=config.hidden_size*2,
            #     dropout=0.1,
            #     norm_first=True
            # )
            # self.lbm =  LookBackModule(config)
            self.out_norm = nn.LayerNorm(config.hidden_size, eps=1e-3)
            self.audio_feature_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
            # self.asr_transformer_encoder = nn.TransformerEncoder(asr_encoder_layer, num_layers=1)

        ########
            self.asr_transformer_encoder = MYEncoderLayer(
                d_model=config.hidden_size,
                nhead=config.num_attention_heads,
                dim_feedforward=config.hidden_size*2,
                dropout=0.1,
            )
        # self.text_projector = nn.Sequential(nn.Linear(config.hidden_size , config.hidden_size*2),
        #                                     ACT2FN["gelu"],
        #                                     nn.Linear(config.hidden_size*2 , config.hidden_size))
        self.avg_pooler = nn.AvgPool1d(2, stride=2)
        del self.layers
        self.act_func = ACT2FN["gelu"]
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
        #######
        input_ids_neg = None,
        attention_mask_neg = None,
        #######
    ) -> Union[Tuple, BaseModelOutputWithPast]:

        # HACK: replace back original embeddings for LLaAA pretraining
        orig_embeds_params = getattr(self, 'orig_embeds_params', None)

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        ######
        if input_ids_neg is not None:
            input_ids_neg = self.embed_tokens(input_ids_neg)
        ######

        # audio_tower = getattr(self, 'audio_tower', None)
        # if audio_tower is not None and (input_ids.shape[1] != 1 or self.training) and audios is not None:
        if (input_ids.shape[1] != 1 or self.training) and audios is not None:
            # audio_tower = audio_tower[0]  # HACK: for FSDP
            audio_list=[]
            
            audio_config = self.audio_tower.config
            #with torch.no_grad():
            #    audio_features = audio_tower.encoder(audios).last_hidden_state
            #for audio_feature in audio_features:
            #    audio_feature = audio_feature.unsqueeze(0)
            
            # for audio in audios:
            #     audio = audio.unsqueeze(0)
            #     audio_feature_t = self.audio_tower.encoder(audio).last_hidden_state

            #     audio_feature = audio_feature_t.view(audio_feature_t.shape[0], audio_feature_t.shape[1]//2, 2 * audio_feature_t.shape[2])
            #     audio_feature = self.mm_projector1(audio_feature)
            #     audio_feature = self.asr_transformer_encoder(audio_feature)
            #     audio_feature = self.out_norm(audio_feature)
            #     audio_list.append(audio_feature[0])

            # audio_features = torch.stack(audio_list, dim=0)

 
            ######
            audio_feature = self.audio_tower.encoder(audios).last_hidden_state.to(audios.dtype)
            audio_feature = audio_feature.view(audio_feature.shape[0], audio_feature.shape[1] // 2, 2 * audio_feature.shape[2])
            
            audio_feature = self.mm_projector1(audio_feature)
            audio_feature = F.layer_norm(audio_feature, audio_feature.shape[-1:])  # or nn.LayerNorm
            audio_feature = self.act_func(audio_feature)
            
            # print(f"audio_feature is : {audio_feature}")
            
            audio_feature = self.asr_transformer_encoder(audio_feature, None, None)[0]
            audio_features = self.out_norm(audio_feature)
            audio_features = self.act_func(audio_feature)
            
            # print(f"audio_features after transencoder is : {audio_features}")
            
            audio_feature_lengths = attention_mask.int().sum(dim=1)  # shape: (batch_size,)
                        
            audio_features_4_loss = audio_features.clone().permute(0, 2, 1)
            while audio_features_4_loss.size(2) // 2 - 1 > audio_feature_lengths.max():
                audio_features_4_loss = self.avg_pooler(audio_features_4_loss)
            audio_features_4_loss = audio_features_4_loss.permute(0, 2, 1)
            
            audio_feature_lengths_neg = attention_mask_neg.int().sum(dim=1)  # shape: (batch_size,)
            
            # print(f"audio_features_4_loss is : {audio_features_4_loss}")
            
            # inputs_embeds = self.text_projector(inputs_embeds)
            # inputs_embeds = F.layer_norm(inputs_embeds, inputs_embeds.shape[-1:])  # or nn.LayerNorm

            predict_logits = self.audio_feature_head(audio_features)
            ######
            
        return_state = {"audio_features": predict_logits}
        #########
        return_state_update = {"audio_feature_lengths": audio_feature_lengths, "audio_features_4_loss": audio_features_4_loss, "inputs_embeds": inputs_embeds, "input_ids_neg": input_ids_neg, "audio_feature_lengths_neg": audio_feature_lengths_neg}
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
        self.logit_scale = nn.Parameter(torch.ones(1) * np.log(1 / 0.07))
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
        
        ######
        # input_ids = input_ids.to(self.device)
        # attention_mask = attention_mask.to(self.device)
        # print(f"input_ids_neg is : {input_ids_neg}")
        ######
        
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
            audios=audios,
            #######
            input_ids_neg=input_ids_neg,
            attention_mask_neg=attention_mask_neg,
            #######
        )
        
        #######
        audio_feature_lengths = outputs.pop("audio_feature_lengths")
        audio_feature_lengths_neg = outputs.pop("audio_feature_lengths_neg")
        audio_features_4_loss = outputs.pop("audio_features_4_loss")
        inputs_embeds = outputs.pop("inputs_embeds")
        input_ids_neg_return = outputs.pop("input_ids_neg")
        #######
        
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
                        blank=self.model.audio_tower.config.audio_patch_token,
                        reduction='mean',
                        zero_infinity=True,
                    )
            else:
                loss_asr=0
            
            #########
            # audio_features_4_loss: [B, T, D] → audio mean
            mean2 = audio_features_4_loss.mean(dim=1)  # [B, D]
            mean2 = F.normalize(mean2, dim=1)

            # inputs_embeds: [2B, L, D]
            # text attention mask（注意，扩展到 2B）
            if input_ids_neg_return is not None:
                inputs_embeds = torch.cat((inputs_embeds, input_ids_neg_return), dim=0)  # [2B, L, D]

            # 手动构造 attention mask（注意长度也扩展）
            text_lengths = torch.cat([audio_feature_lengths, audio_feature_lengths_neg], dim=0)  # [2B]
            mask1 = torch.arange(inputs_embeds.size(1), device=inputs_embeds.device)[None, :] < text_lengths[:, None]  # [2B, L]
            mask1 = mask1.unsqueeze(-1).type_as(inputs_embeds)  # [2B, L, 1]

            # masked mean pooling
            masked_sum1 = (inputs_embeds * mask1).sum(dim=1)  # [2B, D]
            masked_mean1 = masked_sum1 / (mask1.sum(dim=1) + 1e-8)  # [2B, D]
            masked_mean1 = F.normalize(masked_mean1, dim=1)

            temperature = 1.0
            # logits: audio anchor → text pos+neg
            logits = torch.matmul(mean2, masked_mean1.T)  # [B, 2B]
            labels = torch.arange(mean2.size(0), device=mean2.device)  # [B]
            loss_contrastive = F.cross_entropy(logits / temperature, labels)
            #########


            # #########
            # # print(f"audio_features_4_loss is : {audio_features_4_loss}")
            # # 创建 mask1: [B, 512]
            # mask1 = torch.arange(inputs_embeds.size(1), device=inputs_embeds.device)[None, :] < audio_feature_lengths[:, None]
            # mask1 = mask1.unsqueeze(-1)  # [B, 512, 1]

            # # masked mean
            # masked_sum1 = (inputs_embeds * mask1).sum(dim=1)  # [B, 3072]
            # masked_mean1 = masked_sum1 / audio_feature_lengths.unsqueeze(1)     # [B, 3072]

            # # 直接对 encoder_embedding2 做 mean
            # mean2 = audio_features_4_loss.mean(dim=1)  # 假设它无 padding
            # # print(f"masked_mean1 is : {masked_mean1}")
            # # print(f"mean2 is : {mean2}")

            # masked_mean1 = F.normalize(masked_mean1, dim=1)
            # mean2 = F.normalize(mean2, dim=1)

            # ######
            # # === Step 2: 构造 global 对比相似度矩阵 ===
            # # similarity_matrix: [B, B]
            # similarity_matrix = self.similarity_function(
            #     masked_mean1.unsqueeze(1),  # [B, 1, D]
            #     mean2.unsqueeze(0)          # [1, B, D]
            # )
            # # === Step 3: InfoNCE loss ===
            # logits = similarity_matrix / 1.0  # [B, B]
            # log_probs = nn.LogSoftmax(dim=1)(logits)
            # # print(f"log_probs is : {log_probs}")
            # loss_contrastive = -log_probs.diagonal().mean()
            # # print(f"loss_contrastive is : {loss_contrastive}")

            # # inputs_embeds_filter = inputs_embeds[:, :audio_features_4_loss.size(1), :]
            
            # # mask1 = torch.arange(inputs_embeds_filter.size(1), device=inputs_embeds_filter.device)[None, :] < audio_feature_lengths[:, None]
            # # mask1 = mask1.unsqueeze(-1)  # [B, 512, 1]
            
            # # batch_size = audio_features_4_loss.shape[1]
            # # length = audio_features_4_loss.shape[0]
            # # feature_dim = audio_features_4_loss.shape[2]
            # # similarity = self.similarity_function(inputs_embeds_filter.mean(-1), audio_features_4_loss.mean(-1)).mean(-1)
            # # anchor_dot_contrast = self.similarity_function(inputs_embeds_filter.expand((length, length, batch_size, feature_dim)).transpose(0,2).to(torch.float32),
            # # audio_features_4_loss.expand((length, length, batch_size, feature_dim)).transpose(0,2).to(torch.float32))

            # # loss_contrastive = -nn.LogSoftmax(1)(anchor_dot_contrast.to(audio_features_4_loss.dtype)).diagonal().sum()
            # #########


            # #########
            # inputs_embeds_filter = inputs_embeds[:, :audio_features_4_loss.size(1), :]
            # mask1 = torch.arange(inputs_embeds_filter.size(1), device=inputs_embeds_filter.device)[None, :] < audio_feature_lengths[:, None]

            # with torch.cuda.amp.autocast(enabled=False):  # 禁用 autocast
            #     # audio_features_4_loss[~mask1] = 0
            #     # inputs_embeds_filter[~mask1] = 0
                
            #     # audio_features = audio_features_4_loss.mean(1).to(torch.float32)
            #     # text_features = inputs_embeds_filter.mean(1).to(torch.float32)
                
            #     mask1 = mask1.unsqueeze(-1)  # shape: [B, L, 1]
            #     len_x = mask1.sum(dim=1)  # number of valid positions per sample [B, 1]

            #     # mask 并求mean 去length
            #     audio_features_4_loss = audio_features_4_loss * mask1  # masked-out positions will become 0
            #     sum_audio_features_4_loss = audio_features_4_loss.sum(dim=1)  # sum over valid positions
            #     audio_features = sum_audio_features_4_loss / (len_x + 1e-8)  # shape: [B, D]
                
            #     # print(f"audio_features is 111 : {audio_features}")
                
            #     inputs_embeds_filter = inputs_embeds_filter * mask1  # masked-out positions will become 0
            #     sum_inputs_embeds_filter = inputs_embeds_filter.sum(dim=1)  # sum over valid positions
            #     text_features = sum_inputs_embeds_filter / (len_x + 1e-8)  # shape: [B, D]

            #     # print(f"text_features is 111 : {text_features}")

            #     # normalized features
            #     audio_features = audio_features / audio_features.norm(dim=1, keepdim=True).clamp(min=1e-8)
            #     text_features = text_features / text_features.norm(dim=1, keepdim=True).clamp(min=1e-8)
            #     # audio_features_4_loss = F.normalize(audio_features_4_loss, dim=1)
            #     # text_features = F.normalize(text_features, dim=1)

            #     # print(f"audio_features is 222 : {audio_features}")
            #     # print(f"text_features is 222 : {text_features}")

            #     # cosine similarity as logits
            #     logit_scale = self.logit_scale.exp()
            #     logits_per_audio = logit_scale * audio_features @ text_features.t()
            #     logits_per_text = logits_per_audio.t()

            #     # print(f"logits_per_audio is : {logits_per_audio}")

            #     labels = torch.arange(audio_features.size(0), device=logits_per_audio.device)
            #     loss_fn = nn.CrossEntropyLoss()
            #     loss_i = loss_fn(logits_per_audio, labels)
            #     loss_t = loss_fn(logits_per_text, labels)
            #     loss_contrastive = (loss_i + loss_t)/2
                
            #     # print(f"loss_i is : {loss_i}")
            #     # print(f"loss_t is : {loss_t}")
            # ########
            
            # ########
            # alpha = 0.5
            # loss_mse = F.mse_loss(audio_embed, text_embed.detach())
            # loss = loss_contrastive + alpha * loss_mse
            # ########
            
            # loss = loss_contrastive + loss_asr
            loss = loss_contrastive

        return CausalLMOutputWithPast(
           loss=loss,
           logits=asr_logits,
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
