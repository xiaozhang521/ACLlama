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


def padding_tensor(tensor, length, dim=0, pad=False):

    if length == 0:
        return tensor
    
    assert length > 0, f"Wrong padding length: {length}"

    shape = list(tensor.shape)
    assert dim < len(shape), f"dim {dim} out of shape {shape}"
    shape[dim] = length
    padding_tensor = torch.cat(
        (
            tensor,
            torch.full(tuple(shape), pad, dtype=tensor.dtype, device=tensor.device)
        ), 
        dim=dim
    )
    return padding_tensor


class T2ULlamaConfig(LlamaConfig):
    model_type = "T2ULlama"
    
class T2ULlamaForCausalLM(LlamaForCausalLM):
    config_class = T2ULlamaConfig

    def __init__(self, config, embedding_weight=None):
        
        self.current_step = 0
        self.log = {}

        super(LlamaForCausalLM, self).__init__(config)
        self.config = config
        self.training_stage = config.unit_output
        self.pad_token_id = 128009

        llama_config = T2ULlamaConfig(**config.to_dict(), 
                                        batch_first=True, 
                                        norm_first=True
                                    )
        llama_config.architectures = ["T2ULlamaForCausalLM"]
        llama_config.pad_token_id = self.pad_token_id
        llama_config.vocab_size += llama_config.unit_vocab_size
        #######################################################
        llama_config.unit_model = "small"
        llama_config.max_position_embeddings = 2048     # 1024 1536 2048       # origin 1024 reduced 512
        #######################################################
        if hasattr(llama_config, "unit_model"):
            if llama_config.unit_model == "large":
                llama_config.num_hidden_layers = 2
                # llama_config.hidden_size = 4096
                # llama_config.num_attention_heads = 32
                # llama_config.intermediate_size = 14336
                # llama_config.head_dim = llama_config.hidden_size // llama_config.num_attention_heads
            elif llama_config.unit_model == "tiny":
                llama_config.num_hidden_layers = 4
                llama_config.hidden_size = 512
                llama_config.num_attention_heads = 8
                llama_config.intermediate_size = 2048
                llama_config.head_dim = llama_config.hidden_size // llama_config.num_attention_heads
            else:
                llama_config.num_hidden_layers = 6
                llama_config.hidden_size = 512
                llama_config.num_attention_heads = 8
                llama_config.intermediate_size = 2048
                llama_config.head_dim = llama_config.hidden_size // llama_config.num_attention_heads
        else:
            llama_config.num_hidden_layers = 6
            llama_config.hidden_size = 512
            llama_config.num_attention_heads = 8
            llama_config.intermediate_size = 2048
            llama_config.head_dim = llama_config.hidden_size // llama_config.num_attention_heads
        # print(llama_config)
        
        self.model = LlamaModel(llama_config)
        # share embedding 0501 by kkq
        self.model.embed_tokens = nn.Embedding(num_embeddings=config.vocab_size, embedding_dim=config.hidden_size, padding_idx=self.pad_token_id)   # redefine
        self.unit_embedding = nn.Linear(config.hidden_size, llama_config.unit_vocab_size, bias=False) 
        self.adapter = nn.Linear(config.hidden_size, llama_config.hidden_size, bias = True) 
        self.lm_head = nn.Linear(llama_config.hidden_size, llama_config.vocab_size, bias=False)

        if self.training_stage == "pretrain":
            # Π-Model + Dropout 
            # self.embedding_dropout = nn.Dropout(0.3)
            # self.perturb = nn.Sequential(
            #     nn.Linear(config.hidden_size, config.hidden_size // 2),
            #     nn.ReLU(),
            #     nn.Linear(config.hidden_size // 2, config.hidden_size),
            #     nn.Softplus(),
            # )
            pass
        elif self.training_stage == "finetune":
            self.aligner_MLP = nn.Sequential(
                nn.Linear(config.hidden_size, config.intermediate_size),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(config.intermediate_size, config.hidden_size),
            )
            torch.nn.init.ones_(self.aligner_MLP[0].weight)
            torch.nn.init.zeros_(self.aligner_MLP[0].bias)
            torch.nn.init.ones_(self.aligner_MLP[3].weight)
            torch.nn.init.zeros_(self.aligner_MLP[3].bias)

        # Initialize weights and apply final processing
        self.post_init()

    def get_model(self):
        return self.model

    def insert_text_embedding(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        text_labels: Optional[torch.LongTensor] = None,
        shift_text_labels: Optional[torch.LongTensor] = None,
        shift_text_hidden_states: Optional[torch.FloatTensor] = None,
        do_task: str = None,
        **kwargs: dict,
    ):  

        if inputs_embeds == None:
            # inputs_embeds = self.model.embed_tokens(input_ids)
            # share embedding 0501 by kkq
            embed_tokens_weight = torch.cat(
                [
                    self.model.embed_tokens.weight.detach(), self.unit_embedding.weight
                ],
                dim = 0,
            )
            # print(embed_tokens_weight, embed_tokens_weight.shape)
            inputs_embeds = F.embedding(input_ids, embed_tokens_weight, padding_idx=self.pad_token_id)
        
        emb_loss = None
        if do_task == "pretrain":
            if self.training:
                if hasattr(self, "embedding_dropout"):
                    # 输入文本的 Embedding 位置
                    emb_origin_mask = text_labels != -100
                    origin_padding_length = labels.shape[-1] - emb_origin_mask.shape[-1]
                    extend_emb_origin_mask = padding_tensor(emb_origin_mask, origin_padding_length, 1, False)
                    extend_emb_origin_mask = ~extend_emb_origin_mask.unsqueeze(-1).expand_as(inputs_embeds)

                    # Π-Model + noise
                    log_var = self.perturb(inputs_embeds)
                    perturbed_inputs_embeds_2 = inputs_embeds + torch.randn_like(inputs_embeds) * (torch.exp(0.5 * log_var) + 1e-6)
                    # Π-Model + dropout
                    perturbed_inputs_embeds_1 = self.embedding_dropout(inputs_embeds)
                    perturbed_inputs_embeds_2 = self.embedding_dropout(perturbed_inputs_embeds_2)
                    # 还原非 text input 位置的 token embedding （只对 input text 做扰动）
                    perturbed_inputs_embeds_1 = torch.where(extend_emb_origin_mask, inputs_embeds, perturbed_inputs_embeds_1)
                    perturbed_inputs_embeds_2 = torch.where(extend_emb_origin_mask, inputs_embeds, perturbed_inputs_embeds_2)

                    inputs_embeds = torch.cat(
                        (perturbed_inputs_embeds_1, perturbed_inputs_embeds_2),
                        dim=0,
                    )   # 沿batch纬度拼接

                    kl_loss = -0.5 * (1 + log_var - log_var.exp()).mean(dim=-1).sum(dim=-1).mean()  # 正太分布约束
                    contrastive_loss = (1 - F.cosine_similarity(perturbed_inputs_embeds_1, perturbed_inputs_embeds_2, dim=-1)).sum(dim=-1).mean()   # 相似性约束
                    emb_loss = kl_loss + contrastive_loss

                    if kl_loss.device == torch.device("cuda:0"):
                        self.log["kl_loss"] = kl_loss.item()
                        self.log["std"] = torch.exp(0.5 * log_var).mean().item()
                        self.log["contrastive_loss"] = contrastive_loss.item()

            pass
        elif do_task == "finetune":
            # 需要 text 部分替换为 S2T 输出的 text embedding
            inputs_embeds = inputs_embeds.detach()
            inputs_embeds_refer = inputs_embeds.clone().detach()
            shift_text_hidden_states = self.aligner_MLP(shift_text_hidden_states)   # 对齐纬度
            # 获取 Text embedding 的位置 Mask
            emb_origin_mask = text_labels != -100    # get output text pos
            emb_shift_mask = shift_text_labels != -100

            # 补充 Text emb 和 Mask 长度至 T2U 模型输入长度
            origin_padding_length = labels.shape[-1] - emb_origin_mask.shape[-1]
            shift_padding_length = labels.shape[-1] - emb_shift_mask.shape[-1]
            
            extend_emb_origin_mask = padding_tensor(emb_origin_mask, origin_padding_length, 1, False)
            extend_emb_shift_mask = padding_tensor(emb_shift_mask, shift_padding_length, 1, False)
            extend_shift_text_hidden_states = padding_tensor(shift_text_hidden_states, shift_padding_length, 1, 1e-9)
            # check
            extend_text_labels = padding_tensor(text_labels, origin_padding_length, 1, -100)
            extend_shift_text_labels = padding_tensor(shift_text_labels, shift_padding_length, 1, -100)
            # 检测input中的 text label 是否与 hidden state 中的 label 对齐？
            assert torch.equal(
                extend_text_labels[extend_emb_origin_mask], 
                extend_shift_text_labels[extend_emb_shift_mask]
            ), "{}\n{}\n{}\n{}".format(labels, extend_emb_origin_mask, extend_shift_text_labels, extend_emb_shift_mask)
            # S2T 模型输出shift_text_hidden_states 包含speech token，所以需要通过Mask将 shift_text_labels 其对齐回 text_labels
            inputs_embeds[extend_emb_origin_mask.unsqueeze(-1).expand_as(inputs_embeds)] = \
                extend_shift_text_hidden_states[extend_emb_shift_mask.unsqueeze(-1).expand_as(extend_shift_text_hidden_states)].to(dtype=inputs_embeds.dtype)
            
            if self.training:
                contrastive_loss = (1 - F.cosine_similarity(inputs_embeds, inputs_embeds_refer, dim=-1)).sum(-1).mean() # 相似性约束
                emb_loss = contrastive_loss
                if emb_loss.device == torch.device("cuda:0"):
                    self.log["contrastive_loss"] = contrastive_loss.item()
                pass
        else:
            pass

        # 将 s2t 纬度映射至 t2u
        inputs_embeds = self.adapter(inputs_embeds)
        return (emb_loss, inputs_embeds)
    
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
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if inputs_embeds == None:
            # inputs_embeds = self.model.embed_tokens(input_ids)
            # share embedding 0501 by kkq
            embed_tokens_weight = torch.cat(
                [
                    self.model.embed_tokens.weight.detach(), self.unit_embedding.weight
                ],
                dim = 0,
            )
            # print(embed_tokens_weight, embed_tokens_weight.shape)
            inputs_embeds = F.embedding(input_ids, embed_tokens_weight, padding_idx=self.pad_token_id)
            inputs_embeds = self.adapter(inputs_embeds)
        
        if self.training:
            BATCH_SIZE = labels.shape[0]
            assert inputs_embeds.shape[0] % BATCH_SIZE == 0
            if BATCH_SIZE * 2 == inputs_embeds.shape[0]:
                attention_mask = torch.cat(
                    (attention_mask, attention_mask),
                    dim=0
                )
        
        outputs = self.model(
            input_ids=None,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        hidden_states = outputs[0]
        logits = self.lm_head(hidden_states)

        # 计算 Loss
        loss = None
        cr_loss = None
        if labels != None:
            if self.training and BATCH_SIZE * 2 == hidden_states.shape[0]:
                # 使用了 一致性正则化
                perturbed_hidden_states = hidden_states[BATCH_SIZE:]
                hidden_states = hidden_states[:BATCH_SIZE]
                assert hidden_states.shape == perturbed_hidden_states.shape
                # cr_loss = F.mse_loss(perturbed_hidden_states, hidden_states, reduction='none').mean(dim=[1,2]).sum()
                cr_loss = (
                    ((perturbed_hidden_states - hidden_states)**2) / (hidden_states**2 + 1e-8)
                ).mean(dim=-1).sum(-1).mean()
                self.log["cr_loss"] = cr_loss.item()

                labels = torch.cat(
                    (labels, labels),
                    dim=0,
                )

            shift_labels = labels

            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = shift_labels[..., 1:].contiguous()

            loss_fct = CrossEntropyLoss()
        
            shift_logits = shift_logits.view(-1, (self.config.vocab_size + self.config.unit_vocab_size))
            shift_labels = shift_labels.view(-1)
            shift_labels = shift_labels.to(shift_logits.device)
            
            if BATCH_SIZE * 2 == logits.shape[0]:
                LENGTH = shift_labels.shape[0] // 2
                loss_1 = loss_fct(shift_logits[:LENGTH], shift_labels[:LENGTH])
                loss_2 = loss_fct(shift_logits[LENGTH:], shift_labels[LENGTH:])
                loss = 1.0 * ( loss_1 + loss_2 )
                self.log["loss_1"] = loss_1.item()
                self.log["loss_2"] = loss_2.item()
            else:
                loss = loss_fct(shift_logits, shift_labels)
                
            if loss.device == torch.device("cuda:0"):
                self.log["unit_loss"] = loss.item()

            if cr_loss != None:
                target_scale = loss.item() * 0.2
                cr_loss_weight = target_scale / cr_loss.item() if cr_loss > target_scale else 1.0
                loss = loss + cr_loss_weight * cr_loss

            if loss.device == torch.device("cuda:0") and (self.current_step - 10) % 100 == 0:
                print(self.log, loss.device)
        
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

AutoConfig.register("T2ULlama", T2ULlamaConfig)
AutoModelForCausalLM.register(T2ULlamaConfig, T2ULlamaForCausalLM)
