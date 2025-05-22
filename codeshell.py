from typing import Any, Dict, Iterable, List, Optional, Tuple,Union

import torch
from transformers.activations import ACT2FN
from torch import nn
from .configuration_codeshell import CodeShellConfig

from vllm.attention import Attention, AttentionMetadata
from vllm.config import CacheConfig, LoRAConfig
from vllm.distributed import (get_tensor_model_parallel_rank,
                              get_tensor_model_parallel_world_size)
from vllm.model_executor.layers.activation import SiluAndMul
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import (MergedColumnParallelLinear,
                                               QKVParallelLinear,
                                               RowParallelLinear)
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.quantization.base_config import (
    QuantizationConfig)
from vllm.model_executor.layers.rotary_embedding import get_rope
from vllm.model_executor.layers.sampler import Sampler
from vllm.model_executor.layers.vocab_parallel_embedding import (
    DEFAULT_VOCAB_PADDING_SIZE, ParallelLMHead, VocabParallelEmbedding)
from vllm.model_executor.model_loader.weight_utils import (
    default_weight_loader, kv_cache_scales_loader)
from vllm.model_executor.sampling_metadata import SamplingMetadata
from .sequence import SamplerOutput,IntermediateTensors
from vllm.utils import is_hip, print_warning_once

from .interfaces import SupportsLoRA

import torch
torch.cuda.empty_cache()

class CodeShellAttention(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_attention_heads: int,
        group_query_attention: str,
        num_query_groups: int,
        rope_scaling: Optional[Dict[str, Any]] = None,
        max_position_embeddings: int = 8192,
        cache_config: Optional[CacheConfig] = None,
        bias: bool = False,
        quant_config: Optional[QuantizationConfig] = None,
    ) -> None:
        super().__init__()
        self.mask_value = None
        self.rope_scaling = rope_scaling
        self.max_position_embeddings = max_position_embeddings
        self.group_query_attention = group_query_attention
        self.num_query_groups = num_query_groups

        self.embed_dim = hidden_size
        self.num_heads = num_attention_heads
        self.head_dim = self.embed_dim // self.num_heads
        self.kv_heads = self.num_heads
        self.kv_dim = self.kv_heads * self.head_dim
        self.split_size = self.embed_dim

        if self.head_dim * self.num_heads != self.embed_dim:
            raise ValueError(
                f"`embed_dim` must be divisible by num_heads (got `embed_dim`: {self.embed_dim} and `num_heads`:"
                f" {self.num_heads})."
            )


        self.c_attn = QKVParallelLinear(self.embed_dim,64,self.num_heads,bias=bias,quant_config=quant_config)
        self.c_proj = RowParallelLinear(self.embed_dim, self.embed_dim,bias=bias,quant_config=quant_config)

        self.attn_dropout = nn.Dropout(0.1)
        self.resid_dropout = nn.Dropout(0.1)



        self.rotary_emb = get_rope(
            self.head_dim,
            rotary_dim=self.head_dim,
            max_position=max_position_embeddings,
            base=10000,
            rope_scaling=rope_scaling,
        )

        self.attn = Attention(self.num_heads,
                              self.head_dim,
                              self.split_size,
                              num_kv_heads=self.kv_heads,
                              cache_config=cache_config,
                              quant_config=quant_config)
    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: AttentionMetadata,
    ) -> torch.Tensor:
        # print('000000000000000000000000',hidden_states.shape)
        bsz = 1
        q_len, _ = hidden_states.size()
        qkv,_= self.c_attn(hidden_states)
        qkv = qkv.repeat(1,2)
        query_states, key_states, value_states = qkv.chunk(chunks=3,dim = -1)
        #query_states, key_states, value_states= self.c_attn(hidden_states).split([self.embed_dim, 1024, 1024], dim=-1)
        # q, k, v = qkv.split([self.embed_dim, 1024, 1024], dim=2)
        #query_states, key_states = self.rotary_emb(positions, query_states, key_states)
        # query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        # key_states = key_states.view(bsz, q_len, 8, self.head_dim).transpose(1, 2)
        # value_states = value_states.view(bsz, q_len, 8, self.head_dim).transpose(1, 2)
        # query_states = query_states.reshape(bsz, -1)
        # key_states = key_states.reshape(bsz, -1)
        #print('111111111111111111111',query_states.shape)
        #print('22222222222332',value_states.shape)
        #print('22222222222332111111111111115',key_states.shape)
        query_states, key_states = self.rotary_emb(positions, query_states, key_states)
        query_states = query_states.contiguous()
        key_states = key_states.contiguous()
        value_states =  value_states.contiguous()
        attn_output = self.attn(query_states, key_states, value_states, kv_cache, attn_metadata)
        #print('66666666666666666666',attn_output.shape)
        attn_output = self.attn_dropout(attn_output)
        #print('555555555555555555555555555555',attn_output.shape)
        output,_ = self.c_proj(attn_output)
        output = self.resid_dropout(output)
        return output

class CodeShellMLP(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        quant_config: Optional[QuantizationConfig] = None,
        bias: bool = False,
    ) -> None:
        super().__init__()
        embed_dim = hidden_size
        self.c_fc = RowParallelLinear(
            input_size = embed_dim,
            output_size = intermediate_size,
            bias = bias,
            quant_config=quant_config
        )
        self.c_proj = RowParallelLinear(
            input_size = intermediate_size,
            output_size = embed_dim,
            bias = bias,
            quant_config=quant_config
        )
        self.act_fn = ACT2FN['gelu_pytorch_tanh']
        self.dropout = nn.Dropout(0.1)
    def forward(self,x):
       # print('000000000000000',x.shape)
        x, _ = self.c_fc(x)
        # print('11111111111111',x.shape)
        x = self.act_fn(x)
        # print('22222222222222',x.shape)
        x, _ = self.c_proj(x)
        x= self.dropout(x)
        return x





class CodeShellBlock(nn.Module):
    def __init__(
        self,
        config: CodeShellConfig,
        cache_config: Optional[CacheConfig] = None,
        quant_config: Optional[QuantizationConfig] = None,
    ) -> None:
        super().__init__()
        config = CodeShellConfig()
        self.hidden_size = config.hidden_size
        self.inner_dim =  4 * config.hidden_size
        rope_theta = getattr(config, "rope_theta", 10000)
        rope_scaling = getattr(config, "rope_scaling", None)
        if rope_scaling is not None and getattr(
                config, "original_max_position_embeddings", None):
            rope_scaling["original_max_position_embeddings"] = (
                config.original_max_position_embeddings)
        max_position_embeddings = getattr(config, "max_position_embeddings",
                                          8192)
        attention_bias = getattr(config, "attention_bias", False) or getattr(
            config, "bias", False)
        self.ln_1 = RMSNorm(config.hidden_size, eps=config.layer_norm_epsilon)
        self.attn = CodeShellAttention(
            hidden_size=self.hidden_size,
            num_heads=config.num_attention_heads,
            num_attention_heads = config.num_attention_heads,
            group_query_attention = config.group_query_attention,
            num_query_groups = config.num_query_groups,
            rope_scaling=rope_scaling,
            max_position_embeddings=max_position_embeddings,
            quant_config=quant_config,
            bias=attention_bias,
            cache_config=cache_config,
        )
        self.ln_2 = RMSNorm(config.hidden_size, eps=config.layer_norm_epsilon)
        self.mlp = CodeShellMLP(
            hidden_size=self.hidden_size,
            intermediate_size=self.inner_dim,
            quant_config=quant_config,
            bias=False,
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: AttentionMetadata,
        residual: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # residual = hidden_states
        # print('1234ddd56',residual.shape)
        # hidden_states = self.ln_1(hidden_states)
        # print('1234sdadasd56',hidden_states.shape)
        # attn_outputs = self.attn(
        #     positions,
        #     hidden_states,
        #     kv_cache,
        #     attn_metadata,
        # )
        # attn_output = attn_outputs[0]  # output_attn: a, present, (attentions)

        # outputs = attn_outputs[1:]
        # # residual connection
        # hidden_states = attn_output + residual

        # residual = hidden_states
        # hidden_states = self.ln_2(hidden_states)
        # feed_forward_hidden_states = self.mlp(hidden_states)
        # # residual connection
        # hidden_states = residual + feed_forward_hidden_states
        # print('123456',hidden_states.shape)
        # print('23456789123465', outputs.shape)
        # outputs = (hidden_states,) + outputs[1:]
        if residual is None:
            residual = hidden_states
            hidden_states = self.ln_1(hidden_states)
        else:
            hidden_states, residual = self.ln_1(
                hidden_states, residual)
        hidden_states = self.attn(
            positions,
            hidden_states,
            kv_cache,
            attn_metadata,
        )
        # attn_output = hidden_states[0]
        # hidden_states = attn_output + residual

        residual = hidden_states
        #print('hidden_states',hidden_states.shape)
        hidden_states,residual = self.ln_2(hidden_states,residual)
        hidden_states = self.mlp(hidden_states)
        # residual connection
        # hidden_states = residual + feed_forward_hidden_states

        # # if use_cache:
        # #     outputs = (hidden_states,) + outputs
        # # else:
        # #     outputs = (hidden_states,) + outputs[1:]

        return hidden_states,residual



class CodeShellModel(nn.Module):
    def __init__(
        self,
        config: CodeShellConfig,
        cache_config: Optional[CacheConfig] = None,
        quant_config: Optional[QuantizationConfig] = None,
    ) -> None:
        super().__init__()
        config = CodeShellConfig()
        self.config = config
       # self.group_query_attention = config.group_query_attention
        #self.num_query_groups = config.num_query_groups
        #self.position_embedding_type = config.position_embedding_type
        self.embed_dim = config.hidden_size

        self.vocab_size = config.vocab_size
        self.wte = VocabParallelEmbedding(
            self.vocab_size,self.embed_dim
        )

        self.drop = nn.Dropout(config.embd_pdrop)
        self.h = nn.ModuleList([
            CodeShellBlock(config=config,
                           cache_config=cache_config,
                           quant_config=quant_config)
                           for i in range(config.num_hidden_layers)])
        self.norm = RMSNorm(self.embed_dim, eps=config.layer_norm_epsilon)

    def get_input_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.wte(input_ids)


    def forward(
        self,
        input_ids: Optional[torch.Tensor],
        positions: torch.Tensor,
        kv_caches: List[torch.Tensor],
        attn_metadata: AttentionMetadata,
        inputs_embeds: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if inputs_embeds is not None:
            hidden_states = inputs_embeds
        else:
            hidden_states = self.wte(input_ids)
        hidden_states = self.drop(hidden_states)
        residual = None
        for i in range(len(self.h)):
            layer = self.h[i]
            hidden_states, residual = layer(
                positions,
                hidden_states,
                kv_caches[i],
                attn_metadata,
                residual,
            )
        hidden_states, _ = self.norm(hidden_states, residual)
        return hidden_states


class CodeShellForCausalLM(nn.Module):
    def __init__(
        self,
        config: CodeShellConfig,
        cache_config: Optional[CacheConfig] = None,
        quant_config: Optional[QuantizationConfig] = None,
    ) -> None:
        super().__init__()
        config = CodeShellConfig()
        self.config = config

        self.transformer = CodeShellModel(
            config,
            cache_config,
            quant_config,
            )
        self.unpadded_vocab_size = config.vocab_size
        self.lm_head = ParallelLMHead(config.vocab_size,config.n_embd, bias=False)
        self.logits_processor = LogitsProcessor(self.unpadded_vocab_size,
                                                config.vocab_size, 1.0)
        self.sampler = Sampler()


    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        kv_caches: List[torch.Tensor],
        attn_metadata: AttentionMetadata,
        intermediate_tensors: Optional[IntermediateTensors] = None,
)-> Union[torch.Tensor, IntermediateTensors]:
        model_output = self.transformer(input_ids, positions, kv_caches,
                                  attn_metadata, intermediate_tensors)
        return model_output

    def compute_logits(self, hidden_states: torch.Tensor,
                       sampling_metadata: SamplingMetadata) -> torch.Tensor:

        logits = self.logits_processor(self.lm_head, hidden_states,
                                       sampling_metadata)
        return logits

    def sample(
        self,
        logits: torch.Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> Optional[SamplerOutput]:
        next_tokens = self.sampler(logits, sampling_metadata)
        return next_tokens

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            (".c_attn", ".q_proj", "q"),
            (".c_attn", ".k_proj", "k"),
            (".c_attn", ".v_proj", "v"),
        ]

        params_dict = dict(self.named_parameters())
        # for param_name, param in params_dict.items():
        #     print(f"Name: {param_name}, Shape: {param.shape}")
        for name, loaded_weight in weights:

            if "rotary_emb.inv_freq" in name:
                continue
            if ("rotary_emb.cos_cached" in name
                    or "rotary_emb.sin_cached" in name):
                # Models trained using ColossalAI may include these tensors in
                # the checkpoint. Skip them.
                continue
            for (param_name, weight_name, shard_id) in stacked_params_mapping:
                if weight_name not in name:
                    continue
                name = name.replace(weight_name, param_name)
                # Skip loading extra bias for GPTQ models.
                if name.endswith(".bias") and name not in params_dict:
                    continue
                try:
                    param = params_dict[name]
                    weight_loader = param.weight_loader
                    # print(f"Loading {name} with shape {param.shape} from {weight_name} with shape {loaded_weight.shape}")
                    weight_loader(param, loaded_weight, shard_id)
                except KeyError:
                    pass
                break
            else:
                # Skip loading extra bias for GPTQ models.
                if name.endswith(".bias") and name not in params_dict:
                    continue
                # Remapping the name of FP8 kv-scale.
                if name.endswith("kv_scale"):
                    remapped_kv_scale_name = name.replace(
                        ".kv_scale", ".attn.kv_scale")
                    if remapped_kv_scale_name not in params_dict:
                        print_warning_once(
                            f"Found kv scale in the checkpoint (e.g. {name}), "
                            "but not found the expected name in the model "
                            f"(e.g. {remapped_kv_scale_name}). kv-scale is "
                            "not loaded.")
                        continue
                    else:
                        name = remapped_kv_scale_name
                try:
                    param = params_dict[name]
                    weight_loader = getattr(param, "weight_loader",
                                            default_weight_loader)
                    weight_loader(param, loaded_weight)
                except KeyError:
                    pass

                # Skip load
    # If this function is called, it should always initialize KV cache scale
    # factors (or else raise an exception). Thus, handled exceptions should
    # make sure to leave KV cache scale factors in a known good (dummy) state

