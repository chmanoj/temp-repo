import torch
import torch.nn as nn
from transformers import T5Config, T5ForConditionalGeneration
from transformers.models.t5.modeling_t5 import T5Attention
import torch.nn.functional as F

class T5OptimizedAttention(nn.Module):
    def __init__(self, original_attn):
        super().__init__()
        # Copy necessary attributes from original attention module
        self.n_heads = original_attn.n_heads
        self.is_decoder = original_attn.is_decoder
        
        # Get dimensions from linear layers
        self.d_model = original_attn.q.in_features
        # The key layer projects to n_heads * d_kv
        self.d_kv = original_attn.k.out_features // self.n_heads
        
        # Copy the linear layers
        self.q = original_attn.q
        self.k = original_attn.k
        self.v = original_attn.v
        self.o = original_attn.o
        
        # We don't use relative attention bias in this version
        self.has_relative_attention_bias = False

    def forward(self, hidden_states, key_value_states=None, **kwargs):
        # Determine the actual states to use for keys and values
        actual_key_value_states = key_value_states if key_value_states is not None else hidden_states
        
        # Compute Q, K, V
        q = self.q(hidden_states)
        k = self.k(actual_key_value_states)
        v = self.v(actual_key_value_states)
        
        # Reshape to [batch_size, seq_len, num_heads, head_dim]
        batch_size, tgt_len, _ = hidden_states.size()
        q = q.view(batch_size, -1, self.n_heads, self.d_kv).transpose(1, 2)
        k = k.view(batch_size, -1, self.n_heads, self.d_kv).transpose(1, 2)
        v = v.view(batch_size, -1, self.n_heads, self.d_kv).transpose(1, 2)
        
        # Apply PyTorch's optimized attention
        causal = (key_value_states is None) and self.is_decoder
        attn_output = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=None,
            dropout_p=0.0,
            is_causal=causal
        )
        
        # Reshape back to original dimensions
        attn_output = attn_output.transpose(1, 2).reshape(batch_size, -1, self.d_model)
        attn_output = self.o(attn_output)
        
        # Return tuple to match original T5Attention output format
        # (hidden_states, attention_weights)
        return (attn_output, None)

def replace_attention_with_optimized(model):
    for name, module in model.named_modules():
        if isinstance(module, T5Attention):
            parent_name, child_name = name.rsplit('.', 1)
            parent = model.get_submodule(parent_name)
            optimized_attn = T5OptimizedAttention(module)
            setattr(parent, child_name, optimized_attn)

def create_optimized_model():
    # Create T5 model with optimized attention
    config = T5Config.from_pretrained("t5-small")
    config.use_cache = False  # Disable KV caching for fair comparison
    model_optimized = T5ForConditionalGeneration(config)
    replace_attention_with_optimized(model_optimized)
    return model_optimized

# # Create optimized T5 model
# config = T5Config.from_pretrained("t5-small")
# config.use_cache = False
# model_optimized = T5ForConditionalGeneration(config)
# replace_attention_with_optimized(model_optimized)