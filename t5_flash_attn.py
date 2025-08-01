import torch
import torch.nn as nn
from transformers import T5Config, T5ForConditionalGeneration
from transformers.models.t5.modeling_t5 import T5Attention
from flash_attn import flash_attn_func

class T5FlashAttention(T5Attention):
    def __init__(self, config, has_relative_attention_bias=False):
        super().__init__(config, has_relative_attention_bias=False)
        self.is_decoder = getattr(config, "is_decoder", False)

    def forward(self, hidden_states, key_value_states=None, **kwargs):
        # Compute Q, K, V
        q = self.q(hidden_states)
        k = self.k(key_value_states if key_value_states is not None else hidden_states)
        v = self.v(key_value_states if key_value_states is not None else hidden_states)
        
        # Reshape to [batch_size, seq_len, num_heads, head_dim]
        batch_size, tgt_len, _ = hidden_states.size()
        q = q.view(batch_size, -1, self.n_heads, self.d_kv).transpose(1, 2)
        k = k.view(batch_size, -1, self.n_heads, self.d_kv).transpose(1, 2)
        v = v.view(batch_size, -1, self.n_heads, self.d_kv).transpose(1, 2)
        
        # Apply Flash Attention
        causal = (key_value_states is None) and self.is_decoder
        attn_output = flash_attn_func(q, k, v, dropout_p=0.0, causal=causal)
        
        # Reshape back to original dimensions
        attn_output = attn_output.transpose(1, 2).reshape(batch_size, -1, self.d_model)
        return self.o(attn_output)

def replace_attention_with_flash(model):
    for name, module in model.named_modules():
        if isinstance(module, T5Attention):
            parent_name, child_name = name.rsplit('.', 1)
            parent = model.get_submodule(parent_name)
            flash_attn = T5FlashAttention(module.config)
            flash_attn.load_state_dict(module.state_dict())
            setattr(parent, child_name, flash_attn)

# Create T5 model with Flash Attention
config = T5Config.from_pretrained("t5-small")
config.use_cache = False  # Disable KV caching for fair comparison
model_flash = T5ForConditionalGeneration(config)
replace_attention_with_flash(model_flash)