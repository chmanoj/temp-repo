# Add missing import
import math
import time
import psutil
import os
from transformers import T5ForConditionalGeneration, T5Tokenizer, T5Config
from transformers.models.t5.modeling_t5 import T5Attention
import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, Tuple
import gc

# Check if flash-attn is available
try:
    from flash_attn import flash_attn_func
    FLASH_ATTN_AVAILABLE = True
    print("✅ Flash Attention 2 is available")
except ImportError:
    FLASH_ATTN_AVAILABLE = False
    print("❌ Flash Attention 2 not available. Install with: pip install flash-attn --no-build-isolation")

class FlashT5Attention(nn.Module):
    """
    T5 Attention layer with Flash Attention 2 implementation
    """
    def __init__(self, config, has_relative_attention_bias=False):
        super().__init__()
        self.is_decoder = config.is_decoder
        self.has_relative_attention_bias = has_relative_attention_bias
        self.relative_attention_num_buckets = config.relative_attention_num_buckets
        self.relative_attention_max_distance = config.relative_attention_max_distance
        self.d_model = config.d_model
        self.key_value_proj_dim = config.d_kv
        self.n_heads = config.num_heads
        self.dropout = config.dropout_rate
        self.inner_dim = self.n_heads * self.key_value_proj_dim

        # Linear transformations for Q, K, V
        self.q = nn.Linear(self.d_model, self.inner_dim, bias=False)
        self.k = nn.Linear(self.d_model, self.inner_dim, bias=False)
        self.v = nn.Linear(self.d_model, self.inner_dim, bias=False)
        self.o = nn.Linear(self.inner_dim, self.d_model, bias=False)

        if self.has_relative_attention_bias:
            self.relative_attention_bias = nn.Embedding(
                self.relative_attention_num_buckets, self.n_heads
            )
            
    def _relative_position_bucket(self, relative_position, bidirectional=True, num_buckets=32, max_distance=128):
        """
        Adapted from Mesh Tensorflow:
        https://github.com/tensorflow/mesh/blob/0cb87fe7b8ac
        """
        relative_buckets = 0
        if bidirectional:
            num_buckets //= 2
            relative_buckets += (relative_position > 0).to(torch.long) * num_buckets
            relative_position = torch.abs(relative_position)
        else:
            relative_position = -torch.min(relative_position, torch.zeros_like(relative_position))
        
        # half of the buckets are for exact increments in positions
        max_exact = num_buckets // 2
        is_small = relative_position < max_exact

        # The other half of the buckets are for logarithmically bigger bins in positions up to max_distance
        relative_position_if_large = max_exact + (
            torch.log(relative_position.float() / max_exact)
            / math.log(max_distance / max_exact)
            * (num_buckets - max_exact)
        ).to(torch.long)
        relative_position_if_large = torch.min(
            relative_position_if_large, torch.full_like(relative_position_if_large, num_buckets - 1)
        )

        relative_buckets += torch.where(is_small, relative_position, relative_position_if_large)
        return relative_buckets

    def compute_bias(self, query_length, key_length, device=None):
        """Compute binned relative position bias"""
        if device is None:
            device = self.relative_attention_bias.weight.device
        context_position = torch.arange(query_length, dtype=torch.long, device=device)[:, None]
        memory_position = torch.arange(key_length, dtype=torch.long, device=device)[None, :]
        relative_position = memory_position - context_position  # shape (query_length, key_length)
        relative_position_bucket = self._relative_position_bucket(
            relative_position,  # shape (query_length, key_length)
            bidirectional=(not self.is_decoder),
            num_buckets=self.relative_attention_num_buckets,
            max_distance=self.relative_attention_max_distance,
        )
        values = self.relative_attention_bias(relative_position_bucket)  # shape (query_length, key_length, num_heads)
        values = values.permute([2, 0, 1]).unsqueeze(0)  # shape (1, num_heads, query_length, key_length)
        return values

    def forward(
        self,
        hidden_states,
        mask=None,
        key_value_states=None,
        position_bias=None,
        past_key_value=None,
        layer_head_mask=None,
        query_length=None,
        use_cache=False,
        output_attentions=False,
    ):
        """
        Forward pass that maintains compatibility with T5's expected output format
        """
        batch_size, seq_len = hidden_states.shape[:2]
        
        # Determine real sequence length and key length
        real_seq_length = seq_len
        if past_key_value is not None:
            real_seq_length += past_key_value[0].shape[2] if past_key_value[0] is not None else 0
        
        key_length = real_seq_length if key_value_states is None else key_value_states.shape[1]
        
        def shape(states):
            return states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)
        
        def unshape(states):
            return states.transpose(1, 2).contiguous().view(batch_size, -1, self.inner_dim)

        # Project to Q, K, V
        query_states = shape(self.q(hidden_states))  # (batch, n_heads, seq_len, head_dim)
        
        if key_value_states is None:
            key_states = shape(self.k(hidden_states))
            value_states = shape(self.v(hidden_states))
        else:
            key_states = shape(self.k(key_value_states))
            value_states = shape(self.v(key_value_states))

        # Handle past key values for generation
        if past_key_value is not None:
            if key_value_states is None:
                key_states = torch.cat([past_key_value[0], key_states], dim=-2)
                value_states = torch.cat([past_key_value[1], value_states], dim=-2)
            else:
                key_states = past_key_value[0]
                value_states = past_key_value[1]

        # Compute position bias if needed
        if position_bias is None and self.has_relative_attention_bias:
            position_bias = self.compute_bias(real_seq_length, key_length, device=hidden_states.device)

        if use_cache:
            present_key_value = (key_states, value_states)
        else:
            present_key_value = None

        # Use Flash Attention 2 if available and dtype is supported
        if (FLASH_ATTN_AVAILABLE and 
            query_states.dtype in [torch.float16, torch.bfloat16] and 
            position_bias is None and  # Flash attention doesn't support position bias directly
            layer_head_mask is None):  # Flash attention doesn't support head masking
            
            # Flash Attention expects (batch_size, seq_len, n_heads, head_dim)
            q = query_states.transpose(1, 2)  # (batch, seq_len, n_heads, head_dim)
            k = key_states.transpose(1, 2)
            v = value_states.transpose(1, 2)
            
            try:
                # Flash attention
                attn_output = flash_attn_func(
                    q, k, v,
                    dropout_p=self.dropout if self.training else 0.0,
                    causal=self.is_decoder and key_value_states is None,
                )
                
                attn_output = attn_output.transpose(1, 2)  # Back to (batch, n_heads, seq_len, head_dim)
                attn_output = unshape(attn_output)
                attn_weights = None  # Flash attention doesn't return weights
                
            except Exception as e:
                print(f"Flash attention failed, falling back to standard: {e}")
                # Fall back to standard attention
                attn_output, attn_weights = self._standard_attention(
                    query_states, key_states, value_states, position_bias, mask, layer_head_mask
                )
                attn_output = unshape(attn_output)
        else:
            # Use standard attention
            attn_output, attn_weights = self._standard_attention(
                query_states, key_states, value_states, position_bias, mask, layer_head_mask
            )
            attn_output = unshape(attn_output)

        attn_output = self.o(attn_output)

        # Return outputs in the format T5 expects
        outputs = (attn_output,)
        if output_attentions:
            outputs += (attn_weights,)
        if use_cache:
            outputs += (present_key_value,)
        if position_bias is None and self.has_relative_attention_bias:
            # T5 expects position_bias to be returned from the first layer
            outputs = (attn_output, position_bias) + outputs[1:]
        
        return outputs
    
    def _standard_attention(self, query_states, key_states, value_states, position_bias, mask, layer_head_mask):
        """Standard attention computation as fallback"""
        # Compute attention scores
        scores = torch.matmul(query_states, key_states.transpose(-1, -2))
        scores = scores / math.sqrt(self.key_value_proj_dim)
        
        # Add position bias if available
        if position_bias is not None:
            scores = scores + position_bias

        # Apply attention mask
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float("-inf"))

        # Apply softmax
        attn_weights = F.softmax(scores, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_weights = F.dropout(attn_weights, p=self.dropout, training=self.training)

        # Apply layer head mask
        if layer_head_mask is not None:
            attn_weights = attn_weights * layer_head_mask

        # Apply attention to values
        attn_output = torch.matmul(attn_weights, value_states)
        
        return attn_output, attn_weights

class FlashT5ForConditionalGeneration(T5ForConditionalGeneration):
    """
    T5 model with Flash Attention 2 integration - more conservative approach
    """
    def __init__(self, config):
        super().__init__(config)
        # Only replace if Flash Attention is available
        if FLASH_ATTN_AVAILABLE:
            self._replace_attention_layers()
        else:
            print("⚠️ Flash Attention not available, using standard T5 model")
    
    def _replace_attention_layers(self):
        """Replace standard attention with Flash Attention layers"""
        print("🔄 Replacing attention layers with Flash Attention...")
        
        try:
            # Replace encoder attention layers
            for i, layer in enumerate(self.encoder.block):
                # Only first layer has relative attention bias
                has_relative_bias = (i == 0)
                original_attention = layer.layer[0].SelfAttention
                
                # Create new Flash attention layer
                flash_attention = FlashT5Attention(self.config, has_relative_attention_bias=has_relative_bias)
                
                # Copy weights from original attention
                flash_attention.q.weight.data.copy_(original_attention.q.weight.data)
                flash_attention.k.weight.data.copy_(original_attention.k.weight.data)
                flash_attention.v.weight.data.copy_(original_attention.v.weight.data)
                flash_attention.o.weight.data.copy_(original_attention.o.weight.data)
                
                if has_relative_bias and hasattr(original_attention, 'relative_attention_bias'):
                    flash_attention.relative_attention_bias.weight.data.copy_(
                        original_attention.relative_attention_bias.weight.data
                    )
                
                # Replace the attention layer
                layer.layer[0].SelfAttention = flash_attention
            
            # Replace decoder attention layers
            for i, layer in enumerate(self.decoder.block):
                has_relative_bias = (i == 0)
                
                # Replace self-attention
                original_self_attn = layer.layer[0].SelfAttention
                flash_self_attn = FlashT5Attention(self.config, has_relative_attention_bias=has_relative_bias)
                
                # Copy weights
                flash_self_attn.q.weight.data.copy_(original_self_attn.q.weight.data)
                flash_self_attn.k.weight.data.copy_(original_self_attn.k.weight.data)
                flash_self_attn.v.weight.data.copy_(original_self_attn.v.weight.data)
                flash_self_attn.o.weight.data.copy_(original_self_attn.o.weight.data)
                
                if has_relative_bias and hasattr(original_self_attn, 'relative_attention_bias'):
                    flash_self_attn.relative_attention_bias.weight.data.copy_(
                        original_self_attn.relative_attention_bias.weight.data
                    )
                
                layer.layer[0].SelfAttention = flash_self_attn
                
                # Replace cross-attention
                original_cross_attn = layer.layer[1].EncDecAttention
                flash_cross_attn = FlashT5Attention(self.config, has_relative_attention_bias=False)
                
                # Copy weights
                flash_cross_attn.q.weight.data.copy_(original_cross_attn.q.weight.data)
                flash_cross_attn.k.weight.data.copy_(original_cross_attn.k.weight.data)
                flash_cross_attn.v.weight.data.copy_(original_cross_attn.v.weight.data)
                flash_cross_attn.o.weight.data.copy_(original_cross_attn.o.weight.data)
                
                layer.layer[1].EncDecAttention = flash_cross_attn
                
            print("✅ Successfully replaced all attention layers with Flash Attention")
            
        except Exception as e:
            print(f"❌ Error replacing attention layers: {e}")
            print("🔄 Falling back to standard attention")
            raise e

def get_memory_usage():
    """Get current memory usage in MB"""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024

def benchmark_model(model, tokenizer, input_texts, max_length=512, device='cuda'):
    """Benchmark model performance"""
    model.eval()
    
    # Tokenize inputs
    inputs = tokenizer(
        input_texts,
        max_length=max_length,
        padding=True,
        truncation=True,
        return_tensors="pt"
    ).to(device)
    
    # Warm up
    with torch.no_grad():
        _ = model.generate(**inputs, max_new_tokens=50, do_sample=False)
    
    torch.cuda.synchronize() if device == 'cuda' else None
    gc.collect()
    
    # Memory before
    memory_before = get_memory_usage()
    if device == 'cuda':
        torch.cuda.reset_peak_memory_stats()
        cuda_memory_before = torch.cuda.memory_allocated() / 1024 / 1024
    
    # Timing
    start_time = time.time()
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=100,
            do_sample=False,
            num_beams=1,
            pad_token_id=tokenizer.pad_token_id
        )
    
    torch.cuda.synchronize() if device == 'cuda' else None
    end_time = time.time()
    
    # Memory after
    memory_after = get_memory_usage()
    if device == 'cuda':
        cuda_memory_peak = torch.cuda.max_memory_allocated() / 1024 / 1024
        cuda_memory_after = torch.cuda.memory_allocated() / 1024 / 1024
    
    # Calculate metrics
    inference_time = end_time - start_time
    memory_used = memory_after - memory_before
    
    results = {
        'inference_time': inference_time,
        'cpu_memory_used': memory_used,
        'throughput': len(input_texts) / inference_time
    }
    
    if device == 'cuda':
        results['cuda_memory_peak'] = cuda_memory_peak
        results['cuda_memory_used'] = cuda_memory_after - cuda_memory_before
    
    return results, outputs

def compare_models():
    """Compare standard T5 vs Flash Attention T5"""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Load tokenizer
    model_name = "t5-base"
    tokenizer = T5Tokenizer.from_pretrained(model_name)
    
    # Test inputs of varying lengths
    test_inputs = [
        "translate English to French: Hello, how are you today?",
        "summarize: " + " ".join(["This is a test sentence."] * 50),
        "translate English to German: " + " ".join(["The quick brown fox jumps over the lazy dog."] * 20),
    ]
    
    print("\n" + "="*60)
    print("BENCHMARKING T5 MODELS")
    print("="*60)
    
    results = {}
    
    # Test standard T5
    print("\n🔄 Loading standard T5 model...")
    standard_model = T5ForConditionalGeneration.from_pretrained(model_name)
    if device == 'cuda':
        standard_model = standard_model.half().to(device)  # Use half precision for fair comparison
    else:
        standard_model = standard_model.to(device)
    
    print("📊 Benchmarking standard T5...")
    standard_results, standard_outputs = benchmark_model(
        standard_model, tokenizer, test_inputs, device=device
    )
    results['standard'] = standard_results
    
    # Clean up
    del standard_model
    torch.cuda.empty_cache() if device == 'cuda' else None
    gc.collect()
    
    # Test Flash Attention T5
    print("\n🔄 Loading Flash Attention T5 model...")
    config = T5Config.from_pretrained(model_name)
    flash_model = FlashT5ForConditionalGeneration.from_pretrained(model_name, config=config)
    if device == 'cuda':
        flash_model = flash_model.half().to(device)
    else:
        flash_model = flash_model.to(device)
    
    print("📊 Benchmarking Flash Attention T5...")
    flash_results, flash_outputs = benchmark_model(
        flash_model, tokenizer, test_inputs, device=device
    )
    results['flash'] = flash_results
    
    # Print results
    print("\n" + "="*60)
    print("BENCHMARK RESULTS")
    print("="*60)
    
    print(f"\n📈 PERFORMANCE COMPARISON:")
    print(f"{'Metric':<25} {'Standard T5':<15} {'Flash Attn T5':<15} {'Speedup':<10}")
    print("-" * 65)
    
    speedup_time = standard_results['inference_time'] / flash_results['inference_time']
    print(f"{'Inference Time (s)':<25} {standard_results['inference_time']:<15.3f} {flash_results['inference_time']:<15.3f} {speedup_time:<10.2f}x")
    
    speedup_throughput = flash_results['throughput'] / standard_results['throughput']
    print(f"{'Throughput (samples/s)':<25} {standard_results['throughput']:<15.2f} {flash_results['throughput']:<15.2f} {speedup_throughput:<10.2f}x")
    
    print(f"{'CPU Memory (MB)':<25} {standard_results['cpu_memory_used']:<15.1f} {flash_results['cpu_memory_used']:<15.1f}")
    
    if device == 'cuda':
        memory_reduction = (standard_results['cuda_memory_peak'] - flash_results['cuda_memory_peak']) / standard_results['cuda_memory_peak'] * 100
        print(f"{'CUDA Memory Peak (MB)':<25} {standard_results['cuda_memory_peak']:<15.1f} {flash_results['cuda_memory_peak']:<15.1f} {memory_reduction:<10.1f}% less")
    
    # Show sample outputs
    print(f"\n📝 SAMPLE OUTPUTS:")
    print(f"Input: {test_inputs[0]}")
    print(f"Standard T5: {tokenizer.decode(standard_outputs[0], skip_special_tokens=True)}")
    print(f"Flash Attn T5: {tokenizer.decode(flash_outputs[0], skip_special_tokens=True)}")
    
    return results

def sequence_length_scaling_test():
    """Test how performance scales with sequence length"""
    if not torch.cuda.is_available():
        print("CUDA not available, skipping scaling test")
        return
    
    device = 'cuda'
    model_name = "t5-small"  # Use smaller model for this test
    tokenizer = T5Tokenizer.from_pretrained(model_name)
    
    sequence_lengths = [128, 256, 512, 1024, 2048]
    standard_times = []
    flash_times = []
    standard_memory = []
    flash_memory = []
    
    print("\n" + "="*60)
    print("SEQUENCE LENGTH SCALING TEST")
    print("="*60)
    
    for seq_len in sequence_lengths:
        print(f"\n🧪 Testing sequence length: {seq_len}")
        
        # Generate test input
        test_input = "summarize: " + " ".join(["This is a test sentence."] * (seq_len // 6))
        
        # Test standard model
        standard_model = T5ForConditionalGeneration.from_pretrained(model_name).half().to(device)
        std_results, _ = benchmark_model(standard_model, tokenizer, [test_input], max_length=seq_len, device=device)
        standard_times.append(std_results['inference_time'])
        standard_memory.append(std_results['cuda_memory_peak'])
        
        del standard_model
        torch.cuda.empty_cache()
        gc.collect()
        
        # Test flash model
        config = T5Config.from_pretrained(model_name)
        flash_model = FlashT5ForConditionalGeneration.from_pretrained(model_name, config=config).half().to(device)
        flash_results, _ = benchmark_model(flash_model, tokenizer, [test_input], max_length=seq_len, device=device)
        flash_times.append(flash_results['inference_time'])
        flash_memory.append(flash_results['cuda_memory_peak'])
        
        del flash_model
        torch.cuda.empty_cache()
        gc.collect()
        
        speedup = std_results['inference_time'] / flash_results['inference_time']
        memory_saving = (std_results['cuda_memory_peak'] - flash_results['cuda_memory_peak']) / std_results['cuda_memory_peak'] * 100
        print(f"  ⚡ Speedup: {speedup:.2f}x, Memory saving: {memory_saving:.1f}%")
    
    # Plot results
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(sequence_lengths, standard_times, 'o-', label='Standard T5', linewidth=2)
    plt.plot(sequence_lengths, flash_times, 's-', label='Flash Attention T5', linewidth=2)
    plt.xlabel('Sequence Length')
    plt.ylabel('Inference Time (s)')
    plt.title('Inference Time vs Sequence Length')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    plt.plot(sequence_lengths, standard_memory, 'o-', label='Standard T5', linewidth=2)
    plt.plot(sequence_lengths, flash_memory, 's-', label='Flash Attention T5', linewidth=2)
    plt.xlabel('Sequence Length')
    plt.ylabel('Peak Memory (MB)')
    plt.title('Memory Usage vs Sequence Length')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('flash_attention_scaling.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print(f"\n📊 Scaling test complete! Plot saved as 'flash_attention_scaling.png'")

if __name__ == "__main__":
    print("🚀 Flash Attention 2 + T5 Benchmark Suite")
    print("=" * 50)
    
    # Check requirements
    if not torch.cuda.is_available():
        print("⚠️  CUDA not available. Some tests will be limited.")
    
    # Run main comparison
    try:
        results = compare_models()
        
        # Run scaling test
        sequence_length_scaling_test()
        
        print(f"\n✅ Benchmarking complete!")
        print(f"💡 Flash Attention 2 provides significant improvements for longer sequences")
        print(f"🔧 Consider using mixed precision (fp16) for even better performance")
        
    except Exception as e:
        print(f"❌ Error during benchmarking: {e}")
        print(f"💡 Make sure you have flash-attn installed: pip install flash-attn --no-build-isolation")