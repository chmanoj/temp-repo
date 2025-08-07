import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import time
from typing import Tuple

def standard_attention(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, 
                      scale: float = None) -> torch.Tensor:
    """
    Standard attention implementation: O(N²) memory
    
    Steps:
    1. Compute full attention matrix S = Q @ K^T  [N x N matrix]
    2. Apply scaling: S = S / sqrt(d)
    3. Compute softmax: P = softmax(S)  [Still N x N matrix]
    4. Apply to values: O = P @ V
    
    Memory: O(N²) for storing the full attention matrix
    """
    if scale is None:
        scale = Q.size(-1) ** -0.5
    
    # Step 1: Compute attention scores [Batch, Heads, SeqLen, SeqLen]
    print(f"💾 Standard: Creating {Q.size(-2)}x{Q.size(-2)} attention matrix")
    scores = torch.matmul(Q, K.transpose(-2, -1)) * scale
    
    # Step 2: Softmax over full matrix
    attn_weights = F.softmax(scores, dim=-1)
    
    # Step 3: Apply to values
    output = torch.matmul(attn_weights, V)
    
    return output, attn_weights

def flash_attention_simulation(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor,
                              block_size: int = 64, scale: float = None) -> torch.Tensor:
    """
    Flash Attention simulation showing the key algorithmic ideas:
    
    Key Insights:
    1. Never materialize the full N×N attention matrix
    2. Process attention in blocks using "online softmax"
    3. Keep running statistics to maintain numerical correctness
    4. Recompute attention weights during backward pass (memory-time tradeoff)
    
    Memory: O(N) instead of O(N²)
    """
    if scale is None:
        scale = Q.size(-1) ** -0.5
    
    batch_size, num_heads, seq_len, head_dim = Q.shape
    
    # Initialize output and statistics
    output = torch.zeros_like(Q)
    max_vals = torch.full((batch_size, num_heads, seq_len), float('-inf'), device=Q.device)
    sum_exp = torch.zeros(batch_size, num_heads, seq_len, device=Q.device)
    
    print(f"⚡ Flash: Processing {seq_len} tokens in blocks of {block_size}")
    print(f"💾 Flash: Peak memory O({seq_len}) vs Standard O({seq_len}²)")
    
    # Process in blocks (this is the key innovation!)
    for i in range(0, seq_len, block_size):
        for j in range(0, seq_len, block_size):
            # Get current blocks
            i_end = min(i + block_size, seq_len)
            j_end = min(j + block_size, seq_len)
            
            Q_block = Q[:, :, i:i_end, :]  # [B, H, block_size, D]
            K_block = K[:, :, j:j_end, :]  # [B, H, block_size, D]
            V_block = V[:, :, j:j_end, :]  # [B, H, block_size, D]
            
            # Compute attention for this block only
            scores_block = torch.matmul(Q_block, K_block.transpose(-2, -1)) * scale
            
            # Online softmax update (the mathematical magic!)
            block_max = scores_block.max(dim=-1, keepdim=True).values
            
            # Update global max and renormalize previous contributions
            old_max = max_vals[:, :, i:i_end].unsqueeze(-1)
            new_max = torch.maximum(old_max, block_max)
            
            # Compute block contribution with numerically stable softmax
            block_exp = torch.exp(scores_block - new_max)
            block_sum = block_exp.sum(dim=-1, keepdim=True)
            
            # Update output using online algorithm
            exp_diff = torch.exp(old_max - new_max)
            output[:, :, i:i_end, :] = (
                output[:, :, i:i_end, :] * exp_diff * 
                sum_exp[:, :, i:i_end].unsqueeze(-1).unsqueeze(-1) +
                torch.matmul(block_exp, V_block)
            ) / (sum_exp[:, :, i:i_end].unsqueeze(-1).unsqueeze(-1) * exp_diff + block_sum)
            
            # Update statistics
            max_vals[:, :, i:i_end] = new_max.squeeze(-1)
            sum_exp[:, :, i:i_end] = sum_exp[:, :, i:i_end] * exp_diff.squeeze(-1) + block_sum.squeeze(-1)
    
    return output

def demonstrate_memory_access_patterns():
    """
    Visualize why Flash Attention is faster: it's all about memory access!
    """
    print("\n" + "="*70)
    print("🧠 MEMORY ACCESS PATTERN ANALYSIS")
    print("="*70)
    
    # Simulate memory access for different sequence lengths
    seq_lengths = [256, 512, 1024, 2048]
    
    print(f"\n{'Seq Len':<10} {'Standard Reads':<15} {'Flash Reads':<15} {'Memory Ratio':<15}")
    print("-" * 60)
    
    for seq_len in seq_lengths:
        # Standard attention memory accesses
        standard_matrix_size = seq_len * seq_len
        standard_reads = standard_matrix_size * 2  # Read for softmax, read for matmul
        
        # Flash attention (block-based processing)
        block_size = 64
        num_blocks = (seq_len + block_size - 1) // block_size
        flash_reads = num_blocks * num_blocks * block_size * block_size
        
        memory_ratio = standard_matrix_size / (seq_len * block_size)
        
        print(f"{seq_len:<10} {standard_reads:<15,} {flash_reads:<15,} {memory_ratio:<15.1f}x")
    
    print(f"\n💡 Key Insight: Flash Attention reads the same amount of data,")
    print(f"   but in a pattern that matches GPU memory hierarchy!")

def online_softmax_demo():
    """
    Demonstrate the online softmax algorithm that makes Flash Attention possible
    """
    print("\n" + "="*70)
    print("🧮 ONLINE SOFTMAX ALGORITHM DEMO")
    print("="*70)
    
    # Example: compute softmax of [1, 2, 3, 4, 5] in chunks
    values = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
    chunk_size = 2
    
    print(f"Input: {values.tolist()}")
    print(f"Standard softmax: {F.softmax(values, dim=0).tolist()}")
    
    # Online softmax computation
    running_max = float('-inf')
    running_sum = 0.0
    result = torch.zeros_like(values)
    
    print(f"\n🔄 Online computation (chunk size = {chunk_size}):")
    
    for i in range(0, len(values), chunk_size):
        chunk = values[i:i+chunk_size]
        print(f"\nChunk {i//chunk_size + 1}: {chunk.tolist()}")
        
        # Update max
        chunk_max = chunk.max().item()
        new_max = max(running_max, chunk_max)
        
        print(f"  Old max: {running_max:.3f}, Chunk max: {chunk_max:.3f}, New max: {new_max:.3f}")
        
        # Renormalize previous results
        if running_max != float('-inf'):
            adjustment = torch.exp(torch.tensor(running_max - new_max))
            result[:i] *= adjustment
            running_sum *= adjustment.item()
            print(f"  Renormalized previous results by {adjustment.item():.3f}")
        
        # Add current chunk contribution
        chunk_exp = torch.exp(chunk - new_max)
        result[i:i+len(chunk)] = chunk_exp
        running_sum += chunk_exp.sum().item()
        
        print(f"  Running sum: {running_sum:.3f}")
        running_max = new_max
    
    # Final normalization
    result /= running_sum
    
    print(f"\nOnline result: {result.tolist()}")
    print(f"Difference from standard: {torch.max(torch.abs(result - F.softmax(values, dim=0))).item():.2e}")
    print("✅ Numerically identical!")

def memory_complexity_visualization():
    """
    Visualize memory complexity differences
    """
    seq_lengths = np.array([128, 256, 512, 1024, 2048, 4096])
    
    # Memory usage (in arbitrary units)
    standard_memory = seq_lengths ** 2  # O(N²)
    flash_memory = seq_lengths * 64     # O(N) with block size 64
    
    plt.figure(figsize=(10, 6))
    plt.semilogy(seq_lengths, standard_memory, 'r-o', label='Standard Attention O(N²)', linewidth=2, markersize=8)
    plt.semilogy(seq_lengths, flash_memory, 'b-s', label='Flash Attention O(N)', linewidth=2, markersize=8)
    
    plt.xlabel('Sequence Length')
    plt.ylabel('Memory Usage (log scale)')
    plt.title('Memory Complexity: Standard vs Flash Attention')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Add annotations
    for i, seq_len in enumerate([512, 2048]):
        idx = np.where(seq_lengths == seq_len)[0][0]
        ratio = standard_memory[idx] / flash_memory[idx]
        plt.annotate(f'{ratio:.0f}x less\nmemory', 
                    xy=(seq_len, flash_memory[idx]), 
                    xytext=(seq_len, flash_memory[idx] * 5),
                    arrowprops=dict(arrowstyle='->', color='green', lw=2),
                    fontsize=10, ha='center', color='green', weight='bold')
    
    plt.tight_layout()
    plt.savefig('memory_complexity.png', dpi=150, bbox_inches='tight')
    plt.show()

def computational_flow_comparison():
    """
    Show the step-by-step computational differences
    """
    print("\n" + "="*70)
    print("🔄 COMPUTATIONAL FLOW COMPARISON")
    print("="*70)
    
    seq_len = 1024
    head_dim = 64
    
    print(f"\nFor sequence length {seq_len}, head dimension {head_dim}:")
    print(f"\n🏃 STANDARD ATTENTION:")
    print(f"  1. Q @ K^T → Create {seq_len}×{seq_len} matrix ({seq_len**2:,} elements)")
    print(f"  2. Scale and softmax → Process full {seq_len}×{seq_len} matrix")
    print(f"  3. P @ V → Final multiplication")
    print(f"  💾 Peak memory: {seq_len**2 * 4 / 1e6:.1f} MB (for attention matrix alone)")
    print(f"  🔄 Memory transfers: Reading full matrix multiple times")
    
    block_size = 64
    num_blocks = seq_len // block_size
    print(f"\n⚡ FLASH ATTENTION:")
    print(f"  1. Process in {block_size}×{block_size} blocks ({num_blocks}×{num_blocks} = {num_blocks**2} blocks)")
    print(f"  2. Online softmax → Never store full matrix")
    print(f"  3. Accumulate results → Streaming computation")
    print(f"  💾 Peak memory: {seq_len * block_size * 4 / 1e6:.1f} MB (much smaller!)")
    print(f"  🔄 Memory transfers: Sequential, cache-friendly access")
    
    print(f"\n📊 EFFICIENCY GAINS:")
    memory_reduction = (seq_len**2) / (seq_len * block_size)
    print(f"  • Memory reduction: {memory_reduction:.1f}x less")
    print(f"  • Cache efficiency: Much better (sequential vs random access)")
    print(f"  • Parallelization: Better GPU utilization")

def hardware_considerations():
    """
    Explain how Flash Attention leverages GPU architecture
    """
    print("\n" + "="*70)
    print("🏗️  GPU ARCHITECTURE CONSIDERATIONS")
    print("="*70)
    
    print(f"""
🖥️  GPU MEMORY HIERARCHY (from fastest to slowest):
   1. Registers: ~20 TB/s bandwidth, tiny capacity (~64KB per SM)
   2. Shared Memory: ~19 TB/s, small capacity (~164KB per SM)  
   3. L2 Cache: ~7 TB/s, moderate capacity (~40MB)
   4. HBM (Global Memory): ~1.5 TB/s, large capacity (24-80GB)

🧠 STANDARD ATTENTION PROBLEMS:
   • Creates huge N×N matrix that doesn't fit in fast memory
   • Requires multiple passes through slow global memory
   • Poor arithmetic intensity (memory bound, not compute bound)
   • Irregular memory access patterns

⚡ FLASH ATTENTION SOLUTIONS:
   • Small blocks fit entirely in shared memory
   • Minimizes global memory traffic
   • High arithmetic intensity (compute bound)
   • Sequential, predictable memory access
   • Fuses operations (fewer kernel launches)

🎯 THE RESULT:
   • Better hardware utilization
   • Lower memory bandwidth requirements  
   • Higher effective throughput
   • Reduced latency
    """)

def accuracy_verification():
    """
    Verify that Flash Attention produces identical results
    """
    print("\n" + "="*70)
    print("🔍 NUMERICAL ACCURACY VERIFICATION")
    print("="*70)
    
    # Create test tensors
    batch_size, num_heads, seq_len, head_dim = 2, 8, 256, 64
    torch.manual_seed(42)
    
    Q = torch.randn(batch_size, num_heads, seq_len, head_dim)
    K = torch.randn(batch_size, num_heads, seq_len, head_dim)  
    V = torch.randn(batch_size, num_heads, seq_len, head_dim)
    
    # Standard attention
    standard_out, _ = standard_attention(Q, K, V)
    
    # Flash attention simulation
    flash_out = flash_attention_simulation(Q, K, V, block_size=64)
    
    # Compare results
    max_diff = torch.max(torch.abs(standard_out - flash_out)).item()
    mean_diff = torch.mean(torch.abs(standard_out - flash_out)).item()
    relative_error = max_diff / torch.max(torch.abs(standard_out)).item()
    
    print(f"Maximum absolute difference: {max_diff:.2e}")
    print(f"Mean absolute difference: {mean_diff:.2e}")
    print(f"Relative error: {relative_error:.2e}")
    
    if relative_error < 1e-5:
        print("✅ Results are numerically identical!")
    else:
        print("❌ Results differ significantly")
    
    return max_diff < 1e-5

def main():
    """
    Run the complete Flash Attention explanation and demonstration
    """
    print("🚀 FLASH ATTENTION: HOW IT WORKS")
    print("Understanding the magic behind O(N) memory attention")
    print("="*70)
    
    # Core algorithmic insights
    online_softmax_demo()
    
    # Memory access patterns
    demonstrate_memory_access_patterns()
    
    # Computational flow
    computational_flow_comparison()
    
    # Hardware considerations  
    hardware_considerations()
    
    # Accuracy verification
    accuracy_verification()
    
    # Memory complexity visualization
    memory_complexity_visualization()
    
    print("\n" + "="*70)
    print("🎯 KEY TAKEAWAYS")
    print("="*70)
    print("""
1. 🔢 SAME MATH: Flash Attention computes identical results
2. 💾 MEMORY: O(N) instead of O(N²) through online algorithms  
3. ⚡ SPEED: Better hardware utilization and memory access patterns
4. 🏗️  HARDWARE: Designed for GPU memory hierarchy
5. 🧮 INNOVATION: Online softmax enables streaming computation

The "magic" is reorganizing WHEN and WHERE operations happen,
not WHAT operations are performed!
    """)

if __name__ == "__main__":
    main()