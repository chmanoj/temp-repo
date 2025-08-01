import time
import torch
from transformers import T5ForConditionalGeneration
from t5_flash_attn_mac import create_optimized_model

def test_performance():
    # Use MPS if available, otherwise CPU
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Smaller sizes for MacBook testing
    batch_size, seq_len = 2, 256
    input_ids = torch.randint(0, 100, (batch_size, seq_len)).to(device)
    decoder_input_ids = torch.randint(0, 100, (batch_size, seq_len)).to(device)
    
    # Initialize models
    model_original = T5ForConditionalGeneration.from_pretrained("t5-small").to(device)
    model_original.config.use_cache = False
    model_optimized = create_optimized_model()
    model_optimized = model_optimized.to(device)
    
    # === Inference Speed Test ===
    print("\n=== Inference Speed Test ===")
    for model_name, model in [("Original", model_original), ("Optimized", model_optimized)]:
        # Warm-up
        # print(f"Running warm-up for {model_name}...")
        with torch.no_grad():
            _ = model(input_ids=input_ids, decoder_input_ids=decoder_input_ids)
        
        # Synchronize for accurate timing
        if device == "mps":
            torch.mps.synchronize()
        start_time = time.perf_counter()
        
        with torch.no_grad():
            for _ in range(20):  # Fewer iterations for MacBook
                _ = model(input_ids=input_ids, decoder_input_ids=decoder_input_ids)
        
        if device == "mps":
            torch.mps.synchronize()
        avg_time = (time.perf_counter() - start_time) / 20
        print(f"{model_name}: {avg_time:.4f} sec/batch")
    
    # === Memory Test ===
    print("\n=== Memory Usage Test ===")
    for model_name, model in [("Original", model_original), ("Optimized", model_optimized)]:
        # Clear memory
        if device == "mps":
            torch.mps.empty_cache()
        
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        
        optimizer.zero_grad()
        outputs = model(input_ids=input_ids, decoder_input_ids=decoder_input_ids, labels=decoder_input_ids)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        
        # Get memory usage
        if device == "mps":
            memory = torch.mps.current_allocated_memory() / 1024**2  # MB
        else:
            memory = torch.cuda.max_memory_allocated() / 1024**2 if torch.cuda.is_available() else 0
        
        print(f"{model_name}: {memory:.2f} MB")

if __name__ == "__main__":
    test_performance()