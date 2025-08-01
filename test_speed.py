import time
import torch
from transformers import T5ForConditionalGeneration

def test_performance():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    batch_size, seq_len = 4, 512
    input_ids = torch.randint(0, 100, (batch_size, seq_len)).to(device)
    decoder_input_ids = torch.randint(0, 100, (batch_size, seq_len)).to(device)
    
    # Initialize models
    model_original = T5ForConditionalGeneration.from_pretrained("t5-small").to(device)
    model_original.config.use_cache = False
    model_flash = model_flash.to(device)
    
    # === Inference Speed Test ===
    print("=== Inference Speed Test ===")
    for model_name, model in [("Original", model_original), ("Flash Attention", model_flash)]:
        torch.cuda.synchronize()
        start_time = time.perf_counter()
        with torch.no_grad():
            for _ in range(100):
                _ = model(input_ids=input_ids, decoder_input_ids=decoder_input_ids)
        torch.cuda.synchronize()
        avg_time = (time.perf_counter() - start_time) / 100
        print(f"{model_name}: {avg_time:.4f} sec/batch")
    
    # === Training Memory Test ===
    print("\n=== Training Memory Test ===")
    for model_name, model in [("Original", model_original), ("Flash Attention", model_flash)]:
        torch.cuda.reset_peak_memory_stats()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        
        optimizer.zero_grad()
        outputs = model(input_ids=input_ids, decoder_input_ids=decoder_input_ids, labels=decoder_input_ids)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        
        memory = torch.cuda.max_memory_allocated() / 1024**2  # MB
        print(f"{model_name}: {memory:.2f} MB")

if __name__ == "__main__":
    test_performance()