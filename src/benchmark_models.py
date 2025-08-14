"""
RML Model Comparison
Benchmarks embedding models on key metrics
"""

import time
import torch
import numpy as np
from typing import List, Dict
from transformers import AutoModel, AutoTokenizer
import psutil
import faiss

class ModelBenchmark:
    """Benchmark embedding models"""
    
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
        self.test_data = [
            "Albert Einstein developed the theory of relativity.",
            "Marie Curie conducted pioneering research on radioactivity.",
            "The Eiffel Tower is located in Paris, France.",
            "Tesla, Inc. is an American electric vehicle company.",
            "The Great Wall of China is visible from space."
        ]
    
    def run_benchmark(self, model_name: str) -> dict:
        """Run benchmark for a single model"""
        # Load model
        start_mem = psutil.Process().memory_info().rss / 1024 / 1024  # MB
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModel.from_pretrained(
            model_name,
            device_map="auto" if self.device == 'cuda' else None,
            torch_dtype=torch.bfloat16 if self.device != 'cpu' else torch.float32
        ).to(self.device)
        model.eval()
        
        # Encode and measure
        start_time = time.time()
        with torch.no_grad():
            inputs = tokenizer(
                self.test_data, 
                padding=True, 
                truncation=True, 
                return_tensors="pt"
            ).to(self.device)
            outputs = model(**inputs)
            embeddings = outputs.last_hidden_state.mean(dim=1).cpu().numpy()
        
        # Calculate metrics
        encode_time = time.time() - start_time
        tokens = sum(len(ids) for ids in inputs['input_ids'])
        tokens_per_sec = tokens / encode_time
        memory_used = psutil.Process().memory_info().rss / 1024 / 1024 - start_mem
        
        # Clean up
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        return {
            'model': model_name.split('/')[-1],
            'speed_tokens_sec': round(tokens_per_sec, 2),
            'memory_mb': round(memory_used, 2),
            'embedding_dim': embeddings.shape[1]
        }

def main():
    benchmark = ModelBenchmark()
    
    models = [
        "intfloat/e5-mistral-7b-instruct",  # Best quality
        "BAAI/bge-large-en-v1.5",           # Best balance
        "sentence-transformers/all-MiniLM-L6-v2"  # Fastest
    ]
    
    print("\nBenchmarking models...")
    results = []
    for model in models:
        print(f"\nTesting {model}")
        results.append(benchmark.run_benchmark(model))
    
    # Print results
    print("\nResults:")
    print("-" * 80)
    print(f"{'Model':<30} | {'Speed (tok/s)':>12} | {'Memory (MB)':>10} | {'Dim':>6}")
    print("-" * 80)
    for r in results:
        print(f"{r['model']:<30} | {r['speed_tokens_sec']:>12.2f} | {r['memory_mb']:>10.2f} | {r['embedding_dim']:>6}")

if __name__ == "__main__":
    main()
