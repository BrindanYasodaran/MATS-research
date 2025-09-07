#!/usr/bin/env python3
"""
Quick test of emoji steering vectors
"""
import torch
from transformer_lens import HookedTransformer

def load_steering_vectors(vectors_path: str):
    """Load steering vectors from file."""
    return torch.load(vectors_path, map_location='cpu')

def create_steering_hook(steering_vector: torch.Tensor, alpha: float):
    """Create a hook function that adds steering vector to residual stream."""
    def steering_hook(activations, hook):
        vector = steering_vector.to(device=activations.device, dtype=activations.dtype)
        return activations + alpha * vector
    return steering_hook

def test_steering():
    print("Loading model...")
    # Adjust model path as needed
    model = HookedTransformer.from_pretrained_no_processing(
        "meta-llama/Meta-Llama-3-70B-Instruct",
        device="cuda",
        dtype=torch.bfloat16,
    )
    
    print("Loading steering vectors...")
    # Use pre-computed vectors first
    steering_vectors = load_steering_vectors("steer-llama/outputs/EMOJI_VECTOR.pt")
    
    # Test prompt
    prompt = "Hello! How are you today?"
    messages = [{"role": "user", "content": prompt}]
    input_text = model.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    
    print(f"Testing prompt: {prompt}")
    print("\n" + "="*50)
    
    # Generate without steering
    print("WITHOUT STEERING:")
    inputs = model.tokenizer(input_text, return_tensors="pt").to(model.cfg.device)
    with torch.no_grad():
        output = model.generate(inputs.input_ids, max_new_tokens=50, do_sample=True, temperature=0.7)
    response = model.tokenizer.decode(output[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
    print(response)
    
    # Generate with steering
    print("\nWITH EMOJI STEERING (Layer 25, Alpha 2.0):")
    layer_name = "blocks.25.hook_resid_post"  # Middle layer, usually works well
    alpha = 2.0
    
    if layer_name in steering_vectors:
        steering_vector = steering_vectors[layer_name]
        hook_fn = create_steering_hook(steering_vector, alpha)
        
        with model.hooks(fwd_hooks=[(layer_name, hook_fn)]):
            with torch.no_grad():
                output = model.generate(inputs.input_ids, max_new_tokens=50, do_sample=True, temperature=0.7)
        response = model.tokenizer.decode(output[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
        print(response)
    else:
        print(f"Layer {layer_name} not found in steering vectors")
        print(f"Available layers: {list(steering_vectors.keys())[:5]}...")

if __name__ == "__main__":
    test_steering()
