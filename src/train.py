import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from sentence_transformers import SentenceTransformer
from typing import List, Tuple, Dict, Any
import warnings

warnings.filterwarnings("ignore", category=UserWarning, module='tslearn')

global_trajectory_buffer = []

def get_hidden_states_hook(module, input, output):
    """Hook to capture the pre-layernorm hidden states."""
    hidden_state = input[0][0, -1, :].detach().cpu().float().numpy()
    global_trajectory_buffer.append(hidden_state)

def find_last_layernorm(model):
    """Finds the name of the final LayerNorm module in a model."""
    possible_names = ['model.norm', 'transformer.ln_f', 'lm_head.norm']
    for name, module in model.named_modules():
        if name in possible_names:
            print(f"Found final LayerNorm at: {name}")
            return module
    for name, module in reversed(list(model.named_modules())):
        if isinstance(module, torch.nn.LayerNorm):
            print(f"Found final LayerNorm with fallback at: {name}")
            return module
    raise ValueError("Could not find the final LayerNorm module in the model.")

def setup_model_and_tokenizer(model_name: str) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
    """Setup model and tokenizer for trajectory extraction."""
    print(f"Loading model: {model_name}")
    model = AutoModelForCausalLM.from_pretrained(
        model_name, 
        torch_dtype=torch.bfloat16, 
        device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = model.config.eos_token_id
    return model, tokenizer

def generate_ensemble_trajectories(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer, 
    prompt: str,
    k: int = 10,
    generation_params: Dict[str, Any] | None = None
) -> Tuple[List[str], List[np.ndarray]]:
    """Generate ensemble of responses and extract latent trajectories."""
    global global_trajectory_buffer
    
    if generation_params is None:
        generation_params = {
            "max_new_tokens": 50,
            "do_sample": True,
            "top_p": 0.9,
            "temperature": 0.7,
            "pad_token_id": tokenizer.eos_token_id
        }
    
    last_layernorm = find_last_layernorm(model)
    hook = last_layernorm.register_forward_hook(get_hidden_states_hook)

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    ensemble_trajectories = []
    ensemble_texts = []

    for _ in range(k):
        global_trajectory_buffer = []
        with torch.no_grad():
            generated_ids = model.generate(**inputs, **generation_params)
        
        if global_trajectory_buffer:
            ensemble_trajectories.append(np.array(global_trajectory_buffer))
        else:
            ensemble_trajectories.append(np.array([]))
        
        generated_text = tokenizer.decode(
            generated_ids[0][inputs['input_ids'].shape[1]:], 
            skip_special_tokens=True
        )
        ensemble_texts.append(generated_text)

    hook.remove()
    return ensemble_texts, ensemble_trajectories

def setup_baseline_models():
    """Setup baseline models for comparison."""
    print("Loading SentenceTransformer for semantic entropy calculation...")
    sbert_model = SentenceTransformer('all-MiniLM-L6-v2')
    return sbert_model

if __name__ == "__main__":
    print("=== Model Training/Setup Module ===")
    
    test_model_name = "gpt2"
    model, tokenizer = setup_model_and_tokenizer(test_model_name)
    sbert_model = setup_baseline_models()
    
    test_prompt = "What is the capital of France?"
    texts, trajectories = generate_ensemble_trajectories(
        model, tokenizer, test_prompt, k=3
    )
    
    print(f"Generated {len(texts)} responses with {len(trajectories)} trajectories")
    print("Training/setup module working correctly!")
