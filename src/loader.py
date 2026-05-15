import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_model(model_name: str = "meta-llama/Llama-3.2-1B-Instruct",
               device: str = "cuda",
               dtype: torch.dtype = torch.float16):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=dtype, device_map=device,
    )
    model.eval()
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer