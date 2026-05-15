import torch
import numpy as np
from typing import Optional, Dict, Any
import torch.nn.functional as F


def split_activations_by_head(activations, n_heads, resid_dim=None):
    last = activations.shape[-1]
    assert last % n_heads == 0, (
        f"last dim {last} not divisible by n_heads {n_heads} — "
        f"are you sure this is a head-concat tensor and not a residual?"
    )
    head_dim = last // n_heads
    new_shape = activations.shape[:-1] + (n_heads, head_dim)
    return activations.reshape(*new_shape)

def load_top_heads(head_id_path, model_n_heads, topk_heads):
    top_heads_data = torch.load(head_id_path, map_location='cpu')
    top_heads = np.array(top_heads_data['task_ids'])[:topk_heads]
    top_heads = [(head_id // model_n_heads, head_id % model_n_heads) for head_id in top_heads]
    return top_heads


def setup_hooks(model, top_heads, last_token=False):
    activations = {}
    hooks = []
    for layer_idx in set(int(head[0]) for head in top_heads):
        name = f'layer_{layer_idx}'
        layer = model.model.layers[layer_idx].self_attn.o_proj
        def hook(module, inp, out, _name=name, _last=last_token):
            x = inp[0]
            activations[_name] = (x[:, -1:, :] if _last else x).detach()
        hooks.append(layer.register_forward_hook(hook))
    return activations, hooks


# Create data loaders
def collate_fn(batch):
    """Custom collate function to ensure proper tensor batching"""
    input_ids_list = []
    for item in batch:
        input_ids_list.append(torch.tensor(item['input_ids'], dtype=torch.long))
    input_ids = torch.stack(input_ids_list)
    return input_ids


def collate_fn_dict(batch):
    """Custom collate function to ensure proper tensor batching"""
    input_ids_list = []
    attn_mask_list = []
    for item in batch:
        input_ids_list.append(torch.tensor(item['input_ids'], dtype=torch.long))
        attn_mask_list.append(torch.tensor(item['attention_mask'], dtype=torch.long))
    input_ids = torch.stack(input_ids_list)
    attention_masks = torch.stack(attn_mask_list)
    return {'input_ids': input_ids, 'attention_mask': attention_masks}


def extract_activations_attn(model, inputs, activations, top_heads, n_heads, resid_dim, model_head_dim, last_token, device):
    """Extract attention head activations from model forward pass"""
    activations.clear()
    with torch.inference_mode():
        model(**inputs)

    input_ids = inputs['input_ids']
    batch_size = input_ids.size(0)
    seq_len = 1 if last_token else input_ids.size(1)
    function_vectors = torch.zeros((batch_size, seq_len, len(top_heads), model_head_dim),
                                   device=device, dtype=model.dtype)

    for head_id, head_entry in enumerate(top_heads):
        if len(head_entry) == 3:
            L, H, _ = head_entry
        elif len(head_entry) == 2:
            L, H = head_entry
        else:
            raise ValueError(f"Unexpected head entry format: {head_entry}")

        act = activations[f'layer_{L}']
        head_acts = split_activations_by_head(act, n_heads, resid_dim)
        function_vectors[..., head_id, :] = head_acts[..., H, :]

    activations.clear()
    return function_vectors.to(torch.float32)


def setup_hooks_mlp(model, layer_ids):
    activations = {}
    hooks = []
    for layer_idx in layer_ids:
        name = f'layer_{layer_idx}'
        layer = model.model.layers[layer_idx].mlp
        def hook(_module, _inp, out, _name=name):
            activations[_name] = out.detach()
        hooks.append(layer.register_forward_hook(hook))
    return activations, hooks


def extract_activations_mlp(model, inputs, activations, layer_ids, device):
    """Extract MLP activations from model forward pass"""
    activations.clear()
    with torch.inference_mode():
        model(**inputs)

    input_ids = inputs['input_ids']
    batch_size = input_ids.size(0)
    hidden_size = model.config.hidden_size
    seq_len = input_ids.size(1)
    result = torch.zeros((batch_size, seq_len, len(layer_ids), hidden_size), device=device, dtype=model.dtype)

    for i, layer_idx in enumerate(layer_ids):
        result[..., i, :] = activations[f'layer_{layer_idx}']

    activations.clear()
    return result.to(torch.float32)


def extract_activations_sae(model, inputs, activations, top_heads, n_heads, resid_dim, model_head_dim, last_token, device):
    """Extract activations from model forward pass"""
    activations.clear()
    with torch.inference_mode():
        model(**inputs)

    input_ids = inputs['input_ids']
    batch_size = input_ids.size(0)
    seq_len = 1 if last_token else input_ids.size(1)
    function_vectors = torch.zeros((batch_size, seq_len, resid_dim),
                                   device=device, dtype=model.dtype)

    for head_id, head_entry in enumerate(top_heads):
        if len(head_entry) == 3:
            L, H, _ = head_entry
        elif len(head_entry) == 2:
            L, H = head_entry
        else:
            raise ValueError(f"Unexpected head entry format: {head_entry}")

        act = activations[f'layer_{L}']
        head_acts = split_activations_by_head(act, n_heads, resid_dim)

        x = torch.zeros(batch_size, seq_len, resid_dim, device=device, dtype=model.dtype)
        x[..., H * model_head_dim: (H + 1) * model_head_dim] = head_acts[..., H, :]
        function_vectors += model.model.layers[L].self_attn.o_proj(x)

    return function_vectors.to(torch.float32)


def extract_activations_debug(model, inputs, activations, top_heads, n_heads, resid_dim, model_head_dim, last_token, device):
    """Extract activations from model forward pass"""
    activations.clear()
    with torch.inference_mode():
        output = model(**inputs)  # full forward pass needed for output

    input_ids = inputs['input_ids']
    batch_size = input_ids.size(0)
    seq_len = 1 if last_token else input_ids.size(1)
    function_vectors = torch.zeros((batch_size, seq_len, len(top_heads), model_head_dim),
                                   device=device, dtype=model.dtype)

    for head_id, head_entry in enumerate(top_heads):
        if len(head_entry) == 3:
            L, H, _ = head_entry
        elif len(head_entry) == 2:
            L, H = head_entry
        else:
            raise ValueError(f"Unexpected head entry format: {head_entry}")

        act = activations[f'layer_{L}']
        head_acts = split_activations_by_head(act, n_heads, resid_dim)
        function_vectors[..., head_id, :] = head_acts[..., H, :]

    activations.clear()
    return output, function_vectors.to(torch.float32)



