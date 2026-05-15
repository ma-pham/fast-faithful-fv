from tqdm import tqdm
import torch

from utils_v2 import *
from prompt_utils import *


def collect_attn(model, tokenizer, task, n_shots, n_trials, prefixes, seperators, dataset_path, train_split, device, save_path, n_tokens=1, seed=42):
    
    # Model configuration
    n_heads = model.config.num_attention_heads
    n_layers = model.config.num_hidden_layers
    resid_dim = model.config.hidden_size
    model_head_dim = getattr(
    model.config, "head_dim",
    resid_dim // n_heads)

    if save_path and os.path.exists(save_path):
        raw = torch.load(save_path)  # (n_trials, saved_tokens, n_heads, head_dim)
        if raw.shape[1] >= n_tokens:
            print(f"Loading activations from {save_path}")
            return raw[:, -n_tokens:].mean(dim=0).mean(dim=0).reshape(n_layers, n_heads, model_head_dim)
        print(f"Cached file has {raw.shape[1]} token(s), need {n_tokens} — re-collecting...")

    test_size = 1 - train_split
    n_test_examples = 1

    model_config_prepend_bos = True
    prepend_bos = False if model_config_prepend_bos else True

    layers = np.repeat(np.arange(0, n_layers), n_heads)
    heads = np.tile(np.arange(0, n_heads), n_layers)
    all_heads_ids = np.stack((layers, heads), axis=1)

    last_token = (n_tokens == 1)
    activations, hooks = setup_hooks(model, all_heads_ids, last_token=last_token)

    generator = torch.Generator()
    generator.manual_seed(seed)
    dataset = load_dataset(task, root_data_dir=dataset_path, test_size=test_size, seed=seed)

    # Shape: (n_trials, n_tokens, n_heads, head_dim); n_tokens=1 keeps original semantics
    activation_storage = torch.zeros(n_trials, n_tokens, len(all_heads_ids), model_head_dim, device=device)
    for i in tqdm(range(n_trials), desc=f"Generating activations for {task}"):
        train_perm = torch.randperm(len(dataset['train']), generator=generator)[:n_shots].tolist()
        word_pairs = dataset['train'][train_perm]

        val_perm = torch.randperm(len(dataset['valid']), generator=generator)[:n_test_examples]
        word_pairs_test = dataset['valid'][val_perm.tolist()]

        prompt_data = word_pairs_to_prompt_data(word_pairs, query_target_pair=word_pairs_test, prepend_bos_token=prepend_bos, shuffle_labels=False, prefixes=prefixes, separators=seperators)

        query = prompt_data['query_target']['input']
        _, prompt_string = get_token_meta_labels(prompt_data, tokenizer, query, prepend_bos=model_config_prepend_bos)

        inputs = tokenizer([prompt_string], return_tensors='pt').to(device)
        acts = extract_activations(model, inputs, activations, all_heads_ids, n_heads, resid_dim, model_head_dim, last_token=last_token, device=device)
        # acts: (1, seq_len, n_heads, head_dim) — take the last n_tokens positions
        activation_storage[i] = acts.squeeze(0)[-n_tokens:]

    if save_path:
        parent = os.path.dirname(save_path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        print(f"Saving activations to {save_path}")
        torch.save(activation_storage, save_path)
    else:
        print("Warning activations aren't saved")

    for hook in hooks:
        hook.remove()

    # (n_trials, n_tokens, n_layers*n_heads, head_dim) -> (n_layers, n_heads, head_dim)
    return activation_storage.mean(dim=0).mean(dim=0).reshape(n_layers, n_heads, model_head_dim)


def collect_mlp(model, tokenizer, task, n_shots, n_trials, prefixes, seperators, dataset_path, train_split, device, save_path, n_tokens=1, seed=42):

    n_layers = model.config.num_hidden_layers
    hidden_size = model.config.hidden_size

    if save_path and os.path.exists(save_path):
        raw = torch.load(save_path)  # (n_trials, saved_tokens, n_layers, hidden_size)
        if raw.shape[1] >= n_tokens:
            print(f"Loading MLP activations from {save_path}")
            return raw[:, -n_tokens:].mean(dim=0).mean(dim=0)
        print(f"Cached file has {raw.shape[1]} token(s), need {n_tokens} — re-collecting...")

    test_size = 1 - train_split
    n_test_examples = 1

    # Model configuration
    model_config_prepend_bos = True
    prepend_bos = False if model_config_prepend_bos else True

    layer_ids = list(range(n_layers))
    activations, hooks = setup_hooks_mlp(model, layer_ids)

    generator = torch.Generator()
    generator.manual_seed(seed)
    dataset = load_dataset(task, root_data_dir=dataset_path, test_size=test_size, seed=seed)

    # Shape: (n_trials, n_tokens, n_layers, hidden_size)
    activation_storage = torch.zeros(n_trials, n_tokens, n_layers, hidden_size, device=device)
    for i in tqdm(range(n_trials), desc=f"Generating MLP activations for {task}"):
        train_perm = torch.randperm(len(dataset['train']), generator=generator)[:n_shots].tolist()
        word_pairs = dataset['train'][train_perm]

        val_perm = torch.randperm(len(dataset['valid']), generator=generator)[:n_test_examples]
        word_pairs_test = dataset['valid'][val_perm.tolist()]

        prompt_data = word_pairs_to_prompt_data(word_pairs, query_target_pair=word_pairs_test, prepend_bos_token=prepend_bos, shuffle_labels=False, prefixes=prefixes, separators=seperators)

        query = prompt_data['query_target']['input']
        _, prompt_string = get_token_meta_labels(prompt_data, tokenizer, query, prepend_bos=model_config_prepend_bos)

        inputs = tokenizer([prompt_string], return_tensors='pt').to(device)
        acts = extract_activations_mlp(model, inputs, activations, layer_ids, device=device)
        # acts: (1, seq_len, n_layers, hidden_size) — take the last n_tokens positions
        activation_storage[i] = acts.squeeze(0)[-n_tokens:]

    if save_path:
        parent = os.path.dirname(save_path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        print(f"Saving MLP activations to {save_path}")
        torch.save(activation_storage, save_path)
    else:
        print("Warning: MLP activations aren't saved")

    for hook in hooks:
        hook.remove()

    # (n_trials, n_tokens, n_layers, hidden_size) -> (n_layers, hidden_size)
    return activation_storage.mean(dim=0).mean(dim=0)

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
