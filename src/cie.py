"""Causal Indirect Effect (CIE) computation for attention heads and MLP layers.

Two corruption schemes are supported:
  n_shots_corrupted == 0  ->  zero-shot baseline (no demonstrations)
  n_shots_corrupted  > 0  ->  n-shot with randomised / shuffled labels
"""
import os
import torch
from tqdm import tqdm

from utils_v2 import split_activations_by_head
from prompt_utils import load_dataset, word_pairs_to_prompt_data, get_token_meta_labels, get_answer_id

# Llama / most decoder models prepend BOS automatically; we never add it manually.
_MODEL_PREPENDS_BOS = True
_PREPEND_BOS = not _MODEL_PREPENDS_BOS   # False


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _build_corrupted_prompt(dataset, n_shots_corrupted, cie_gen, prefixes, separators):
    """Return a prompt_data dict for one CIE trial.

    Zero-shot (n_shots_corrupted == 0): no demonstrations, random valid query.
    n-shot randomised (n_shots_corrupted  > 0): n demos with shuffled labels.
    """
    val_perm       = torch.randperm(len(dataset['valid']), generator=cie_gen)[:1]
    word_pair_test = dataset['valid'][val_perm.tolist()]

    train_perm = torch.randperm(len(dataset['train']), generator=cie_gen)[:n_shots_corrupted].tolist()
    word_pairs = dataset['train'][train_perm] if n_shots_corrupted > 0 else {'input': [], 'output': []}

    return word_pairs_to_prompt_data(
        word_pairs,
        query_target_pair=word_pair_test,
        shuffle_labels=(n_shots_corrupted > 0),
        prepend_bos_token=_PREPEND_BOS,
        prefixes=prefixes,
        separators=separators,
    )


def _get_prompt_string_and_target_ids(prompt_data, tokenizer, device):
    """Return (prompt_string, token_ids_tensor) for a prompt_data dict."""
    query  = prompt_data['query_target']['input']
    target = prompt_data['query_target']['output']
    target = target[0] if isinstance(target, list) else target

    _, prompt_string = get_token_meta_labels(prompt_data, tokenizer, query=query, prepend_bos=_MODEL_PREPENDS_BOS)

    token_ids = get_answer_id(prompt_string, target, tokenizer)
    if isinstance(token_ids, list):
        token_ids = token_ids[:1]
    return prompt_string, torch.LongTensor(token_ids).to(device)


# ---------------------------------------------------------------------------
# Per-module CIE
# ---------------------------------------------------------------------------

def compute_cie_heads(model, tokenizer, task, mean_head_acts, n_shots_corrupted, n_trials,
                      prefixes, separators, dataset_path, device, train_split=0.7, seed=42, save_path=None):
    """Per-head Causal Indirect Effect.

    For each trial we build a corrupted prompt, run a clean forward pass, then
    re-run n_layers * n_heads times each time patching one head's last-token
    pre-o_proj activation with its task mean.

    Returns: Tensor (n_layers, n_heads) — CIE averaged over trials.
    """
    if save_path and os.path.exists(save_path):
        print(f"[CIE] Loading head CIE for '{task}' from {save_path}")
        return torch.load(save_path, map_location=device)

    n_heads  = model.config.num_attention_heads
    n_layers = model.config.num_hidden_layers

    dataset = load_dataset(task, root_data_dir=dataset_path, test_size=1.0 - train_split, seed=seed)
    cie_gen = torch.Generator()
    cie_gen.manual_seed(seed)

    indirect_effect = torch.zeros(n_trials, n_layers, n_heads, device=device)

    with torch.inference_mode():
        for idx in tqdm(range(n_trials), desc=f"Head CIE [{task}]"):
            prompt_data = _build_corrupted_prompt(dataset, n_shots_corrupted, cie_gen, prefixes, separators)
            prompt_string, token_ids = _get_prompt_string_and_target_ids(prompt_data, tokenizer, device)

            inputs      = tokenizer(prompt_string, return_tensors='pt').to(device)
            clean_probs = torch.softmax(model(**inputs).logits[:, -1, :][0], dim=-1)

            for layer_id in range(n_layers):
                for head_id in range(n_heads):
                    mean_act = mean_head_acts[layer_id, head_id].to(model.dtype)

                    def _pre_hook(module, inp, _hid=head_id, _act=mean_act):
                        x = inp[0]
                        B, S, D = x.shape
                        x_heads = split_activations_by_head(x, n_heads)
                        x_heads[:, -1, _hid] = _act
                        return (x_heads.reshape(B, S, D),)

                    hook = model.model.layers[layer_id].self_attn.o_proj.register_forward_pre_hook(_pre_hook, with_kwargs=False)
                    iv_probs = torch.softmax(model(**inputs).logits[:, -1, :], dim=-1)
                    hook.remove()

                    indirect_effect[idx, layer_id, head_id] = (iv_probs - clean_probs).squeeze()[token_ids]

    cie = indirect_effect.mean(dim=0)

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        torch.save(cie, save_path)
        print(f"[CIE] Saved head CIE for '{task}' to {save_path}")

    return cie


def compute_cie_mlp(model, tokenizer, task, mean_mlp_acts, n_shots_corrupted, n_trials,
                    prefixes, separators, dataset_path, device, train_split=0.7, seed=42, save_path=None):
    """Per-layer MLP Causal Indirect Effect.

    For each trial, for each MLP layer we patch the last-token MLP output with
    its task mean and measure the probability change.

    Returns: Tensor (n_layers,) — CIE averaged over trials.
    """
    if save_path and os.path.exists(save_path):
        print(f"[CIE] Loading MLP CIE for '{task}' from {save_path}")
        return torch.load(save_path, map_location=device)

    n_layers = model.config.num_hidden_layers

    dataset = load_dataset(task, root_data_dir=dataset_path, test_size=1.0 - train_split, seed=seed)
    cie_gen = torch.Generator()
    cie_gen.manual_seed(seed)

    indirect_effect = torch.zeros(n_trials, n_layers, device=device)

    with torch.inference_mode():
        for idx in tqdm(range(n_trials), desc=f"MLP  CIE [{task}]"):
            prompt_data = _build_corrupted_prompt(dataset, n_shots_corrupted, cie_gen, prefixes, separators)
            prompt_string, token_ids = _get_prompt_string_and_target_ids(prompt_data, tokenizer, device)

            inputs      = tokenizer(prompt_string, return_tensors='pt').to(device)
            clean_probs = torch.softmax(model(**inputs).logits[:, -1, :][0], dim=-1)

            for layer_id in range(n_layers):
                mean_act = mean_mlp_acts[layer_id].to(model.dtype)

                def _post_hook(module, inp, out, _act=mean_act):
                    out = out.clone()
                    out[:, -1, :] = _act
                    return out

                hook = model.model.layers[layer_id].mlp.register_forward_hook(_post_hook)
                iv_probs = torch.softmax(model(**inputs).logits[:, -1, :], dim=-1)
                hook.remove()

                indirect_effect[idx, layer_id] = (iv_probs - clean_probs).squeeze()[token_ids]

    cie = indirect_effect.mean(dim=0)

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        torch.save(cie, save_path)
        print(f"[CIE] Saved MLP CIE for '{task}' to {save_path}")

    return cie


# ---------------------------------------------------------------------------
# Convenience wrappers
# ---------------------------------------------------------------------------

def get_or_compute_cie(model, tokenizer, task, mean_head_acts, mean_mlp_acts, n_shots_corrupted,
                       n_trials, prefixes, separators, dataset_path, device,
                       train_split=0.7, seed=42, cache_dir=None):
    """Compute (or load from cache) CIE for both heads and MLP for *one* task.

    Returns dict with keys:
        "heads"  ->  Tensor (n_layers, n_heads)
        "mlp"    ->  Tensor (n_layers,)
    """
    head_path = mlp_path = None
    if cache_dir:
        shot_key  = f"{n_shots_corrupted}shot"
        head_path = os.path.join(cache_dir, shot_key, "attn", f"{task}.pt")
        mlp_path  = os.path.join(cache_dir, shot_key, "mlp",  f"{task}.pt")

    head_cie = compute_cie_heads(model, tokenizer, task, mean_head_acts, n_shots_corrupted,
                                 n_trials, prefixes, separators, dataset_path, device,
                                 train_split, seed, head_path)
    mlp_cie  = compute_cie_mlp(model, tokenizer, task, mean_mlp_acts, n_shots_corrupted,
                                n_trials, prefixes, separators, dataset_path, device,
                                train_split, seed, mlp_path)

    return {"heads": head_cie, "mlp": mlp_cie}


def get_or_compute_prompt_cie(model, tokenizer, task, mean_head_acts, n_trials,
                               prefixes, separators, dataset_path, device,
                               train_split=0.7, seed=42, cache_dir=None):
    """Compute (or load from cache) prompt-based CIE for attention heads for one task.

    Corrupted baseline is zero-shot (empty instruction), since mean activations
    were collected with instruction prompts. This measures how much the
    instruction-derived head activations cause task behaviour when patched into
    a prompt that has no instruction.

    Returns: Tensor (n_layers, n_heads).
    """
    head_path = None
    if cache_dir:
        head_path = os.path.join(cache_dir, "prompt", "attn", f"{task}.pt")

    return compute_cie_heads(
        model, tokenizer, task, mean_head_acts,
        n_shots_corrupted=0, n_trials=n_trials,
        prefixes=prefixes, separators=separators,
        dataset_path=dataset_path, device=device,
        train_split=train_split, seed=seed,
        save_path=head_path,
    )


def compute_aie_heads(model, tokenizer, tasks, mean_head_acts_per_task, n_shots_corrupted,
                      n_trials, prefixes, separators, dataset_path, device,
                      train_split=0.7, seed=42):
    """Average Indirect Effect (AIE) for attention heads across multiple tasks.

    Returns: Tensor (n_layers, n_heads) averaged over the supplied task list.
    """
    all_cie = [
        compute_cie_heads(model, tokenizer, task, mean_head_acts_per_task[task], n_shots_corrupted,
                          n_trials, prefixes, separators, dataset_path, device, train_split, seed)
        for task in tasks
    ]
    return torch.stack(all_cie).mean(dim=0)
