"""Collect mean attention-head activations using pre-filtered instruction prompts.

Uses the prompts and per-prompt filter sets produced by prompt_filter_main.py
(stored under storage/) instead of ICL demonstrations.
"""
import json
import os

import numpy as np
import torch
from tqdm import tqdm

from utils_v2 import extract_activations_attn, setup_hooks
from prompt_utils import get_token_meta_labels, load_dataset, word_pairs_to_prompt_data


def collect_prompt_attn(
    model,
    tokenizer,
    task,
    n_best_prompts,
    n_trials_per_prompt,
    prefixes,
    separators,
    dataset_path,
    train_split,
    device,
    save_path,
    prompts_root,
    results_root,
    model_short,
    seed=42,
):
    """Collect mean attention head activations using pre-filtered instruction prompts.

    Selects the top ``n_best_prompts`` by train top-1 accuracy, then for each
    prompt samples ``n_trials_per_prompt`` examples from the per-prompt filter
    set (examples the model already answers correctly) and records the last-token
    attention head activations.

    Returns:
        Tensor of shape (n_layers, n_heads, head_dim).
    """
    n_heads = model.config.num_attention_heads
    n_layers = model.config.num_hidden_layers
    resid_dim = model.config.hidden_size
    model_head_dim = getattr(model.config, "head_dim", resid_dim // n_heads)

    per_prompt_results_path = os.path.join(results_root, model_short, task, "per_prompt_results.json")
    prompts_path = os.path.join(prompts_root, f"{task}_prompts.json")

    with open(per_prompt_results_path) as f:
        per_prompt_results = json.load(f)
    with open(prompts_path) as f:
        all_prompts = json.load(f)["prompts"]

    train_topk = per_prompt_results["train"]["clean_topk"]
    # Sort all prompts by train top-1 accuracy (descending) and take the best N.
    # Only consider prompts present in the stored prompts file.
    prompt_set = set(all_prompts)
    selected_prompts = sorted(
        (p for p in train_topk if p in prompt_set),
        key=lambda p: train_topk[p][0][1],
        reverse=True,
    )[:n_best_prompts]

    # Shared filter set: examples that all selected prompts answer correctly.
    # Sum rank lists across prompts; indices that remain 0 passed every prompt.
    rank_lists = per_prompt_results["train"]["clean_rank_list"]
    summed_ranks = np.sum([np.array(rank_lists[p]) for p in selected_prompts], axis=0)
    filter_set = np.where(summed_ranks == 0)[0]

    if save_path and os.path.exists(save_path):
        raw = torch.load(save_path)
        print(f"Loading prompt activations from {save_path}")
        return raw.mean(dim=0).mean(dim=0).reshape(n_layers, n_heads, model_head_dim)

    layers = np.repeat(np.arange(n_layers), n_heads)
    heads = np.tile(np.arange(n_heads), n_layers)
    all_heads_ids = np.stack((layers, heads), axis=1)
    activations, hooks = setup_hooks(model, all_heads_ids, last_token=True)

    test_size = 1 - train_split
    dataset = load_dataset(task, root_data_dir=dataset_path, test_size=test_size, seed=seed)

    # model prepends BOS automatically, so we don't add it manually
    prepend_bos = False

    rng = np.random.default_rng(seed)

    # (n_prompts, n_trials, n_layers*n_heads, head_dim)
    storage = torch.zeros(
        len(selected_prompts),
        n_trials_per_prompt,
        len(all_heads_ids),
        model_head_dim,
        device=device,
    )

    for p_idx, prompt in enumerate(selected_prompts):
        fs = filter_set
        if len(fs) == 0:
            print(f"Warning: no passing examples for prompt '{prompt}', skipping")
            continue

        for trial in tqdm(range(n_trials_per_prompt), desc=f"{task} prompt {p_idx + 1}/{len(selected_prompts)}"):
            idx = int(rng.choice(fs))
            example = dataset["train"][idx]

            # 0-shot: instruction only, no ICL examples
            prompt_data = word_pairs_to_prompt_data(
                {"input": [], "output": []},
                query_target_pair=example,
                instructions=prompt,
                prefixes=prefixes,
                separators=separators,
                prepend_bos_token=prepend_bos,
            )

            query = prompt_data["query_target"]["input"]
            if isinstance(query, list):
                query = query[0]
            _, prompt_string = get_token_meta_labels(prompt_data, tokenizer, query, prepend_bos=True)

            inputs = tokenizer([prompt_string], return_tensors="pt").to(device)
            acts = extract_activations_attn(
                model, inputs, activations, all_heads_ids,
                n_heads, resid_dim, model_head_dim, last_token=True, device=device,
            )
            # acts: (1, 1, n_all_heads, head_dim) -> (n_all_heads, head_dim)
            storage[p_idx, trial] = acts.squeeze(0).squeeze(0)

    for hook in hooks:
        hook.remove()

    if save_path:
        parent = os.path.dirname(save_path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        print(f"Saving prompt activations to {save_path}")
        torch.save(storage, save_path)

    # (n_prompts, n_trials, n_all_heads, head_dim) -> (n_layers, n_heads, head_dim)
    return storage.mean(dim=0).mean(dim=0).reshape(n_layers, n_heads, model_head_dim)
