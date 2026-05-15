"""Shared utilities for eval scripts."""
import json
import os
import numpy as np
import torch
from tqdm import tqdm

from prompt_utils import load_dataset, word_pairs_to_prompt_data, create_prompt, get_answer_id, rank_of_token, topk_acc
from utils_v2 import split_activations_by_head

PREFIXES           = {"input": "Q:", "output": "A:", "instructions": ""}
SEPARATORS         = {"input": "\n", "output": "\n\n", "instructions": ""}
PROMPT_SEPARATORS  = {"input": "\n", "output": "\n\n", "instructions": "\n"}


# ── helpers ─────────────────────────────────────────────────

def load_per_prompt_results(results_root, model_short, task):
    path = os.path.join(results_root, model_short, task, "per_prompt_results.json")
    with open(path) as f:
        return json.load(f)


def top_prompts(results_root, model_short, task, n=5):
    """Return the top-n prompts for a task sorted by train top-1 accuracy."""
    results = load_per_prompt_results(results_root, model_short, task)
    train_topk = results["train"]["clean_topk"]
    return sorted(train_topk, key=lambda p: train_topk[p][0][1], reverse=True)[:n]


def test_filter_set(results_root, model_short, task, prompts):
    """Shared test filter set: indices where all selected prompts answer correctly."""
    results = load_per_prompt_results(results_root, model_short, task)
    rank_lists = results["test"]["clean_rank_list"]
    summed = np.sum([np.array(rank_lists[p]) for p in prompts], axis=0)
    return np.where(summed == 0)[0]


def train_filter_set(results_root, model_short, task, prompts):
    """Shared train filter set: indices where all selected prompts answer correctly."""
    results = load_per_prompt_results(results_root, model_short, task)
    # Changed from "test" to "train"
    rank_lists = results["train"]["clean_rank_list"]
    summed = np.sum([np.array(rank_lists[p]) for p in prompts], axis=0)
    return np.where(summed == 0)[0]


def save_json(out, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"  saved → {path}")



# ── Mean activation loading ───────────────────────────────────────────────────


def load_prompt_cie(results_root: str, model_short: str, task: str, selection: str, baseline: str = "equiprobable") -> torch.Tensor:
    """Load prompt-based indirect effect for a single task.

    File shape: (n_prompts, n_trials, n_layers, n_heads)
    Returns:    (n_layers, n_heads) averaged over prompts and trials.
    """
    if selection == "lrp":
        path = os.path.join(results_root, model_short, task, f"{task}_lrp.pt")
    else:
        path = os.path.join(results_root, model_short, task, f"{task}_{baseline}_indirect_effect.pt")

    raw = torch.load(path, map_location="cpu")  # (n_prompts, n_trials, n_layers, n_heads)
    return raw.mean(dim=0).mean(dim=0)


def load_prompt_aie(results_root: str, model_short: str, tasks: list[str], selection: str, baseline: str = "equiprobable") -> torch.Tensor:
    """Macro-average prompt indirect effect over tasks.

    Returns: (n_layers, n_heads)
    Missing task files are skipped with a warning.
    """
    attn_list = []
    for task in tasks:
        try:
            attn_list.append(load_prompt_cie(results_root, model_short, task, selection, baseline))
        except FileNotFoundError:
            print(f"  [warn] missing prompt IE for '{task}', skipping")
    return torch.stack(attn_list).mean(dim=0)

def load_mean_prompt_attn(results_root: str, model_short: str, task: str) -> torch.Tensor:
    """Load prompt-based mean attn activations from the reference storage layout.

    File shape: (n_layers, n_heads, n_dummy_labels, head_dim)
    Returns:    (n_all_heads, 1, head_dim)  — compatible with build_steering_vecs
    Takes the last dummy-label position (query output token).
    """
    path = os.path.join(results_root, model_short, task, f"{task}_mean_head_activations.pt")
    raw = torch.load(path, map_location="cpu")  # (n_layers, n_heads, n_dummy_labels, head_dim)
    n_layers, n_heads, _, head_dim = raw.shape
    return raw[:, :, -1, :].reshape(n_layers * n_heads, head_dim).unsqueeze(1)  # (n_all_heads, 1, head_dim)



# ── Module selection ──────────────────────────────────────────────────────────

def select_modules(
    attn_cie: torch.Tensor,              # (n_layers, n_heads)
    mlp_cie:  torch.Tensor | None = None,  # (n_layers,); required for "mlp"/"joint"
    module_type: str = "attn",           # "attn" | "mlp" | "joint"
    topk_attn:  int = 20,
    topk_mlp:   int = 6,
    topk_joint: int = 26,
) -> tuple[list, list]:
    """Return (top_heads, top_mlp_layers).

    top_heads:      list of (layer, head) pairs
    top_mlp_layers: list of layer indices
    """
    n_layers, n_heads = attn_cie.shape

    if module_type == "attn":
        idx = torch.topk(attn_cie.flatten(), k=topk_attn).indices
        return [(int(i) // n_heads, int(i) % n_heads) for i in idx], []
    else:
        raise NotImplementedError

# ── Steering vector construction ──────────────────────────────────────────────

def build_steering_vecs(
    mean_attn: torch.Tensor,              # (n_all_heads, n_tokens, head_dim)
    top_heads: list,
    n_heads: int,
    mean_mlp:  torch.Tensor | None = None,  # (n_layers, n_tokens, hidden_size)
    top_mlp_layers: list | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Slice mean activations for the selected heads and MLP layers.

    Returns:
        dist_fv: (n_top_heads, n_tokens, head_dim)    in bfloat16
        dist_fv_mlp:   (n_top_layers, n_tokens, hidden_size) in bfloat16
    """
    top_mlp_layers = top_mlp_layers or []
    global_ids = [L * n_heads + H for L, H in top_heads]
    n_tokens = mean_attn.shape[1]
    dist_fv = mean_attn[global_ids].to(torch.bfloat16) if global_ids else torch.zeros(0, n_tokens, mean_attn.shape[-1])
    if top_mlp_layers and mean_mlp is not None:
        dist_fv_mlp = mean_mlp[top_mlp_layers].to(torch.bfloat16)
    else:
        dist_fv_mlp = torch.zeros(0, n_tokens, mean_attn.shape[-1])
    return dist_fv, dist_fv_mlp


# ── Evaluation ────────────────────────────────────────────────────────────────

@torch.inference_mode()
def eval_prompt_steered(
    task, model, tokenizer,
    top_heads, dist_fv,
    n_heads, steer_scale,
    prompt,
    dataset_path, train_split, seed, device,
    n_tokens=1,
    filter_set=None,
) -> dict:
    """Eval with additive attn steering using an instruction prompt.

    Identical to eval_steered but constructs '{instruction}\\nQ: {input}\\nA:'
    instead of the bare 'Q: {input}\\nA:' zero-shot format.

    prompt:     instruction string to prepend to every eval query.
    filter_set: optional array of test-split indices to evaluate on (shared
                intersection of examples all selected prompts answer correctly).
    n_tokens:   number of trailing token positions to steer simultaneously.
    Returns dict with task, scale, prompt, steered_topk, n_examples.
    """
    # Use a 2-way split when filter_set is provided so indices match the reference
    # (reference uses train/test only, no validation set).
    dataset = load_dataset(task, root_data_dir=dataset_path, test_size=1 - train_split, seed=seed,
                           split_valid=(filter_set is None))

    heads_by_layer = {}
    for gid, (L, H) in enumerate(top_heads):
        heads_by_layer.setdefault(L, []).append((gid, H))

    hooks = []
    for L, head_list in heads_by_layer.items():
        def attn_hook(_module, inputs, _head_list=head_list):
            x = inputs[0]
            B, S, D = x.shape
            x_heads = split_activations_by_head(x, n_heads=n_heads)
            for gid, H in _head_list:
                for t in range(n_tokens):
                    pos = -(n_tokens - t)
                    x_heads[:, pos, H] = x_heads[:, pos, H] + steer_scale * dist_fv[gid, t].to(x.device)
                    #x_heads[:, pos, H] = steer_scale * dist_fv[gid, t].to(x.device)
            return (x_heads.view(B, S, D),)
        hooks.append(model.model.layers[L].self_attn.o_proj.register_forward_pre_hook(attn_hook, with_kwargs=False))

    indices = filter_set if filter_set is not None else range(len(dataset["test"]))

    ranks = []
    for j in tqdm(indices, desc=f"{task} scale={steer_scale}", leave=False):
        word_pair_test = dataset["test"][int(j)]

        prompt_data = word_pairs_to_prompt_data(
            word_pairs={"input": [], "output": []},
            query_target_pair=word_pair_test,
            instructions=prompt,
            prepend_bos_token=False,
            shuffle_labels=False,
            prefixes=PREFIXES,
            separators=PROMPT_SEPARATORS,
        )

        target = prompt_data["query_target"]["output"]
        target = target[0] if isinstance(target, list) else target
        sentence = create_prompt(prompt_data)
        target_token_id = get_answer_id(sentence, target, tokenizer)[0]

        toks = tokenizer(sentence, return_tensors="pt").to(device)
        logits = model(input_ids=toks["input_ids"]).logits[:, -1, :].float()
        ranks.append(int(rank_of_token(logits, target_token_id)))

    for h in hooks:
        h.remove()

    return {
        "task":         task,
        "scale":        steer_scale,
        "prompt":       prompt,
        "topk":         {K: topk_acc(ranks, K) for K in (1, 2, 3)},
        "ranks":        ranks,
        "n_examples":   len(ranks),
    }


# ── Function-vector steering (reference approach) ─────────────────────────────

def compute_fv(
    mean_attn: torch.Tensor,  # (n_all_heads, 1, head_dim)
    top_heads: list,           # [(L, H), ...]
    n_heads: int,
    model,
) -> torch.Tensor:
    """Build a function vector by summing o_proj(zero-padded mean head act) across top heads.

    For each selected head (L, H): place mean_attn[L*n_heads+H, 0] at the head's
    slice of a zero vector, pass through that layer's o_proj, accumulate the result.
    This matches compute_function_vector() from the reference project.

    Returns: (resid_dim,) in model dtype, on model.device.
    """
    resid_dim = model.config.hidden_size
    head_dim  = getattr(model.config, "head_dim", resid_dim // n_heads)

    embed_dim = n_heads * head_dim

    fv = torch.zeros(1, 1, resid_dim, device=model.device, dtype=model.dtype)

    with torch.inference_mode():
        for L, H in top_heads:
            global_id = L * n_heads + H
            x = torch.zeros(1, 1, embed_dim, device=model.device, dtype=model.dtype)
            x[0, 0, H * head_dim : (H + 1) * head_dim] = (
                mean_attn[global_id, 0].to(device=model.device, dtype=model.dtype)
            )
            fv = fv + model.model.layers[L].self_attn.o_proj(x)

    return fv.reshape(resid_dim)


@torch.inference_mode()
def eval_fv_steered(
    task, model, tokenizer,
    fv, edit_layer, steer_scale,
    prompt,
    dataset_path, train_split, seed, device,
    filter_set=None,
) -> dict:
    """Eval with a function vector added to the residual stream at a single layer.

    fv:         (resid_dim,) function vector (from compute_fv)
    edit_layer: transformer layer index at which to inject the FV
    steer_scale: multiplicative scale applied to fv before addition
    prompt:     instruction string prepended to every query ('' for zero-shot)
    filter_set: optional array of test-split indices to evaluate on

    The FV is added to the last-token position of the residual stream output of
    model.model.layers[edit_layer], matching the reference's TraceDict/edit_output
    approach in prompt_based_function_vector.py.
    """
    dataset = load_dataset(task, root_data_dir=dataset_path, test_size=1 - train_split, seed=seed,
                           split_valid=(filter_set is None))

    fv_scaled = (steer_scale * fv).to(device)

    def _fv_hook(_module, _inp, out):
        # transformers 5.x: LlamaDecoderLayer returns a plain tensor, not a tuple
        if isinstance(out, tuple):
            hidden = out[0].clone()
            hidden[:, -1, :] = hidden[:, -1, :] + fv_scaled
            #hidden[:, -1, :] =  fv_scaled
            return (hidden,) + out[1:]
        out = out.clone()
        out[:, -1, :] = out[:, -1, :] + fv_scaled
        #out[:, -1, :] = fv_scaled
        return out

    hook = model.model.layers[edit_layer].register_forward_hook(_fv_hook)

    indices = filter_set if filter_set is not None else range(len(dataset["test"]))

    ranks = []
    for j in tqdm(indices, desc=f"{task} scale={steer_scale}", leave=False):
        word_pair_test = dataset["test"][int(j)]

        prompt_data = word_pairs_to_prompt_data(
            word_pairs={"input": [], "output": []},
            query_target_pair=word_pair_test,
            instructions=prompt,
            prepend_bos_token=False,
            shuffle_labels=False,
            prefixes=PREFIXES,
            separators=PROMPT_SEPARATORS,
        )

        target = prompt_data["query_target"]["output"]
        target = target[0] if isinstance(target, list) else target
        sentence = create_prompt(prompt_data)
        target_token_id = get_answer_id(sentence, target, tokenizer)[0]

        toks = tokenizer(sentence, return_tensors="pt").to(device)
        logits = model(input_ids=toks["input_ids"]).logits[:, -1, :].float()
        ranks.append(int(rank_of_token(logits, target_token_id)))

    hook.remove()

    return {
        "task":         task,
        "scale":        steer_scale,
        "prompt":       prompt,
        "topk":         {K: topk_acc(ranks, K) for K in (1, 2, 3)},
        "ranks":        ranks,
        "n_examples":   len(ranks),
    }


# ------------------ not really in use right now -------------------


# def load_mean_attn(cache_dir: str, model_short: str, task: str, n_tokens: int = 1) -> torch.Tensor:
#     """Load mean attn activations averaged over trials.

#     Cache shape: (n_trials, n_saved_tokens, n_layers*n_heads, head_dim)
#     Returns:     (n_layers*n_heads, n_tokens, head_dim)
#     """
#     path = os.path.join(cache_dir, model_short, "mean_acts", "attn", f"{task}.pt")
#     raw = torch.load(path, map_location="cpu")  # (n_trials, n_saved_tokens, n_all_heads, head_dim)
#     if raw.shape[1] < n_tokens:
#         raise ValueError(
#             f"mean_acts attn cache for '{task}' has {raw.shape[1]} token position(s), "
#             f"but n_tokens={n_tokens} was requested. Re-collect with a larger n_tokens."
#         )
#     # average over trials, take last n_tokens positions, reorder to (n_all_heads, n_tokens, head_dim)
#     return raw.mean(dim=0)[-n_tokens:].permute(1, 0, 2)



# def load_mean_mlp(cache_dir: str, model_short: str, task: str, n_tokens: int = 1) -> torch.Tensor:
#     """Load mean MLP activations averaged over trials.

#     Cache shape: (n_trials, n_saved_tokens, n_layers, hidden_size)
#     Returns:     (n_layers, n_tokens, hidden_size)
#     """
#     path = os.path.join(cache_dir, model_short, "mean_acts", "mlp", f"{task}.pt")
#     raw = torch.load(path, map_location="cpu")  # (n_trials, n_saved_tokens, n_layers, hidden_size)
#     if raw.shape[1] < n_tokens:
#         raise ValueError(
#             f"mean_acts mlp cache for '{task}' has {raw.shape[1]} token position(s), "
#             f"but n_tokens={n_tokens} was requested. Re-collect with a larger n_tokens."
#         )
#     # average over trials, take last n_tokens positions, reorder to (n_layers, n_tokens, hidden_size)
#     return raw.mean(dim=0)[-n_tokens:].permute(1, 0, 2)



# def select_modules(
#     attn_cie: torch.Tensor,              # (n_layers, n_heads)
#     mlp_cie:  torch.Tensor | None = None,  # (n_layers,); required for "mlp"/"joint"
#     module_type: str = "attn",           # "attn" | "mlp" | "joint"
#     topk_attn:  int = 20,
#     topk_mlp:   int = 6,
#     topk_joint: int = 26,
# ) -> tuple[list, list]:
#     """Return (top_heads, top_mlp_layers).

#     top_heads:      list of (layer, head) pairs
#     top_mlp_layers: list of layer indices
#     """
#     n_layers, n_heads = attn_cie.shape

#     if module_type == "attn":
#         idx = torch.topk(attn_cie.flatten(), k=topk_attn).indices
#         return [(int(i) // n_heads, int(i) % n_heads) for i in idx], []

#     if module_type == "mlp":
#         idx = torch.topk(mlp_cie, k=topk_mlp).indices
#         return [], idx.tolist()

#     # joint: pool attn and mlp entries, rank together, pick top-k
#     entries = []
#     for l in range(n_layers):
#         for h in range(n_heads):
#             entries.append(("attn", l, h, attn_cie[l, h].item()))
#         entries.append(("mlp", l, -1, mlp_cie[l].item()))

#     entries.sort(key=lambda x: x[3], reverse=True)
#     top = entries[:topk_joint]

#     top_heads      = [(l, h) for kind, l, h, _ in top if kind == "attn"]
#     top_mlp_layers = [l      for kind, l, _, _ in top if kind == "mlp"]
#     return top_heads, top_mlp_layers


# ── CIE loading ──────────────────────────────────────────────────────────────

# def load_cie(cache_dir: str, model_short: str, shot: str, task: str):
#     """Load trial-averaged CIE tensors for a single task.

#     Returns:
#         attn: (n_layers, n_heads)
#         mlp:  (n_layers,)
#     """
#     attn_path = os.path.join(cache_dir, model_short, "cie", shot, "attn", f"{task}.pt")
#     mlp_path  = os.path.join(cache_dir, model_short, "cie", shot, "mlp",  f"{task}.pt")
#     return (
#         torch.load(attn_path, map_location="cpu"),
#         torch.load(mlp_path,  map_location="cpu"),
#     )


# def load_aie(cache_dir: str, model_short: str, shot: str, tasks: list[str]):
#     """Macro-average CIE over tasks (AIE).

#     Returns:
#         attn: (n_layers, n_heads)
#         mlp:  (n_layers,)
#     Missing task files are skipped with a warning.
#     """
#     attn_list, mlp_list = [], []
#     for task in tasks:
#         try:
#             attn, mlp = load_cie(cache_dir, model_short, shot, task)
#             attn_list.append(attn)
#             mlp_list.append(mlp)
#         except FileNotFoundError:
#             print(f"  [warn] {shot}: missing CIE for '{task}', skipping")

#     return (
#         torch.stack(attn_list).mean(dim=0),
#         torch.stack(mlp_list).mean(dim=0),
#     )


# @torch.inference_mode()
# def eval_steered(
#     task, model, tokenizer,
#     top_heads, top_mlp_layers,
#     dist_fv, dist_fv_mlp,
#     n_heads, steer_scale,
#     dataset_path, train_split, seed, device,
#     n_shots=0, n_tokens=1,
# ) -> dict:
#     """Eval with additive attn + MLP steering at a single scale.

#     n_shots:  ICL examples per prompt (0 = zero-shot).
#     n_tokens: number of trailing token positions to steer simultaneously.
#     Returns dict with task, scale, n_shots, steered_topk, n_examples.
#     """
#     dataset = load_dataset(task, root_data_dir=dataset_path, test_size=1 - train_split, seed=seed)

#     heads_by_layer = {}
#     for gid, (L, H) in enumerate(top_heads):
#         heads_by_layer.setdefault(L, []).append((gid, H))

#     hooks = []
#     for L, head_list in heads_by_layer.items():
#         def attn_hook(module, inputs, _head_list=head_list):
#             x = inputs[0]
#             B, S, D = x.shape
#             x_heads = split_activations_by_head(x, n_heads=n_heads)
#             for gid, H in _head_list:
#                 for t in range(n_tokens):
#                     pos = -(n_tokens - t)
#                     x_heads[:, pos, H] = x_heads[:, pos, H] + steer_scale * dist_fv[gid, t].to(x.device)
#             return (x_heads.view(B, S, D),)
#         hooks.append(model.model.layers[L].self_attn.o_proj.register_forward_pre_hook(attn_hook, with_kwargs=False))

#     for mlp_idx, L in enumerate(top_mlp_layers):
#         def mlp_hook(module, inp, out, _idx=mlp_idx):
#             out = out.clone()
#             for t in range(n_tokens):
#                 pos = -(n_tokens - t)
#                 out[:, pos, :] = out[:, pos, :] + steer_scale * dist_fv_mlp[_idx, t].to(out.device)
#             return out
#         hooks.append(model.model.layers[L].mlp.register_forward_hook(mlp_hook))

#     rng = torch.Generator()
#     rng.manual_seed(seed)

#     ranks = []
#     for j in tqdm(range(len(dataset["test"])), desc=f"{task} scale={steer_scale}", leave=False):
#         word_pair_test = dataset["test"][j]

#         if n_shots > 0:
#             perm = torch.randperm(len(dataset["train"]), generator=rng)[:n_shots].tolist()
#             word_pairs = dataset["train"][perm]
#         else:
#             word_pairs = {"input": [], "output": []}

#         prompt_data = word_pairs_to_prompt_data(
#             word_pairs=word_pairs,
#             query_target_pair=word_pair_test,
#             prepend_bos_token=False,
#             shuffle_labels=False,
#             prefixes=PREFIXES,
#             separators=SEPARATORS,
#         )

#         target = prompt_data["query_target"]["output"]
#         target = target[0] if isinstance(target, list) else target
#         sentence = create_prompt(prompt_data)
#         target_token_id = get_answer_id(sentence, target, tokenizer)[0]

#         toks = tokenizer(sentence, return_tensors="pt").to(device)
#         logits = model(input_ids=toks["input_ids"]).logits[:, -1, :].float()
#         ranks.append(int(rank_of_token(logits, target_token_id)))

#     for h in hooks:
#         h.remove()

#     return {
#         "task":         task,
#         "scale":        steer_scale,
#         "n_shots":      n_shots,
#         "steered_topk": {K: topk_acc(ranks, K) for K in (1, 2, 3)},
#         "n_examples":   len(ranks),
#     }



# @torch.inference_mode()
# def eval_prompt_steered(
#     task, model, tokenizer,
#     top_heads, dist_fv,
#     n_heads, steer_scale,
#     prompt,
#     dataset_path, train_split, seed, device,
#     top_mlp_layers=None, dist_fv_mlp=None,
#     n_tokens=1,
#     filter_set=None,
# ) -> dict:
#     """Eval with additive attn steering using an instruction prompt.

#     Identical to eval_steered but constructs '{instruction}\\nQ: {input}\\nA:'
#     instead of the bare 'Q: {input}\\nA:' zero-shot format.

#     prompt:     instruction string to prepend to every eval query.
#     filter_set: optional array of test-split indices to evaluate on (shared
#                 intersection of examples all selected prompts answer correctly).
#     n_tokens:   number of trailing token positions to steer simultaneously.
#     Returns dict with task, scale, prompt, steered_topk, n_examples.
#     """
#     # Use a 2-way split when filter_set is provided so indices match the reference
#     # (reference uses train/test only, no validation set).
#     dataset = load_dataset(task, root_data_dir=dataset_path, test_size=1 - train_split, seed=seed,
#                            split_valid=(filter_set is None))

#     heads_by_layer = {}
#     for gid, (L, H) in enumerate(top_heads):
#         heads_by_layer.setdefault(L, []).append((gid, H))

#     hooks = []
#     for L, head_list in heads_by_layer.items():
#         def attn_hook(_module, inputs, _head_list=head_list):
#             x = inputs[0]
#             B, S, D = x.shape
#             x_heads = split_activations_by_head(x, n_heads=n_heads)
#             for gid, H in _head_list:
#                 for t in range(n_tokens):
#                     pos = -(n_tokens - t)
#                     x_heads[:, pos, H] = x_heads[:, pos, H] + steer_scale * dist_fv[gid, t].to(x.device)
#             return (x_heads.view(B, S, D),)
#         hooks.append(model.model.layers[L].self_attn.o_proj.register_forward_pre_hook(attn_hook, with_kwargs=False))

#     for mlp_idx, L in enumerate(top_mlp_layers):
#         def mlp_hook(_module, _inp, out, _idx=mlp_idx):
#             out = out.clone()
#             for t in range(n_tokens):
#                 pos = -(n_tokens - t)
#                 out[:, pos, :] = out[:, pos, :] + steer_scale * dist_fv_mlp[_idx, t].to(out.device)
#             return out
#         hooks.append(model.model.layers[L].mlp.register_forward_hook(mlp_hook))

#     indices = filter_set if filter_set is not None else range(len(dataset["test"]))

#     ranks = []
#     for j in tqdm(indices, desc=f"{task} scale={steer_scale}", leave=False):
#         word_pair_test = dataset["test"][int(j)]

#         prompt_data = word_pairs_to_prompt_data(
#             word_pairs={"input": [], "output": []},
#             query_target_pair=word_pair_test,
#             instructions=prompt,
#             prepend_bos_token=False,
#             shuffle_labels=False,
#             prefixes=PREFIXES,
#             separators=PROMPT_SEPARATORS,
#         )

#         target = prompt_data["query_target"]["output"]
#         target = target[0] if isinstance(target, list) else target
#         sentence = create_prompt(prompt_data)
#         target_token_id = get_answer_id(sentence, target, tokenizer)[0]

#         toks = tokenizer(sentence, return_tensors="pt").to(device)
#         logits = model(input_ids=toks["input_ids"]).logits[:, -1, :].float()
#         ranks.append(int(rank_of_token(logits, target_token_id)))

#     for h in hooks:
#         h.remove()

#     return {
#         "task":         task,
#         "scale":        steer_scale,
#         "prompt":       prompt,
#         "steered_topk": {K: topk_acc(ranks, K) for K in (1, 2, 3)},
#         "n_examples":   len(ranks),
#     }
