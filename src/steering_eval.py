"""Joint attention + MLP additive steering evaluation."""
import torch
from tqdm import tqdm

from utils_v2 import split_activations_by_head
from prompt_utils import load_dataset, word_pairs_to_prompt_data, get_answer_id, create_prompt, rank_of_token, topk_acc


def build_steering_vecs(task_mean_acts, top_heads, top_mlp_layers):
    """Slice mean activations for the selected heads and MLP layers.

    Args:
        task_mean_acts: dict with
            "heads": Tensor (n_layers, n_heads, head_dim) — already meaned over trials
            "mlps":  Tensor (n_layers, hidden_size)       — already meaned over trials
        top_heads:      list of (layer, head) index pairs
        top_mlp_layers: list of layer indices

    Returns:
        steering_heads: Tensor (n_top_heads,  head_dim)
        steering_mlp:   Tensor (n_top_layers, hidden_size)
    """
    heads_tensor = task_mean_acts["heads"]  # (n_layers, n_heads, head_dim)
    mlp_tensor   = task_mean_acts["mlps"]   # (n_layers, hidden_size)

    steering_heads = torch.stack([heads_tensor[L, H] for L, H in top_heads]).to(torch.bfloat16)
    steering_mlp   = mlp_tensor[top_mlp_layers].to(torch.bfloat16)

    return steering_heads, steering_mlp


@torch.inference_mode()
def eval_joint_steered(
    task, model, tokenizer, top_heads, top_mlp_layers,
    steering_heads, steering_mlp, n_heads,
    steer_scale, cfg, n_shots=0,
):
    """Eval with additive attention + MLP steering at a single scale.

    n_shots: number of ICL examples to include in each eval prompt (0 = zero-shot).
    Returns dict: {task, scale, n_shots, steered_topk, n_examples}
    """
    prefixes     = cfg["prefixes"]
    separators   = cfg["separators"]
    dataset_path = cfg["dataset_path"]
    train_split  = cfg["train_split"]
    seed         = cfg["seed"]
    device       = cfg["device"]
    max_examples = cfg["steering_sweep"].get("max_examples")

    dataset  = load_dataset(task, root_data_dir=dataset_path, test_size=1 - train_split, seed=seed)
    test_set  = dataset["test"]
    train_set = dataset["train"]
    n_examples = len(test_set["input"]) if isinstance(test_set, dict) else len(test_set)
    if max_examples is not None:
        n_examples = min(n_examples, max_examples)

    rng = torch.Generator()
    rng.manual_seed(seed)

    # Group heads by layer for efficient hook installation
    heads_by_layer = {}
    for global_id, (L, H) in enumerate(top_heads):
        heads_by_layer.setdefault(L, []).append((global_id, H))

    hooks = []

    # Attention: additive pre-hook on o_proj
    for L, head_list in heads_by_layer.items():
        def attn_hook(module, inputs, _L=L, _head_list=head_list):
            x = inputs[0]
            B, S, D = x.shape
            x_heads = split_activations_by_head(x, n_heads=n_heads)
            for global_id, H in _head_list:
                x_heads[:, -1, H] = x_heads[:, -1, H] + steer_scale * steering_heads[global_id].to(x.device)
            return (x_heads.view(B, S, D),)

        hooks.append(
            model.model.layers[L].self_attn.o_proj.register_forward_pre_hook(attn_hook, with_kwargs=False)
        )

    # MLP: additive post-hook
    for mlp_idx, L in enumerate(top_mlp_layers):
        def mlp_hook(module, inp, out, _idx=mlp_idx):
            out = out.clone()
            out[:, -1, :] = out[:, -1, :] + steer_scale * steering_mlp[_idx].to(out.device)
            return out

        hooks.append(model.model.layers[L].mlp.register_forward_hook(mlp_hook))

    ranks = []
    for j in tqdm(range(n_examples), desc=f"Eval {task} scale={steer_scale}", leave=False):
        word_pair_test = {k: v[j] for k, v in test_set.items()} if isinstance(test_set, dict) else test_set[j]

        if n_shots > 0:
            perm = torch.randperm(len(train_set["input"]), generator=rng)[:n_shots].tolist()
            word_pairs = {k: [train_set[k][i] for i in perm] for k in train_set}
        else:
            word_pairs = {"input": [], "output": []}

        prompt_data = word_pairs_to_prompt_data(
            word_pairs=word_pairs,
            query_target_pair=word_pair_test,
            prepend_bos_token=False,
            shuffle_labels=False,
            prefixes=prefixes,
            separators=separators,
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

    metrics = {K: topk_acc(ranks, K) for K in (1, 2, 3)}
    return {
        "task": task,
        "scale": steer_scale,
        "n_shots": n_shots,
        "steered_topk": metrics,
        "n_examples": n_examples,
    }
