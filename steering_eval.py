import os
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

from utils_v2 import *
from prompt_utils import *
import numpy as np


def load_top_heads_from_cie(cie_path, task, n_heads, topk_heads, device='cpu'):
    """Select the top-K heads by mean CIE for a given task."""
    cie = torch.load(f"{cie_path}/{task}.pt", map_location=device)  # (cie_trials, n_layers*n_heads)
    mean_cie = cie.mean(dim=0)
    _, topk_inds = torch.topk(mean_cie, k=topk_heads)
    return [(int(idx) // n_heads, int(idx) % n_heads) for idx in topk_inds]


def eval_sae(args, model=None, tokenizer=None, top_heads=None, device=None):
    """Steering evaluation using raw head activations."""

    pre_hooks = []
    prefixes = {"input": "Q:", "output": "A:", "instructions": ""}
    separators = {"input": "\n", "output": "\n\n", "instructions": ""}
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') if device is None else device

    seed = getattr(args, 'seed', 42)
    generator = torch.Generator()
    generator.manual_seed(seed)

    if model is None or tokenizer is None:
        print(f"Loading model: {args.model_name}")
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        ).to(device)
        model.eval()

        tokenizer = AutoTokenizer.from_pretrained(args.model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

    n_heads = model.config.num_attention_heads
    resid_dim = model.config.hidden_size
    model_head_dim = model.config.head_dim
    model_config_prepend_bos = True
    prepend_bos = not model_config_prepend_bos

    if top_heads is None:
        if args.head_mode == "task":
            top_heads = load_top_heads_from_cie(args.cie_path, args.dataset_name, n_heads, args.topk_heads, device=device)
        else:  # "global"
            top_heads = load_top_heads(args.head_ids_path, n_heads, args.topk_heads)

    print(f'Processing with {len(top_heads)} heads')

    act_hooks, hooks = setup_hooks(model, top_heads, last_token=True)

    test_size = 1 - args.train_split
    dataset = load_dataset(args.dataset_name, root_data_dir=args.dataset_path, test_size=test_size, seed=seed)

    # Reuse all-heads activations from compute_indirect_effect_v2 if available,
    # selecting the columns that correspond to top_heads.
    all_acts_file = f"{args.activations_path}/{args.dataset_name}.pt"
    if os.path.exists(all_acts_file):
        all_acts = torch.load(all_acts_file, map_location=device)  # (trials, n_all_heads, head_dim)
        global_ids = [L * n_heads + H for L, H in top_heads]
        activation_storage = all_acts[:, global_ids, :]  # (trials, n_top_heads, head_dim)
        print(f"Loaded and selected activations: {activation_storage.shape}")
    else:
        activation_storage = torch.zeros(args.mean_acts_trials, len(top_heads), model_head_dim, device=device)
        with torch.inference_mode():
            for i in range(args.mean_acts_trials):
                train_perm = torch.randperm(len(dataset['train']), generator=generator)[:args.n_icl_examples].tolist()
                word_pairs = dataset['train'][train_perm]

                val_perm = torch.randperm(len(dataset['valid']), generator=generator)[:1]
                word_pairs_test = dataset['valid'][val_perm.tolist()]

                prompt_data = word_pairs_to_prompt_data(word_pairs, query_target_pair=word_pairs_test, prepend_bos_token=prepend_bos, shuffle_labels=False, prefixes=prefixes, separators=separators)

                query = prompt_data['query_target']['input']
                _, prompt_string = get_token_meta_labels(prompt_data, tokenizer, query, prepend_bos=model_config_prepend_bos)

                inputs = tokenizer([prompt_string], return_tensors='pt').to(device)
                acts = extract_activations_attn(model, inputs, act_hooks, top_heads, n_heads, resid_dim, model_head_dim, last_token=True, device=device)
                activation_storage[i] = acts.squeeze(0).squeeze(0)  # (n_top_heads, head_dim)

    for hook in hooks:
        hook.remove()

    # mean over trials: (n_top_heads, head_dim)
    steering_heads = activation_storage.mean(dim=0).to(torch.bfloat16)

    heads_by_layer = {}
    for global_head_id, (L, H) in enumerate(top_heads):
        heads_by_layer.setdefault(L, []).append((global_head_id, H))

    steer_scale = getattr(args, "steer_scale", 1.)

    def steerhook_input(module, inputs):
        x = inputs[0]
        B, S, D = x.shape
        x_heads = split_activations_by_head(x, n_heads=n_heads)
        L = module._steer_layer_idx
        if L in heads_by_layer:
            for global_head_id, head_id in heads_by_layer[L]:
                x_heads[:, -1, head_id] = x_heads[:, -1, head_id] + steer_scale * steering_heads[global_head_id]
        return (x_heads.view(B, S, D),)

    for L in heads_by_layer.keys():
        m = model.model.layers[L].self_attn.o_proj
        m._steer_layer_idx = L
        pre_hooks.append(m.register_forward_pre_hook(steerhook_input, with_kwargs=False))

    print(f"Installed steering hooks on {len(pre_hooks)} layers with scale={steer_scale}")

    ranks = []
    for j in tqdm(range(len(dataset['test'])), desc="Zero-shot eval"):
        word_pair_test = dataset['test'][j]
        prompt_data = word_pairs_to_prompt_data(
            word_pairs={'input': [], 'output': []},
            query_target_pair=word_pair_test,
            prepend_bos_token=False,
            shuffle_labels=False,
            prefixes=prefixes,
            separators=separators,
        )

        query, target = prompt_data['query_target']['input'], prompt_data['query_target']['output']
        query = query[0] if isinstance(query, list) else query
        target = target[0] if isinstance(target, list) else target

        sentence = create_prompt(prompt_data)
        target_token_id = get_answer_id(sentence, target, tokenizer)[0]

        toks = tokenizer(sentence, return_tensors='pt').to(device)
        last_token = model(input_ids=toks["input_ids"]).logits[:, -1, :].float()
        ranks.append(int(rank_of_token(last_token, target_token_id)))

    for h in pre_hooks:
        h.remove()

    metrics = {K: topk_acc(ranks, K) for K in (1, 2, 3)}
    print("top-k:", [(K, metrics[K]) for K in (1, 2, 3)])

    return {
        "scale": steer_scale,
        "task": args.dataset_name,
        "steered_topk": metrics,
        "n_examples": len(dataset['test']),
    }


class Argsclass:
    pass


def main():
    args = Argsclass()
    args.model_name = "Qwen/Qwen3-4B-Instruct-2507"
    model_short = args.model_name.split('/')[-1]
    args.dataset_path = "dataset_files_fv"
    args.dataset_name = "antonym"
    args.head_mode = "task"       # "global" or "task"
    args.head_ids_path = f"heads_{model_short}.pt"
    args.cie_path = f"cie_scores/{model_short}"
    args.activations_path = f"mean_acts/{model_short}"
    args.topk_heads = 20
    args.train_split = 0.7
    args.mean_acts_trials = 100
    args.n_icl_examples = 10
    args.steer_scale = 1.0
    args.seed = 42

    eval_sae(args)


if __name__ == '__main__':
    main()
