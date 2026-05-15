"""Steer a single sample or custom string and show multi-token generation.

Edit the CONFIG block below, then run:
    python demo_steer.py
"""

import json, os, sys, random
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from eval.utils import (
    load_mean_prompt_attn, load_prompt_aie, load_prompt_cie,
    select_modules, build_steering_vecs, compute_fv,
    test_filter_set,
)
from prompt_utils import load_dataset, word_pairs_to_prompt_data, create_prompt
from utils_v2 import split_activations_by_head

# ── CONFIG ────────────────────────────────────────────────────────────────────

MODEL   = "meta-llama/Llama-3.2-3B-Instruct"
DEVICE  = "cuda:0"

TASK    = "english-spanish"

# Custom prompt string — set to None to sample from the dataset instead.
# The string should end with "A:" (or similar) so generation continues from there.
CUSTOM_STRING = "Q: I love Yellowstone National Park. A:"
# CUSTOM_STRING = "Q: happy\nA:"

# Dataset sample index — set to None for a random test example (only used when CUSTOM_STRING is None).
SAMPLE_IDX = 6

# Steering scenario:
#   intervention : "fv"   – add function vector to residual stream at FV_EDIT_LAYER
#                  "dist" – patch per-attention-head activations via hook
#   score_type   : "ie"  | "lrp" | "universal"
#   aggregation  : "per_task" | "averaged"  (ignored when score_type="universal")
INTERVENTION  = "dist"
SCORE_TYPE    = "lrp"       # "ie" | "lrp" | "universal"
AGGREGATION   = "averaged" # "per_task" | "averaged"

SCALE         = 1.
FV_EDIT_LAYER = 11   # reference default for 3B; 14 for 8B

# Universal heads settings (only used when SCORE_TYPE="universal")
N_UNIVERSAL_HEADS = 20
HEAD_FILE_KEY     = "both_all_top_heads"

GEN_TOKENS = 26   # max new tokens for generation

# ── Paths (mirror run_eval_simple.py) ─────────────────────────────────────────

STORAGE_ROOT      = "storage"
INTERMEDIATE_PATH = os.path.join(STORAGE_ROOT, "function_vectors", "full_results_prompt_based_short")
DATASET_PATH      = os.path.join(STORAGE_ROOT, "function_vectors", "dataset_files")
TOP_HEADS_PATH    = os.path.join(STORAGE_ROOT, "function_vectors", "full_results_top_heads")
TOPK_ATTN   = 20
TRAIN_SPLIT = 0.7
SEED        = 42
N_TOKENS    = 1  # trailing token positions to steer (dist mode)

# ── Head loading ───────────────────────────────────────────────────────────────

def load_heads(model_short):
    if SCORE_TYPE == "universal":
        path = os.path.join(TOP_HEADS_PATH, f"{model_short}_{HEAD_FILE_KEY}.json")
        with open(path) as f:
            data = json.load(f)
        heads = [(int(L), int(H)) for L, H in data["top_heads"][:N_UNIVERSAL_HEADS]]
        print(f"Universal heads ({N_UNIVERSAL_HEADS}, {HEAD_FILE_KEY}): {heads[:5]} ...")
        return heads

    if AGGREGATION == "averaged":
        base = os.path.join(INTERMEDIATE_PATH, model_short)
        tasks = sorted(d for d in os.listdir(base) if os.path.isdir(os.path.join(base, d)))
        scores = load_prompt_aie(INTERMEDIATE_PATH, model_short, tasks, SCORE_TYPE)
    else:
        scores = load_prompt_cie(INTERMEDIATE_PATH, model_short, TASK, SCORE_TYPE)

    heads, _ = select_modules(scores, topk_attn=TOPK_ATTN)
    print(f"Selected {len(heads)} heads ({SCORE_TYPE}/{AGGREGATION}): {heads[:5]} ...")
    return heads

# ── Prompt from dataset ────────────────────────────────────────────────────────

def sample_prompt():
    dataset = load_dataset(TASK, root_data_dir=DATASET_PATH,
                           test_size=1 - TRAIN_SPLIT, seed=SEED, split_valid=False)
    test_set = dataset["test"]

    model_short = MODEL.split("/")[-1]
    prompts_path = os.path.join(INTERMEDIATE_PATH, model_short, TASK,
                                f"{TASK}_equiprobable_indirect_effect_baseline_prompts.json")
    with open(prompts_path) as f:
        prompts = list(json.load(f).keys())
    filter_set = test_filter_set(INTERMEDIATE_PATH, model_short, TASK, prompts)
    print(f"Filter set: {len(filter_set)} / {len(test_set)} test examples")

    pool = filter_set if len(filter_set) > 0 else range(len(test_set))
    idx  = SAMPLE_IDX if SAMPLE_IDX is not None else random.choice(pool)
    print(f"Dataset index: {idx}")

    pair = test_set[int(idx)]
    prompt_data = word_pairs_to_prompt_data(
        word_pairs={"input": [], "output": []},
        query_target_pair=pair,
        instructions="",
        prepend_bos_token=False,
        shuffle_labels=False,
        prefixes={"input": "Q:", "output": "A:", "instructions": ""},
        separators={"input": "\n", "output": "\n\n", "instructions": "\n"},
    )
    sentence = create_prompt(prompt_data)
    target   = prompt_data["query_target"]["output"]
    target   = target[0] if isinstance(target, list) else target
    return sentence, str(target)

# ── Hooks ──────────────────────────────────────────────────────────────────────

def register_dist_hooks(model, top_heads, dist_fv, n_heads):
    heads_by_layer = {}
    for gid, (L, H) in enumerate(top_heads):
        heads_by_layer.setdefault(L, []).append((gid, H))

    hooks = []
    for L, head_list in heads_by_layer.items():
        def attn_hook(_module, inputs, _hl=head_list):
            x = inputs[0]
            B, S, D = x.shape
            x_heads = split_activations_by_head(x, n_heads=n_heads)
            for gid, H in _hl:
                for t in range(N_TOKENS):
                    pos = -(N_TOKENS - t)
                    x_heads[:, pos, H] = x_heads[:, pos, H] + SCALE * dist_fv[gid, t].to(x.device)
            return (x_heads.view(B, S, D),)
        hooks.append(
            model.model.layers[L].self_attn.o_proj.register_forward_pre_hook(attn_hook, with_kwargs=False)
        )
    return hooks


def register_fv_hook(model, fv):
    fv_scaled = (SCALE * fv).to(next(model.parameters()).device)

    def _hook(_module, _inp, out):
        if isinstance(out, tuple):
            hidden = out[0].clone()
            hidden[:, -1, :] = hidden[:, -1, :] + fv_scaled
            return (hidden,) + out[1:]
        out = out.clone()
        out[:, -1, :] = out[:, -1, :] + fv_scaled
        return out

    return model.model.layers[FV_EDIT_LAYER].register_forward_hook(_hook)

# ── Generation ─────────────────────────────────────────────────────────────────

@torch.inference_mode()
def generate(prompt_str, model, tokenizer):
    toks = tokenizer(prompt_str, return_tensors="pt").to(DEVICE)
    out  = model.generate(
        **toks, max_new_tokens=GEN_TOKENS,
        do_sample=False, temperature=None, top_p=None,
        pad_token_id=tokenizer.eos_token_id,
    )
    return tokenizer.decode(out[0][toks["input_ids"].shape[1]:], skip_special_tokens=True)

# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    model_short = MODEL.split("/")[-1]

    print(f"Loading {MODEL} ...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, trust_remote_code=True, torch_dtype=torch.bfloat16,
    ).to(DEVICE).eval()
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    n_heads = model.config.num_attention_heads

    if CUSTOM_STRING is not None:
        prompt_str, expected = CUSTOM_STRING, "?"
    else:
        prompt_str, expected = sample_prompt()

    scenario = f"{INTERVENTION}/{SCORE_TYPE}" + (f"/{AGGREGATION}" if SCORE_TYPE != "universal" else "")
    print(f"\nTask     : {TASK}")
    print(f"Scenario : {scenario}  scale={SCALE}" + (f"  edit_layer={FV_EDIT_LAYER}" if INTERVENTION == "fv" else ""))
    print(f"Prompt   : {repr(prompt_str)}")
    print(f"Expected : {repr(expected)}")
    print("=" * 64)

    top_heads = load_heads(model_short)
    mean_attn = load_mean_prompt_attn(INTERMEDIATE_PATH, model_short, TASK)

    if INTERVENTION == "fv":
        fv = compute_fv(mean_attn, top_heads, n_heads, model)
    else:
        dist_fv, _ = build_steering_vecs(mean_attn, top_heads, n_heads)
        dist_fv = dist_fv.to(DEVICE)

    baseline_out = generate(prompt_str, model, tokenizer)
    print(f"\n[BASELINE]\n  {repr(baseline_out)}")

    if INTERVENTION == "fv":
        hook = register_fv_hook(model, fv)
        steered_out = generate(prompt_str, model, tokenizer)
        hook.remove()
        print(f"\n[FV  scale={SCALE}  layer={FV_EDIT_LAYER}]\n  {repr(steered_out)}")
    else:
        hooks = register_dist_hooks(model, top_heads, dist_fv, n_heads)
        steered_out = generate(prompt_str, model, tokenizer)
        for h in hooks:
            h.remove()
        print(f"\n[DIST  scale={SCALE}]\n  {repr(steered_out)}")


if __name__ == "__main__":
    main()
