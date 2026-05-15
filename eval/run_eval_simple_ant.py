"""Prompt-based steering eval: top-N instruction prompts, attention modules only."""
import json
import os, sys
import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from eval.utils import (
    load_mean_prompt_attn, load_prompt_aie,
    select_modules, build_steering_vecs, eval_prompt_steered,
    compute_fv, eval_fv_steered,
)

MODEL       = "meta-llama/Llama-3.2-3B-Instruct"
MODEL_SHORT = MODEL.split("/")[-1]
CACHE_DIR   = "./cache"
RESULTS_DIR = f"./results/{MODEL_SHORT}/simple_eval"
DEVICE      = "mps"

DATASET_PATH   = "storage/dataset_files"
TRAIN_SPLIT    = 0.7
SEED           = 42

#RESULTS_ROOT   = "storage/function_vectors/lrp"
RESULTS_ROOT   = "storage/function_vectors/full_results_prompt_based_short"


N_BEST_PROMPTS = 5

# SCALES   = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
SCALES = [1.0]
N_TOKENS = 1

TOPK_ATTN  = 20
TOPK_MLP   = 0
TOPK_JOINT = 0

MODULES = "attn"

# TASKS = [
#     "antonym", "capitalize", "capitalize_first_letter", "country-capital",
#     "country-currency", "english-french", "english-german", "english-spanish",
#     "landmark-country", "lowercase_first_letter", "national_parks", "park-country",
#     "person-sport", "present-past", "product-company", "sentiment",
#     "singular-plural", "synonym",
# ]
TASKS=["antonym"]

SELECTIONS = ["task_aie","lrp"]

# ── FV experiment settings ────────────────────────────────────────────────────
# Layer at which to inject the function vector into the residual stream.
# The reference used layer 14 for Llama-3.2-3B-Instruct on these tasks.
FV_EDIT_LAYER = 14
FV_SELECTIONS = ["task_aie"]


def _load_per_prompt_results(task):
    path = os.path.join(RESULTS_ROOT, MODEL_SHORT, task, "per_prompt_results.json")
    with open(path) as f:
        return json.load(f)


def _top_prompts(task, n=N_BEST_PROMPTS):
    """Return the top-n prompts for a task sorted by train top-1 accuracy."""
    results = _load_per_prompt_results(task)
    train_topk = results["train"]["clean_topk"]
    return sorted(train_topk, key=lambda p: train_topk[p][0][1], reverse=True)[:n]


def _test_filter_set(task, prompts):
    """Shared test filter set: indices where all selected prompts answer correctly."""
    results = _load_per_prompt_results(task)
    rank_lists = results["test"]["clean_rank_list"]
    summed = np.sum([np.array(rank_lists[p]) for p in prompts], axis=0)
    return np.where(summed == 0)[0]


def run_experiment(selection, model, tokenizer, n_heads):
    aie_attn = load_prompt_aie(RESULTS_ROOT, MODEL_SHORT, TASKS,selection)
    shared_heads, shared_mlp_layers = select_modules(
        aie_attn, torch.zeros(aie_attn.shape[0]),
        MODULES, TOPK_ATTN, TOPK_MLP, TOPK_JOINT,
    )

    all_results = {}
    for task in tqdm(TASKS, desc=f"prompt/{selection}"):
        mean_attn = load_mean_prompt_attn(RESULTS_ROOT, MODEL_SHORT, task)
        steering_heads, steering_mlp = build_steering_vecs(mean_attn, torch.empty(1), shared_heads, shared_mlp_layers, n_heads)
        steering_heads = steering_heads.to(DEVICE)

        #rand_steering_heads = torch.randn_like(steering_heads)
        #steering_heads = rand_steering_heads * (torch.norm(steering_heads) / torch.norm(rand_steering_heads))

        prompts = _top_prompts(task)
        filter_set = _test_filter_set(task, prompts)

        task_results = {}
        for scale in SCALES:
            per_prompt = [
                eval_prompt_steered(
                    task=task, model=model, tokenizer=tokenizer,
                    top_heads=shared_heads,top_mlp_layers=shared_mlp_layers,
                    steering_heads=steering_heads, steering_mlp=steering_mlp,
                    n_heads=n_heads, steer_scale=scale,
                    prompt=" ",
                    dataset_path=DATASET_PATH, train_split=TRAIN_SPLIT, seed=SEED,
                    device=DEVICE, n_tokens=N_TOKENS,
                    filter_set=filter_set,
                )
                #for prompt in prompts
            ]
            avg_topk = {
                K: sum(r["steered_topk"][K] for r in per_prompt) / len(per_prompt)
                for K in (1, 2, 3)
            }
            task_results[scale] = {**per_prompt[0], "steered_topk": avg_topk, "prompts": prompts}

        all_results[task] = task_results

    return all_results


def run_fv_experiment(selection, model, tokenizer, n_heads):
    """FV approach: sum o_proj(mean_head_act) across top heads → single vector,
    then add to residual stream at FV_EDIT_LAYER during eval."""
    aie_attn = load_prompt_aie(RESULTS_ROOT, MODEL_SHORT, TASKS,selection)
    shared_heads, _ = select_modules(
        aie_attn, torch.zeros(aie_attn.shape[0]),
        MODULES, TOPK_ATTN, TOPK_MLP, TOPK_JOINT,
    )

    all_results = {}
    for task in tqdm(TASKS, desc=f"fv/{selection}"):
        mean_attn = load_mean_prompt_attn(RESULTS_ROOT, MODEL_SHORT, task)

        fv = compute_fv(mean_attn, shared_heads, n_heads, model)
        """
        # 1. Compute the real FV to get its shape and magnitude
        real_fv = compute_fv(mean_attn, shared_heads, n_heads, model)

        # 2. Generate a random vector with the same shape and data type
        random_fv = torch.randn_like(real_fv)

        # 3. Scale the random vector so its L2 norm matches the real FV
        fv = random_fv * (torch.norm(real_fv) / torch.norm(random_fv))
        """
        prompts = _top_prompts(task)
        filter_set = _test_filter_set(task, prompts)

        task_results = {}
        for scale in SCALES:
            per_prompt = [
                eval_fv_steered(
                    task=task, model=model, tokenizer=tokenizer,
                    fv=fv, edit_layer=FV_EDIT_LAYER, steer_scale=scale,
                    prompt=" ",
                    dataset_path=DATASET_PATH, train_split=TRAIN_SPLIT, seed=SEED,
                    device=DEVICE,
                    filter_set=filter_set,
                )
               #for prompt in prompts
            ]
            avg_topk = {
                K: sum(r["steered_topk"][K] for r in per_prompt) / len(per_prompt)
                for K in (1, 2, 3)
            }
            task_results[scale] = {**per_prompt[0], "steered_topk": avg_topk, "prompts": prompts}

        all_results[task] = task_results

    return all_results


def main():
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, trust_remote_code=True, torch_dtype=torch.bfloat16,
    ).to(DEVICE).eval()
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    n_heads = model.config.num_attention_heads
    os.makedirs(RESULTS_DIR, exist_ok=True)

    print(
        f"Running eval: {len(SELECTIONS)} head-patching + {len(FV_SELECTIONS)} FV × {len(TASKS)} tasks × {len(SCALES)} scales\n")

    # --- 1. RUN PER-HEAD PATCHING ---
    for selection in SELECTIONS:
        save_path = os.path.join(RESULTS_DIR, f"prompt_attn_{selection}.pt")

        print(f"\n{'=' * 60}\n  head_patch/{selection}\n{'=' * 60}")
        results = run_experiment(selection, model, tokenizer, n_heads)
        print(results)

        # Print a quick summary of the Top-1 Accuracy for scale 1.0
        if 1.0 in results.get(TASKS[0], {}):
            acc = results[TASKS[0]][1.0]["steered_topk"][1]
            print(f"  -> Top-1 Zero-Shot Accuracy (Scale 1.0): {acc * 100:.2f}%")

        torch.save({
            "selection": selection, "scales": SCALES, "tasks": TASKS, "results": results,
        }, save_path)
        print(f"  saved → {save_path}")

    # --- 2. RUN FUNCTION VECTOR INJECTION ---
    for selection in FV_SELECTIONS:
        save_path = os.path.join(RESULTS_DIR, f"prompt_fv_{selection}.pt")

        print(f"\n{'=' * 60}\n  fv/{selection}  (edit_layer={FV_EDIT_LAYER})\n{'=' * 60}")
        results = run_fv_experiment(selection, model, tokenizer, n_heads)
        print(results)

        if 1.0 in results.get(TASKS[0], {}):
            acc = results[TASKS[0]][1.0]["steered_topk"][1]
            print(f"  -> Top-1 Zero-Shot Accuracy (Scale 1.0): {acc * 100:.2f}%")

        torch.save({
            "selection": selection, "edit_layer": FV_EDIT_LAYER,
            "scales": SCALES, "tasks": TASKS, "results": results,
        }, save_path)
        print(f"  saved → {save_path}")



if __name__ == "__main__":
    main()

