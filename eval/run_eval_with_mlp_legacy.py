"""Full steering eval across all parameter combinations.

Dimensions:
  eval_shot : 0shot | 10shot          (eval prompt ICL examples)
  cie_shot  : 0shot | 10shot          (CIE source for head selection)
  modules   : attn | mlp | joint      (top-20 attn / top-6 MLP / top-26 mixed)
  selection : task_aie | per_task     (shared macro-avg heads vs per-task heads)
  scales    : 0.5 … 3.0 in 0.5 steps (swept within each experiment)

18 experiments total: eval_shot × cie_shot (3 pairs) × 3 modules × 2 selections.
"""
import os, sys
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from eval.utils import (
    load_cie, load_aie, load_mean_attn, load_mean_mlp,
    select_modules, build_steering_vecs, eval_steered,
)

# MODEL       = "Qwen/Qwen2.5-7B-Instruct"
MODEL       = "meta-llama/Llama-3.2-1B-Instruct"
MODEL_SHORT = MODEL.split("/")[-1]
CACHE_DIR   = "./cache"
RESULTS_DIR = f"./results/{MODEL_SHORT}/full_eval_10heads"
DEVICE      = "cuda:1"

DATASET_PATH = "./dataset_files_fv"
TRAIN_SPLIT  = 0.7
SEED         = 42

SCALES   = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
N_TOKENS = 1

TOPK_ATTN  = 10
TOPK_MLP   = 6
TOPK_JOINT = 26

SHOT_TO_N = {"0shot": 0, "10shot": 10}

TASKS = [
    "antonym", "capitalize", "capitalize_first_letter", "country-capital",
    "country-currency", "english-french", "english-german", "english-spanish",
    "landmark-country", "lowercase_first_letter", "national_parks", "park-country",
    "person-sport", "present-past", "product-company", "sentiment",
    "singular-plural", "synonym",
]

# (eval_shot, cie_shot): standard matched pairs + 0shot eval with 10shot CIE
SHOT_PAIRS = [("0shot", "0shot"), ("10shot", "10shot"), ("0shot", "10shot")]

COMBOS = [
    (eval_shot, cie_shot, modules, selection)
    for eval_shot, cie_shot in SHOT_PAIRS
    # for modules   in ["attn", "mlp", "joint"]
    for modules   in ["attn"]
    for selection in ["task_aie", "per_task"]
]


def save_name(eval_shot, cie_shot, modules, selection):
    if eval_shot == cie_shot:
        return f"{eval_shot}_{modules}_{selection}.pt"
    return f"{eval_shot}_cie{cie_shot}_{modules}_{selection}.pt"


def run_experiment(eval_shot, cie_shot, modules, selection, model, tokenizer, n_heads):
    n_shots_eval = SHOT_TO_N[eval_shot]

    if selection == "task_aie":
        aie_attn, aie_mlp = load_aie(CACHE_DIR, MODEL_SHORT, cie_shot, TASKS)
        shared_heads, shared_mlp_layers = select_modules(
            aie_attn, aie_mlp, modules, TOPK_ATTN, TOPK_MLP, TOPK_JOINT,
        )

    all_results = {}
    for task in tqdm(TASKS, desc=f"{eval_shot}/cie{cie_shot}/{modules}/{selection}"):
        if selection == "per_task":
            task_attn_cie, task_mlp_cie = load_cie(CACHE_DIR, MODEL_SHORT, cie_shot, task)
            top_heads, top_mlp_layers = select_modules(
                task_attn_cie, task_mlp_cie, modules, TOPK_ATTN, TOPK_MLP, TOPK_JOINT,
            )
        else:
            top_heads, top_mlp_layers = shared_heads, shared_mlp_layers

        mean_attn = load_mean_attn(CACHE_DIR, MODEL_SHORT, task, N_TOKENS)
        mean_mlp  = load_mean_mlp(CACHE_DIR, MODEL_SHORT, task, N_TOKENS)
        dist_fv, dist_fv_mlp = build_steering_vecs(mean_attn, top_heads, n_heads, mean_mlp=mean_mlp, top_mlp_layers=top_mlp_layers)
        dist_fv = dist_fv.to(DEVICE)
        dist_fv_mlp   = dist_fv_mlp.to(DEVICE)

        task_results = {}
        for scale in SCALES:
            task_results[scale] = eval_steered(
                task=task, model=model, tokenizer=tokenizer,
                top_heads=top_heads, top_mlp_layers=top_mlp_layers,
                dist_fv=dist_fv, dist_fv_mlp=dist_fv_mlp,
                n_heads=n_heads, steer_scale=scale,
                dataset_path=DATASET_PATH, train_split=TRAIN_SPLIT, seed=SEED,
                device=DEVICE, n_shots=n_shots_eval, n_tokens=N_TOKENS,
            )

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

    print(f"{len(COMBOS)} experiments × {len(TASKS)} tasks × {len(SCALES)} scales\n")

    for eval_shot, cie_shot, modules, selection in COMBOS:
        save_path = os.path.join(RESULTS_DIR, save_name(eval_shot, cie_shot, modules, selection))
        if os.path.exists(save_path):
            print(f"[skip] {eval_shot}/cie{cie_shot}/{modules}/{selection}")
            continue

        print(f"\n{'='*60}\n  eval={eval_shot}  cie={cie_shot}  {modules}  {selection}\n{'='*60}")
        results = run_experiment(eval_shot, cie_shot, modules, selection, model, tokenizer, n_heads)

        torch.save({
            "eval_shot": eval_shot, "cie_shot": cie_shot,
            "modules": modules, "selection": selection,
            "scales": SCALES, "tasks": TASKS, "results": results,
        }, save_path)
        print(f"  saved → {save_path}")


if __name__ == "__main__":
    main()
