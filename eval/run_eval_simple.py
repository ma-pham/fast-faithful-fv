"""Prompt-based steering eval: top-N instruction prompts, attention modules only."""
import os, sys
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from eval.utils import (
    load_mean_prompt_attn, load_prompt_aie, load_prompt_cie,
    select_modules, build_steering_vecs, eval_prompt_steered,
    compute_fv, eval_fv_steered,
    top_prompts, test_filter_set,
    save_json
)

STORAGE_ROOT = os.environ.get("STORAGE_ROOT")
INTERMEDIATE_PATH = os.path.join(STORAGE_ROOT, "function_vectors", "full_results_prompt_based_short")
DATASET_PATH = os.path.join(STORAGE_ROOT, "function_vectors", "dataset_files")
RESULTS_PATH =  os.path.join(STORAGE_ROOT, "eval")

# MODEL       = "meta-llama/Llama-3.2-3B-Instruct"
MODEL = "Qwen/Qwen3-4B-Instruct-2507"
MODEL_SHORT = MODEL.split("/")[-1]
DEVICE      = "cuda:0"

TRAIN_SPLIT    = 0.7
SEED           = 42

N_BEST_PROMPTS = 5

# SCALES   = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
SCALES = [1.0]
N_TOKENS = 1

TOPK_ATTN = 40

# TASKS = [
#     "antonym", "capitalize", "capitalize_first_letter", "country-capital",
#     "country-currency", "english-french", "english-german", "english-spanish",
#     "landmark-country", "lowercase_first_letter", "national_parks", "park-country",
#     "person-sport", "present-past", "product-company", "sentiment",
#     "singular-plural", "synonym",
# ]
# TASKS=["alphabetically_first_3"]
# TASKS = [
#      "antonym", "capitalize", "country-capital",
#      "country-currency", "english-french", "english-german", "english-spanish",
#      "landmark-country", "national_parks", "park-country",
#      "present-past", "product-company",
#      "singular-plural", "synonym",
# ]

TASKS = [
    # abstractive
    "antonym", "capitalize", "capitalize_first_letter", "capitalize_last_letter", 
    "capitalize_second_letter",
    "country-capital", "country-currency", "english-french", "english-german", "english-spanish",
    "landmark-country", "lowercase_first_letter", "lowercase_last_letter", "national_parks",
    "next_capital_letter", "next_item", "park-country", "present-past", "prev_item",
    "product-company", "singular-plural", "synonym", "word_length",
    # extractive
    #"adjective_v_verb_3", "adjective_v_verb_5",
    "alphabetically_first_3", "alphabetically_first_5",
    "alphabetically_last_3", "alphabetically_last_5",
    "animal_v_object_3", "animal_v_object_5",
    "choose_first_of_3", "choose_first_of_5",
    "choose_last_of_3", "choose_last_of_5",
    "choose_middle_of_3", "choose_middle_of_5",
    "color_v_animal_3", "color_v_animal_5",
    "concept_v_object_3", "concept_v_object_5",
    "conll2003_location", "conll2003_organization", "conll2003_person",
    "fruit_v_animal_3", "fruit_v_animal_5",
    "object_v_concept_3", "object_v_concept_5",
    # "sentiment", "squad_val", 
    "verb_v_adjective_3", "verb_v_adjective_5",
]

# score_type: "lrp" | "ie"  (head importance scoring method)
# aggregation: "averaged" | "per_task"  (shared macro-avg heads vs per-task heads)
# SCORE_TYPES  = ["lrp"]
# AGGREGATIONS = ["averaged"]
SCORE_TYPES  = ["lrp", "ie"]
# AGGREGATIONS = ["averaged"]
AGGREGATIONS = ["averaged", "per_task"]

# ── FV experiment settings ────────────────────────────────────────────────────
# Layer at which to inject the function vector into the residual stream.
# The reference used layer 14 for Llama-3.2-3B-Instruct on these tasks.
FV_EDIT_LAYER    = 20
FV_SCORE_TYPES   = ["lrp", "ie"]
# FV_AGGREGATIONS  = ["averaged"]
FV_AGGREGATIONS  = ["averaged", "per_task"]

def run_dist_experiment(score_type, aggregation, model, tokenizer, n_heads):
    if aggregation == "averaged":
        aie_attn = load_prompt_aie(INTERMEDIATE_PATH, MODEL_SHORT, TASKS, score_type)
        shared_heads, _ = select_modules(aie_attn, topk_attn=TOPK_ATTN)

    for task in tqdm(TASKS, desc=f"prompt/{score_type}/{aggregation}"):
        if aggregation == "per_task":
            try:
                task_cie = load_prompt_cie(INTERMEDIATE_PATH, MODEL_SHORT, task, score_type)
            except FileNotFoundError:
                print(f"  [warn] missing per-task IE for '{task}', skipping")
                continue
            top_heads, _ = select_modules(task_cie, topk_attn=TOPK_ATTN)
        else:
            top_heads = shared_heads

        try:
            mean_attn = load_mean_prompt_attn(INTERMEDIATE_PATH, MODEL_SHORT, task)
        except FileNotFoundError:
            print(f"  [warn] missing mean attn for '{task}', skipping")
            continue
        dist_fv, _ = build_steering_vecs(mean_attn, top_heads, n_heads)
        dist_fv = dist_fv.to(DEVICE)

        prompts = top_prompts(INTERMEDIATE_PATH, MODEL_SHORT, task, N_BEST_PROMPTS)
        filter_set = test_filter_set(INTERMEDIATE_PATH, MODEL_SHORT, task, prompts)

        for scale in SCALES:
            out = {"intervention_topk": {}, "intervention_ranks": {}}
            #for prompt in prompts:
            prompt = ""
            r = eval_prompt_steered(
                task=task, model=model, tokenizer=tokenizer,
                top_heads=top_heads, dist_fv=dist_fv,
                n_heads=n_heads, steer_scale=scale,
                prompt=prompt,
                dataset_path=DATASET_PATH, train_split=TRAIN_SPLIT, seed=SEED,
                device=DEVICE, n_tokens=N_TOKENS, filter_set=filter_set,
            )
            out["intervention_topk"][prompt] = [[k, r["topk"][k]] for k in (1, 2, 3)]
            out["intervention_ranks"][prompt] = r["ranks"]

            scores = [out["intervention_topk"][prompt][i][1] for i in range(3)]
            print(f"  {task}  scale={scale}  top1={scores[0]:.3f}  top2={scores[1]:.3f}  top3={scores[2]:.3f}")
            save_json(out, os.path.join(RESULTS_PATH, MODEL_SHORT, task, f"prompt_dist_{score_type}_{aggregation}_topk{TOPK_ATTN}_scale{scale}.json"))


def run_fv_experiment(score_type, aggregation, model, tokenizer, n_heads):
    """FV approach: sum o_proj(mean_head_act) across top heads → single vector,
    then add to residual stream at FV_EDIT_LAYER during eval."""
    if aggregation == "averaged":
        aie_attn = load_prompt_aie(INTERMEDIATE_PATH, MODEL_SHORT, TASKS, score_type)
        shared_heads, _ = select_modules(aie_attn, topk_attn=TOPK_ATTN)

    for task in tqdm(TASKS, desc=f"fv/{score_type}/{aggregation}"):
        if aggregation == "per_task":
            try:
                task_cie = load_prompt_cie(INTERMEDIATE_PATH, MODEL_SHORT, task, score_type)
            except FileNotFoundError:
                print(f"  [warn] missing per-task IE for '{task}', skipping")
                continue
            top_heads, _ = select_modules(task_cie, topk_attn=TOPK_ATTN)
        else:
            top_heads = shared_heads

        try:
            mean_attn = load_mean_prompt_attn(INTERMEDIATE_PATH, MODEL_SHORT, task)
        except FileNotFoundError:
            print(f"  [warn] missing mean attn for '{task}', skipping")
            continue
        fv = compute_fv(mean_attn, top_heads, n_heads, model)

        prompts = top_prompts(INTERMEDIATE_PATH, MODEL_SHORT, task, N_BEST_PROMPTS)
        filter_set = test_filter_set(INTERMEDIATE_PATH, MODEL_SHORT, task, prompts)

        for scale in SCALES:
            out = {"intervention_topk": {}, "intervention_ranks": {}}
            #for prompt in prompts:
            prompt = ""
            r = eval_fv_steered(
                    task=task, model=model, tokenizer=tokenizer,
                    fv=fv, edit_layer=FV_EDIT_LAYER, steer_scale=scale,
                    prompt=prompt,
                    dataset_path=DATASET_PATH, train_split=TRAIN_SPLIT, seed=SEED,
                    device=DEVICE, filter_set=filter_set,
                )
            out["intervention_topk"][prompt]  = [[k, r["topk"][k]] for k in (1, 2, 3)]
            out["intervention_ranks"][prompt] = r["ranks"]

            scores = [out["intervention_topk"][prompt][i][1] for i in range(3)]
            print(f"  {task}  scale={scale}  top1={scores[0]:.3f}  top2={scores[1]:.3f}  top3={scores[2]:.3f}")
            save_json(out, os.path.join(RESULTS_PATH, MODEL_SHORT, task, f"prompt_fv_{score_type}_{aggregation}_topk{TOPK_ATTN}_editlayer{FV_EDIT_LAYER}_scale{scale}.json"))


def main():
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, trust_remote_code=True, torch_dtype=torch.bfloat16,
    ).to(DEVICE).eval()
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    n_heads = model.config.num_attention_heads
    os.makedirs(RESULTS_PATH, exist_ok=True)

    n_dist = len(SCORE_TYPES) * len(AGGREGATIONS)
    n_fv   = len(FV_SCORE_TYPES) * len(FV_AGGREGATIONS)
    print(f"{n_dist} head-steering + {n_fv} FV experiments × {len(TASKS)} tasks × {len(SCALES)} scales\n")

    for score_type in SCORE_TYPES:
        for aggregation in AGGREGATIONS:
            print(f"\n{'=' * 60}\n  dist  score={score_type}  agg={aggregation}\n{'=' * 60}")
            run_dist_experiment(score_type, aggregation, model, tokenizer, n_heads)

    for score_type in FV_SCORE_TYPES:
        for aggregation in FV_AGGREGATIONS:
            print(f"\n{'=' * 60}\n  fv  score={score_type}  agg={aggregation}  edit_layer={FV_EDIT_LAYER}\n{'=' * 60}")
            run_fv_experiment(score_type, aggregation, model, tokenizer, n_heads)

if __name__ == "__main__":
    main()
