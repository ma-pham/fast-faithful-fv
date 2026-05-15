"""Prompt-based steering eval using pre-computed universal top heads from storage."""
import json, os, sys
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from eval.utils import (
    load_mean_prompt_attn,
    build_steering_vecs, eval_prompt_steered,
    compute_fv, eval_fv_steered,
    top_prompts, test_filter_set,
    save_json
)

# STORAGE_ROOT = os.environ.get("STORAGE_ROOT")
STORAGE_ROOT = "storage"
INTERMEDIATE_PATH = os.path.join(STORAGE_ROOT, "function_vectors", "full_results_prompt_based_short")
DATASET_PATH      = os.path.join(STORAGE_ROOT, "function_vectors", "dataset_files")
# RESULTS_PATH      = os.path.join(STORAGE_ROOT, "eval")
RESULTS_PATH      = os.path.join("storage", "eval")
TOP_HEADS_PATH    = os.path.join(STORAGE_ROOT, "function_vectors", "full_results_top_heads")

MODEL       = "meta-llama/Llama-3.2-3B-Instruct"
#MODEL = "meta-llama/Llama-3.1-8B-Instruct"
#MODEL = "Qwen/Qwen3-4B-Instruct-2507"

MODEL_SHORT = MODEL.split("/")[-1]
DEVICE      = "cuda:0"

TRAIN_SPLIT    = 0.7
SEED           = 42

N_BEST_PROMPTS = 5

#SCALES = [0.5, 1.5, 2.0, 2.5, 3.0]
SCALES = [1.0]

N_TOKENS = 1

# How many heads to take from the loaded universal list.
# Reference default is 20; paper notebooks sweep (10, 20, 40).
N_UNIVERSAL_HEADS = 20

# Which pre-computed heads file to load. Available suffixes include:
#   "both_all_top_heads", "icl_top_heads", "universal_all_min_abs_heads"
HEAD_FILE_KEY = "both_all_top_heads"


TASKS = [
    # abstractive
    "antonym", "capitalize", "capitalize_first_letter", "capitalize_last_letter",
    "capitalize_second_letter",
    "country-capital", "country-currency", "english-french", "english-german", "english-spanish",
    "landmark-country", "lowercase_first_letter", "lowercase_last_letter", "national_parks",
    "next_capital_letter", "next_item", "park-country", "present-past", "prev_item",
    "product-company", "singular-plural", "synonym", "word_length",
    # extractive
    "adjective_v_verb_3", "adjective_v_verb_5",
    "alphabetically_last_3",
    "animal_v_object_3", "animal_v_object_5",
    "choose_first_of_3", "choose_first_of_5",
    "choose_last_of_3", "choose_last_of_5",
    "choose_middle_of_3", "choose_middle_of_5",
    "color_v_animal_3", "color_v_animal_5",
    "concept_v_object_3", "concept_v_object_5",
    "conll2003_location", "conll2003_organization", "conll2003_person",
    "fruit_v_animal_3", "fruit_v_animal_5",
    "object_v_concept_3", "object_v_concept_5",
    "verb_v_adjective_3", "verb_v_adjective_5",
]

# ── FV experiment settings ────────────────────────────────────────────────────
# Reference: layer 11 for Llama-3.2-3B-Instruct, layer 14 for Llama-3.1-8B-Instruct
FV_EDIT_LAYER_LIST = [14]
#FV_EDIT_LAYER_LIST = [14]  # 8B
#FV_EDIT_LAYER_LIST = [1,8,9,10,11,12,13,14,15,16]


def load_universal_heads(model_short: str, key: str, n: int) -> list:
    path = os.path.join(TOP_HEADS_PATH, f"{model_short}_{key}.json")
    with open(path) as f:
        data = json.load(f)
    return [(int(L), int(H)) for L, H in data["top_heads"][:n]]


def run_dist_experiment(top_heads, model, tokenizer, n_heads_model):
    for task in tqdm(TASKS, desc="prompt/universal"):
        try:
            mean_attn = load_mean_prompt_attn(INTERMEDIATE_PATH, MODEL_SHORT, task)
        except FileNotFoundError:
            print(f"  [warn] missing mean attn for '{task}', skipping")
            continue
        dist_fv, _ = build_steering_vecs(mean_attn, top_heads, n_heads_model)
        dist_fv = dist_fv.to(DEVICE)

        prompts    = top_prompts(INTERMEDIATE_PATH, MODEL_SHORT, task, N_BEST_PROMPTS)
        filter_set = test_filter_set(INTERMEDIATE_PATH, MODEL_SHORT, task, prompts)

        for scale in SCALES:
            out = {"intervention_topk": {}, "intervention_ranks": {}}
            prompt = ""
            r = eval_prompt_steered(
                task=task, model=model, tokenizer=tokenizer,
                top_heads=top_heads, dist_fv=dist_fv,
                n_heads=n_heads_model, steer_scale=scale,
                prompt=prompt,
                dataset_path=DATASET_PATH, train_split=TRAIN_SPLIT, seed=SEED,
                device=DEVICE, n_tokens=N_TOKENS, filter_set=filter_set,
            )
            out["intervention_topk"][prompt] = [[k, r["topk"][k]] for k in (1, 2, 3)]
            out["intervention_ranks"][prompt] = r["ranks"]

            scores = [out["intervention_topk"][prompt][i][1] for i in range(3)]
            print(f"  {task}  scale={scale}  top1={scores[0]:.3f}  top2={scores[1]:.3f}  top3={scores[2]:.3f}")
            fname = f"prompt_dist_universal_{HEAD_FILE_KEY}_heads{N_UNIVERSAL_HEADS}_scale{scale}.json"
            save_json(out, os.path.join(RESULTS_PATH, MODEL_SHORT, task, fname))


def run_fv_experiment(top_heads, model, tokenizer, n_heads_model):
    for task in tqdm(TASKS, desc="fv/universal"):
        try:
            mean_attn = load_mean_prompt_attn(INTERMEDIATE_PATH, MODEL_SHORT, task)
        except FileNotFoundError:
            print(f"  [warn] missing mean attn for '{task}', skipping")
            continue
        fv = compute_fv(mean_attn, top_heads, n_heads_model, model)

        prompts    = top_prompts(INTERMEDIATE_PATH, MODEL_SHORT, task, N_BEST_PROMPTS)
        filter_set = test_filter_set(INTERMEDIATE_PATH, MODEL_SHORT, task, prompts)

        for scale in SCALES:
            for FV_EDIT_LAYER in FV_EDIT_LAYER_LIST:
                out = {"intervention_topk": {}, "intervention_ranks": {}}
                prompt = ""
                r = eval_fv_steered(
                    task=task, model=model, tokenizer=tokenizer,
                    fv=fv, edit_layer=FV_EDIT_LAYER, steer_scale=scale,
                    prompt=prompt,
                    dataset_path=DATASET_PATH, train_split=TRAIN_SPLIT, seed=SEED,
                    device=DEVICE, filter_set=filter_set,
                )
                out["intervention_topk"][prompt] = [[k, r["topk"][k]] for k in (1, 2, 3)]
                out["intervention_ranks"][prompt] = r["ranks"]

                scores = [out["intervention_topk"][prompt][i][1] for i in range(3)]
                print(f"  {task}  scale={scale}  top1={scores[0]:.3f}  top2={scores[1]:.3f}  top3={scores[2]:.3f}")
                fname = f"prompt_fv_universal_{HEAD_FILE_KEY}_heads{N_UNIVERSAL_HEADS}_editlayer{FV_EDIT_LAYER}_scale{scale}.json"
                save_json(out, os.path.join(RESULTS_PATH, MODEL_SHORT, task, fname))


def main():
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, trust_remote_code=True, torch_dtype=torch.bfloat16,
    ).to(DEVICE).eval()
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    n_heads   = model.config.num_attention_heads
    num_layers = model.config.num_hidden_layers
    head_dim  = getattr(model.config, "head_dim", model.config.hidden_size // n_heads)
    print(f"Model Layers: {num_layers} | Heads: {n_heads} | Head Dim: {head_dim}")

    top_heads = load_universal_heads(MODEL_SHORT, HEAD_FILE_KEY, N_UNIVERSAL_HEADS)
    print(f"Loaded {len(top_heads)} universal heads from {HEAD_FILE_KEY}: {top_heads}")

    os.makedirs(RESULTS_PATH, exist_ok=True)

    print(f"\n{'=' * 60}\n  dist  universal heads={N_UNIVERSAL_HEADS}  key={HEAD_FILE_KEY}\n{'=' * 60}")
    run_dist_experiment(top_heads, model, tokenizer, n_heads)

    print(f"\n{'=' * 60}\n  fv  universal heads={N_UNIVERSAL_HEADS}  key={HEAD_FILE_KEY}  edit_layers={FV_EDIT_LAYER_LIST}\n{'=' * 60}")
    run_fv_experiment(top_heads, model, tokenizer, n_heads)


if __name__ == "__main__":
    main()
