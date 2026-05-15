"""FV eval with top-20 heads from macro-averaged IE across all tasks.

Computes a single shared set of top-20 attention heads by averaging the
indirect-effect tensors across ALL_TASKS, then evaluates each task in TASKS
using an FV built from those shared heads + the task's own mean activations.

Output files: {RESULTS_PATH}/{MODEL_SHORT}/{task}/prompt_fv_ie_averaged_editlayer{L}_scale{s}.json
(same naming convention as run_eval_simple.py so print_3b_accuracies.py picks them up)
"""
import os, sys
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from eval.utils import (
    load_prompt_aie,
    select_modules, compute_fv, eval_fv_steered,
    top_prompts, test_filter_set,
    save_json,
)

STORAGE_ROOT    = os.environ.get("STORAGE_ROOT")
INTERMEDIATE_PATH = os.path.join(STORAGE_ROOT, "function_vectors", "full_results_prompt_based_short")
DATASET_PATH    = os.path.join(STORAGE_ROOT, "function_vectors", "dataset_files")
RESULTS_PATH    = os.path.join(STORAGE_ROOT, "eval")

MODEL       = "meta-llama/Llama-3.1-8B-Instruct"
MODEL_SHORT = MODEL.split("/")[-1]
DEVICE      = "cuda:0"

TRAIN_SPLIT    = 0.7
SEED           = 42
N_BEST_PROMPTS = 5

TOPK_ATTN  = 20
SCALES     = [1.0]
FV_EDIT_LAYER = 14

# Full task list used to compute the macro-averaged IE (AIE).
# Add or remove tasks here to change which tasks inform the shared head set.
ALL_TASKS = [
    "adjective_v_verb_3", "adjective_v_verb_5",
    "alphabetically_first_3", "alphabetically_first_5",
    "alphabetically_last_3", "alphabetically_last_5",
    "animal_v_object_3", "animal_v_object_5",
    "antonym",
    "capitalize", "capitalize_first_letter", "capitalize_last_letter", "capitalize_second_letter",
    "choose_first_of_3", "choose_first_of_5",
    "choose_last_of_3", "choose_last_of_5",
    "choose_middle_of_3", "choose_middle_of_5",
    "color_v_animal_3", "color_v_animal_5",
    "commonsense_qa",
    "concept_v_object_3", "concept_v_object_5",
    "conll2003_location", "conll2003_organization", "conll2003_person",
    "country-capital", "country-currency",
    "english-french", "english-german", "english-spanish",
    "fruit_v_animal_3", "fruit_v_animal_5",
    "landmark-country",
    "lowercase_first_letter", "lowercase_last_letter",
    "national_parks",
    "next_capital_letter", "next_item",
    "object_v_concept_3", "object_v_concept_5",
    "park-country",
    "person-instrument", "person-occupation", "person-sport",
    "present-past", "prev_item",
    "product-company",
    "sentiment",
    "singular-plural",
    "synonym",
    "verb_v_adjective_3", "verb_v_adjective_5",
    "word_length",
]

# Tasks to actually evaluate (subset or same as ALL_TASKS).
TASKS = ALL_TASKS


def main():
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, trust_remote_code=True, torch_dtype=torch.bfloat16,
    ).to(DEVICE).eval()
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    n_heads = model.config.num_attention_heads

    # ── Step 1: compute shared top-20 heads from macro-averaged IE ────────────
    print(f"Computing macro-averaged IE over {len(ALL_TASKS)} tasks...")
    aie_attn = load_prompt_aie(INTERMEDIATE_PATH, MODEL_SHORT, ALL_TASKS, selection="ie")
    shared_heads, _ = select_modules(aie_attn, topk_attn=TOPK_ATTN)
    print(f"Shared top-{TOPK_ATTN} heads: {shared_heads[:5]} ... (showing first 5)")

    # ── Step 2: evaluate each task with the shared heads ─────────────────────
    for task in tqdm(TASKS, desc="tasks"):
        from eval.utils import load_mean_prompt_attn
        try:
            mean_attn = load_mean_prompt_attn(INTERMEDIATE_PATH, MODEL_SHORT, task)
        except FileNotFoundError:
            print(f"  [warn] missing mean activations for '{task}', skipping")
            continue

        fv = compute_fv(mean_attn, shared_heads, n_heads, model)

        prompts    = top_prompts(INTERMEDIATE_PATH, MODEL_SHORT, task, N_BEST_PROMPTS)
        filter_set = test_filter_set(INTERMEDIATE_PATH, MODEL_SHORT, task, prompts)

        for scale in SCALES:
            out = {"intervention_topk": {}, "intervention_ranks": {}}
            r = eval_fv_steered(
                task=task, model=model, tokenizer=tokenizer,
                fv=fv, edit_layer=FV_EDIT_LAYER, steer_scale=scale,
                prompt="",
                dataset_path=DATASET_PATH, train_split=TRAIN_SPLIT, seed=SEED,
                device=DEVICE, filter_set=filter_set,
            )
            out["intervention_topk"][""] = [[k, r["topk"][k]] for k in (1, 2, 3)]
            out["intervention_ranks"][""] = r["ranks"]

            scores = [out["intervention_topk"][""][i][1] for i in range(3)]
            print(f"  {task}  scale={scale}  top1={scores[0]:.3f}  top2={scores[1]:.3f}  top3={scores[2]:.3f}")
            save_json(out, os.path.join(
                RESULTS_PATH, MODEL_SHORT, task,
                f"prompt_fv_ie_averaged_editlayer{FV_EDIT_LAYER}_scale{scale}.json",
            ))


if __name__ == "__main__":
    main()
