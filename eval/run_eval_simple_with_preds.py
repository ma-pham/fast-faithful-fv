"""Same as run_eval_simple.py but stores per-example predictions in the JSON output."""
import os, sys
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from eval.utils import (
    load_mean_prompt_attn, load_prompt_aie, load_prompt_cie,
    select_modules, build_steering_vecs,
    compute_fv,
    top_prompts, test_filter_set,
    save_json,
    PREFIXES, PROMPT_SEPARATORS,
)
from prompt_utils import (
    load_dataset,
    word_pairs_to_prompt_data, create_prompt,
    get_answer_id, rank_of_token, topk_acc,
)
from utils_v2 import split_activations_by_head

# STORAGE_ROOT = os.environ.get("STORAGE_ROOT")
STORAGE_ROOT = "remote/storage"
INTERMEDIATE_PATH = os.path.join(STORAGE_ROOT, "function_vectors", "full_results_prompt_based_short")
DATASET_PATH = os.path.join(STORAGE_ROOT, "function_vectors", "dataset_files")
RESULTS_PATH =  os.path.join(STORAGE_ROOT, "eval")

MODEL  = "meta-llama/Llama-3.2-3B-Instruct"
#MODEL = "Qwen/Qwen3-4B-Instruct-2507"
#MODEL = "meta-llama/Llama-3.1-8B-Instruct"

MODEL_SHORT = MODEL.split("/")[-1]
DEVICE      = "cuda:0"

TRAIN_SPLIT    = 0.7
SEED           = 42

N_BEST_PROMPTS = 5

SCALES = [1.0]

N_TOKENS = 1

TOPK_ATTN = 20


# Full task list — required for averaged IE score calculations (load_prompt_aie).
ALL_TASKS = [
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

# Tasks to actually run inference on.
RUN_TASKS = [
    "antonym",
]

SCORE_TYPES   = ["lrp"]
AGGREGATIONS = ["averaged"]

# FV_SCORE_TYPES   = ["lrp", "ie"]
# FV_AGGREGATIONS = ["averaged"]

FV_SCORE_TYPES = []
FV_AGGREGATIONS = []

# Layer at which to inject the function vector into the residual stream.
# Reference uses layer 11 for Llama-3.2-3B-Instruct, 14 for 8B.
FV_EDIT_LAYER = 11


def _collect_predictions(logits, target_token_id, target, word_pair, tokenizer):
    """Return (rank, prediction_dict) for one example."""
    rank = int(rank_of_token(logits, target_token_id))
    predicted_id = int(logits.argmax(dim=-1).item())
    predicted_str = tokenizer.decode([predicted_id], skip_special_tokens=True)
    return rank, {
        "input":     word_pair.get("input", ""),
        "target":    target,
        "predicted": predicted_str,
        "rank":      rank,
    }


def run_dist_experiment(score_type, aggregation, model, tokenizer, n_heads):
    if aggregation == "averaged":
        aie_attn = load_prompt_aie(INTERMEDIATE_PATH, MODEL_SHORT, ALL_TASKS, score_type)
        shared_heads, _ = select_modules(aie_attn, topk_attn=TOPK_ATTN)

    for task in tqdm(RUN_TASKS, desc=f"prompt/{score_type}/{aggregation}"):
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

        dataset = load_dataset(task, root_data_dir=DATASET_PATH, test_size=1 - TRAIN_SPLIT, seed=SEED,
                               split_valid=(filter_set is None))

        heads_by_layer = {}
        for gid, (L, H) in enumerate(top_heads):
            heads_by_layer.setdefault(L, []).append((gid, H))

        for scale in SCALES:
            out = {"intervention_topk": {}, "intervention_ranks": {}, "predictions": {}}
            prompt = ""

            hooks = []
            for L, head_list in heads_by_layer.items():
                def attn_hook(_module, inputs, _head_list=head_list):
                    x = inputs[0]
                    B, S, D = x.shape
                    x_heads = split_activations_by_head(x, n_heads=n_heads)
                    for gid, H in _head_list:
                        for t in range(N_TOKENS):
                            pos = -(N_TOKENS - t)
                            x_heads[:, pos, H] = x_heads[:, pos, H] + scale * dist_fv[gid, t].to(x.device)
                    return (x_heads.view(B, S, D),)
                hooks.append(model.model.layers[L].self_attn.o_proj.register_forward_pre_hook(attn_hook, with_kwargs=False))

            indices = filter_set if filter_set is not None else range(len(dataset["test"]))
            ranks = []
            preds = []

            for j in tqdm(indices, desc=f"{task} scale={scale}", leave=False):
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

                toks = tokenizer(sentence, return_tensors="pt").to(DEVICE)
                logits = model(input_ids=toks["input_ids"]).logits[:, -1, :].float()

                rank, pred = _collect_predictions(logits, target_token_id, target, word_pair_test, tokenizer)
                ranks.append(rank)
                preds.append(pred)

            for h in hooks:
                h.remove()

            out["intervention_topk"][prompt] = [[k, topk_acc(ranks, k)] for k in (1, 2, 3)]
            out["intervention_ranks"][prompt] = ranks
            out["predictions"][prompt] = preds

            scores = [out["intervention_topk"][prompt][i][1] for i in range(3)]
            print(f"  {task}  scale={scale}  top1={scores[0]:.3f}  top2={scores[1]:.3f}  top3={scores[2]:.3f}")
            save_json(out, os.path.join(RESULTS_PATH, MODEL_SHORT, task, f"prompt_dist_{score_type}_{aggregation}_scale{scale}_preds.json"))


@torch.inference_mode()
def run_fv_experiment(score_type, aggregation, model, tokenizer, n_heads):
    """FV approach with per-example predictions stored in JSON."""
    if aggregation == "averaged":
        aie_attn = load_prompt_aie(INTERMEDIATE_PATH, MODEL_SHORT, ALL_TASKS, score_type)
        shared_heads, _ = select_modules(aie_attn, topk_attn=TOPK_ATTN)

    for task in tqdm(RUN_TASKS, desc=f"fv/{score_type}/{aggregation}"):
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

        dataset = load_dataset(task, root_data_dir=DATASET_PATH, test_size=1 - TRAIN_SPLIT, seed=SEED,
                               split_valid=(filter_set is None))

        for scale in SCALES:
            fv_scaled = (scale * fv).to(DEVICE)

            def _fv_hook(_module, _inp, out):
                if isinstance(out, tuple):
                    hidden = out[0].clone()
                    hidden[:, -1, :] = hidden[:, -1, :] + fv_scaled
                    return (hidden,) + out[1:]
                out = out.clone()
                out[:, -1, :] = out[:, -1, :] + fv_scaled
                return out

            hook = model.model.layers[FV_EDIT_LAYER].register_forward_hook(_fv_hook)

            indices = filter_set if filter_set is not None else range(len(dataset["test"]))
            ranks = []
            preds = []

            for j in tqdm(indices, desc=f"{task} scale={scale}", leave=False):
                word_pair_test = dataset["test"][int(j)]
                prompt = ""
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

                toks = tokenizer(sentence, return_tensors="pt").to(DEVICE)
                logits = model(input_ids=toks["input_ids"]).logits[:, -1, :].float()

                rank, pred = _collect_predictions(logits, target_token_id, target, word_pair_test, tokenizer)
                ranks.append(rank)
                preds.append(pred)

            hook.remove()

            prompt = ""
            out = {
                "intervention_topk": {prompt: [[k, topk_acc(ranks, k)] for k in (1, 2, 3)]},
                "intervention_ranks": {prompt: ranks},
                "predictions":        {prompt: preds},
            }

            scores = [out["intervention_topk"][prompt][i][1] for i in range(3)]
            print(f"  {task}  scale={scale}  top1={scores[0]:.3f}  top2={scores[1]:.3f}  top3={scores[2]:.3f}")
            save_json(out, os.path.join(RESULTS_PATH, MODEL_SHORT, task, f"prompt_fv_{score_type}_{aggregation}_editlayer{FV_EDIT_LAYER}_scale{scale}_preds.json"))


def main():
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, trust_remote_code=True, torch_dtype=torch.bfloat16,
    ).to(DEVICE).eval()
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    n_heads = model.config.num_attention_heads
    num_layers = model.config.num_hidden_layers
    head_dim = getattr(model.config, "head_dim", model.config.hidden_size // n_heads)

    print(f"Model Layers: {num_layers} | Heads: {n_heads} | Head Dim: {head_dim}")

    os.makedirs(RESULTS_PATH, exist_ok=True)

    n_dist = len(SCORE_TYPES) * len(AGGREGATIONS)
    n_fv   = len(FV_SCORE_TYPES) * len(FV_AGGREGATIONS)
    print(f"{n_dist} head-steering + {n_fv} FV experiments × {len(RUN_TASKS)} tasks × {len(SCALES)} scales\n")

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
