"""
Show the top-5 instruction prompts for a task (Llama-3B) and display
the zero-shot prompt that would have been constructed for a random
correct train example.

Usage:
    python show_top_prompts.py [task] [--n N] [--seed SEED]

Examples:
    python show_top_prompts.py park-country
    python show_top_prompts.py antonym --n 3 --seed 7
"""

import argparse
import json
import random
import sys
import os
import numpy as np

# ── constants matching eval/utils.py ──────────────────────────────────────────
PREFIXES = {"input": "Q:", "output": "A:", "instructions": ""}
PROMPT_SEPARATORS = {"input": "\n", "output": "\n\n", "instructions": "\n"}

RESULTS_ROOT = "remote/storage/function_vectors/full_results_prompt_based_short"
MODEL_SHORT  = "Llama-3.2-3B-Instruct"
DATA_DIR     = "remote/storage/function_vectors/dataset_files"


def load_per_prompt_results(task):
    path = os.path.join(RESULTS_ROOT, MODEL_SHORT, task, "per_prompt_results.json")
    with open(path) as f:
        return json.load(f)


def top_n_prompts(results, n=5):
    """Sort prompts by train top-1 accuracy, return top-n."""
    train_topk = results["train"]["clean_topk"]
    return sorted(train_topk, key=lambda p: train_topk[p][0][1], reverse=True)[:n]


def train_filter_set(results, prompts):
    """Indices where ALL selected prompts answer correctly (rank == 0)."""
    rank_lists = results["train"]["clean_rank_list"]
    summed = np.sum([np.array(rank_lists[p]) for p in prompts], axis=0)
    return np.where(summed == 0)[0]


def build_prompt(instruction, input_text):
    """Reconstruct the zero-shot prompt used in eval_prompt_steered."""
    # mirrors word_pairs_to_prompt_data + create_prompt with empty examples
    # prefixes['instructions']="" + instruction + separators['instructions']="\n"
    # then Q: {input}\n then A:
    return (
        PREFIXES["instructions"]
        + instruction
        + PROMPT_SEPARATORS["instructions"]
        + PREFIXES["input"]
        + " " + input_text       # prepend_space=True (default)
        + PROMPT_SEPARATORS["input"]
        + PREFIXES["output"]
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("task", nargs="?", default="concept_v_object_5")
    parser.add_argument("--n", type=int, default=5, help="number of top prompts")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # ── load results & pick top prompts ───────────────────────────────────────
    results = load_per_prompt_results(args.task)
    prompts = top_n_prompts(results, n=args.n)
    train_topk = results["train"]["clean_topk"]

    print(f"Task: {args.task}  |  Model: {MODEL_SHORT}")
    print(f"Top-{args.n} prompts by train top-1 accuracy:")
    for i, p in enumerate(prompts, 1):
        acc1 = train_topk[p][0][1]
        acc3 = train_topk[p][2][1]
        print(f"  {i}. [{acc1:.3f} / {acc3:.3f} top-1/3]  \"{p}\"")

    # ── find a correct train example ──────────────────────────────────────────
    filter_idx = train_filter_set(results, prompts)
    print(f"\nTrain filter set size: {len(filter_idx)} / {len(results['train']['clean_rank_list'][prompts[0]])}")

    rng = random.Random(args.seed)
    chosen_idx = int(rng.choice(filter_idx))

    # load dataset (2-way split to match reference indices)
    sys.path.insert(0, os.path.dirname(__file__))
    from prompt_utils import load_dataset
    dataset = load_dataset(args.task, root_data_dir=DATA_DIR, test_size=0.3, seed=3, split_valid=False)
    example = dataset["train"][chosen_idx]

    inp  = example["input"]
    out  = example["output"]
    print(f"\nSampled train example (index {chosen_idx}):  input={inp!r}  target={out!r}")

    # ── show the reconstructed prompt for each top instruction ─────────────────
    print("\n" + "=" * 70)
    for i, instruction in enumerate(prompts, 1):
        prompt_str = build_prompt(instruction, inp)
        print(f"\n--- Prompt #{i} ---")
        print(repr(prompt_str))
        print()
        print(prompt_str + f" {out}")   # show with answer appended
        print("=" * 70)


if __name__ == "__main__":
    main()
