"""List top attention heads and MLP layers by CIE for a model/task(s).

Produces two ranked lists (0shot, 10shot) and a combined list that
averages the two shots' CIE scores before ranking.

Each entry is labelled as:
  attn  L{layer}H{head}   (attention head)
  mlp   L{layer}          (MLP layer)
"""
import os, sys
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from eval.utils import load_aie

MODEL_SHORT = "Llama-3.2-1B-Instruct"
CACHE_DIR   = "cache"
TOPK        = 20
# Set to None to average over all tasks, or list specific tasks.
TASKS = ["english-french", "english-german", "english-spanish"]

ALL_TASKS = [
    "antonym", "capitalize", "capitalize_first_letter", "country-capital",
    "country-currency", "english-french", "english-german", "english-spanish",
    "landmark-country", "lowercase_first_letter", "national_parks", "park-country",
    "person-sport", "present-past", "product-company", "sentiment",
    "singular-plural", "synonym",
]



def rank_modules(attn: torch.Tensor, mlp: torch.Tensor, topk: int, shot: str = ""):
    """Return a ranked list of (cie_value, label) for the top-k modules."""
    n_layers, n_heads = attn.shape
    prefix = f"[{shot}] " if shot else ""

    entries = []
    for l in range(n_layers):
        for h in range(n_heads):
            entries.append((attn[l, h].item(), f"{prefix}attn  L{l:02d}H{h:02d}"))
        entries.append((mlp[l].item(), f"{prefix}mlp   L{l:02d}    "))

    entries.sort(key=lambda x: x[0], reverse=True)
    return entries[:topk]


def print_ranked(label: str, entries: list):
    print(f"\n{'='*48}")
    print(f"  {label}")
    print(f"{'='*48}")
    print(f"  {'Rank':<6} {'Module':<18} {'CIE':>10}")
    print(f"  {'-'*38}")
    for rank, (val, name) in enumerate(entries, 1):
        print(f"  {rank:<6} {name:<18} {val:>10.6f}")


def main():
    tasks = TASKS or ALL_TASKS

    print(f"Model : {MODEL_SHORT}")
    print(f"Tasks : {tasks if len(tasks) <= 4 else f'{tasks[:4]} ... ({len(tasks)} total)'}")
    print(f"Top-k : {TOPK}")

    attn_0, mlp_0 = load_aie(CACHE_DIR, MODEL_SHORT, "0shot", tasks)
    attn_10, mlp_10 = load_aie(CACHE_DIR, MODEL_SHORT, "10shot", tasks)

    ranked_0  = rank_modules(attn_0,  mlp_0,  TOPK, shot="0shot")
    ranked_10 = rank_modules(attn_10, mlp_10, TOPK, shot="10shot")

    # Combined: pool all entries from both shots and re-rank
    all_entries = (
        rank_modules(attn_0,  mlp_0,  len(attn_0.flatten()) + len(mlp_0),  shot="0shot") +
        rank_modules(attn_10, mlp_10, len(attn_10.flatten()) + len(mlp_10), shot="10shot")
    )
    all_entries.sort(key=lambda x: x[0], reverse=True)
    ranked_combined = all_entries[:TOPK]

    print_ranked("0-shot", ranked_0)
    print_ranked("10-shot", ranked_10)
    print_ranked("Combined (0shot + 10shot pooled)", ranked_combined)


if __name__ == "__main__":
    main()
