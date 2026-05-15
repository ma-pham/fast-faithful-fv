"""Compare AIE-identified heads against compute_indirect_effect_v2 results."""
import os, sys
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from eval.utils import load_aie

MODEL_SHORT  = "Llama-3.2-1B-Instruct"
CACHE_DIR    = "cache"
SHOT         = "10shot"
REFERENCE_PT = f"heads_{MODEL_SHORT}.pt"
TOPK         = 10

TASKS = [
    "antonym", "capitalize", "capitalize_first_letter", "country-capital",
    "country-currency", "english-french", "english-german", "english-spanish",
    "landmark-country", "lowercase_first_letter", "national_parks", "park-country",
    "person-sport", "present-past", "product-company", "sentiment",
    "singular-plural", "synonym",
]


def main():
    mean_aie = load_aie(CACHE_DIR, MODEL_SHORT, SHOT, TASKS)[0].flatten()
    _, topk_inds = torch.topk(mean_aie, k=TOPK, largest=True)

    print(f"=== Recomputed top-{TOPK} heads (flat indices) ===")
    print(topk_inds.tolist())

    ref = torch.load(REFERENCE_PT, map_location="cpu")
    _, ref_inds = torch.topk(ref["mean_indirect_effect"], k=TOPK, largest=True)

    print(f"\n=== Reference top-{TOPK} heads (flat indices) ===")
    print(ref_inds.tolist())

    overlap = set(topk_inds.tolist()) & set(ref_inds.tolist())
    print(f"\nOverlap: {len(overlap)}/{TOPK} heads match")
    print(f"Matching indices: {sorted(overlap)}")

    # Rank correlation on the full distribution
    ref_mean = ref["mean_indirect_effect"]  # (n_heads_flat,)
    n = mean_aie.shape[0]
    assert ref_mean.shape[0] == n, f"Shape mismatch: {mean_aie.shape} vs {ref_mean.shape}"

    recomputed_ranks = torch.argsort(torch.argsort(mean_aie, descending=True))
    ref_ranks        = torch.argsort(torch.argsort(ref_mean,  descending=True))
    spearman = 1 - 6 * ((recomputed_ranks.float() - ref_ranks.float()) ** 2).sum() / (n * (n**2 - 1))
    print(f"\nSpearman rank correlation: {spearman:.4f}")


if __name__ == "__main__":
    main()
