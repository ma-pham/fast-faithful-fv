"""Print top-26 joint modules by AIE and their attn/MLP breakdown."""
import os, sys
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from eval.utils import load_aie, select_modules

MODEL_SHORT = "Llama-3.2-1B-Instruct"
CACHE_DIR   = "./cache"
SHOT        = "10shot"
TOPK_JOINT  = 26

TASKS = [
    "antonym", "capitalize", "capitalize_first_letter", "country-capital",
    "country-currency", "english-french", "english-german", "english-spanish",
    "landmark-country", "lowercase_first_letter", "national_parks", "park-country",
    "person-sport", "present-past", "product-company", "sentiment",
    "singular-plural", "synonym",
]


def main():
    attn_cie, mlp_cie = load_aie(CACHE_DIR, MODEL_SHORT, SHOT, TASKS)
    n_layers, n_heads = attn_cie.shape

    # Rebuild ranked joint list with scores
    entries = []
    for l in range(n_layers):
        for h in range(n_heads):
            entries.append(("attn", l, h, attn_cie[l, h].item()))
        entries.append(("mlp", l, -1, mlp_cie[l].item()))
    entries.sort(key=lambda x: x[3], reverse=True)
    top = entries[:TOPK_JOINT]

    print(f"Top-{TOPK_JOINT} joint modules ({SHOT} CIE, macro-avg over {len(TASKS)} tasks)\n")
    print(f"  {'Rank':<5} {'Type':<6} {'Location':<12} {'AIE':>10}")
    print(f"  {'-'*37}")
    for rank, (kind, l, h, val) in enumerate(top, 1):
        loc = f"L{l:02d}H{h:02d}" if kind == "attn" else f"L{l:02d}"
        print(f"  {rank:<5} {kind:<6} {loc:<12} {val:>10.6f}")

    n_attn = sum(1 for kind, *_ in top if kind == "attn")
    n_mlp  = TOPK_JOINT - n_attn
    print(f"\n  Attn heads : {n_attn}")
    print(f"  MLP layers : {n_mlp}")


if __name__ == "__main__":
    main()
