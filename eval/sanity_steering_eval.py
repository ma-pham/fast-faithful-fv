"""Sanity steering eval: 0-shot, whole test set, all tasks, attn-only.

Head selection: top-20 heads by macro-averaged AIE from 10-shot CIE.
Same heads used for every task; steering vectors come from each task's
own mean activations.
"""
import os, sys
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from eval.utils import load_aie, load_mean_attn, load_mean_mlp, select_modules, build_steering_vecs, eval_steered

MODEL       = "meta-llama/Llama-3.2-1B-Instruct"
MODEL_SHORT = MODEL.split("/")[-1]
CACHE_DIR   = "./cache"
TOPK_HEADS  = 20
SCALE       = 1.0
DEVICE      = "cuda:1"

DATASET_PATH = "./dataset_files_fv"
TRAIN_SPLIT  = 0.7
SEED         = 42

TASKS = [
    "antonym", "capitalize", "capitalize_first_letter", "country-capital",
    "country-currency", "english-french", "english-german", "english-spanish",
    "landmark-country", "lowercase_first_letter", "national_parks", "park-country",
    "person-sport", "present-past", "product-company", "sentiment",
    "singular-plural", "synonym",
]


def main():
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, trust_remote_code=True, torch_dtype=torch.bfloat16,
    ).to(DEVICE).eval()
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    n_heads = model.config.num_attention_heads

    # Select top-k heads by macro-averaged 10-shot AIE — same for all tasks
    aie_attn, aie_mlp = load_aie(CACHE_DIR, MODEL_SHORT, "10shot", TASKS)
    top_heads, _ = select_modules(aie_attn, aie_mlp, "attn", topk_attn=TOPK_HEADS)
    print(f"Top-{TOPK_HEADS} heads: {top_heads}\n")

    results = {}
    for task in TASKS:
        mean_attn = load_mean_attn(CACHE_DIR, MODEL_SHORT, task)
        mean_mlp  = load_mean_mlp(CACHE_DIR, MODEL_SHORT, task)
        dist_fv, dist_fv_mlp = build_steering_vecs(mean_attn, top_heads, n_heads, mean_mlp=mean_mlp)
        dist_fv = dist_fv.to(DEVICE)

        result = eval_steered(
            task=task, model=model, tokenizer=tokenizer,
            top_heads=top_heads, top_mlp_layers=[],
            dist_fv=dist_fv, dist_fv_mlp=dist_fv_mlp,
            n_heads=n_heads, steer_scale=SCALE,
            dataset_path=DATASET_PATH, train_split=TRAIN_SPLIT, seed=SEED,
            device=DEVICE, n_shots=0,
        )
        results[task] = result
        print(f"{task:30s}  top1={result['steered_topk'][1]:.3f}  top3={result['steered_topk'][3]:.3f}  n={result['n_examples']}")

    avg1 = sum(r["steered_topk"][1] for r in results.values()) / len(results)
    avg3 = sum(r["steered_topk"][3] for r in results.values()) / len(results)
    print(f"\nMacro avg  top1={avg1:.3f}  top3={avg3:.3f}")

    os.makedirs(f"results/{MODEL_SHORT}", exist_ok=True)
    save_path = f"results/{MODEL_SHORT}/sanity_0shot_attn_top{TOPK_HEADS}_scale{SCALE}.pt"
    torch.save({"top_heads": top_heads, "scale": SCALE, "results": results}, save_path)
    print(f"Saved to {save_path}")


if __name__ == "__main__":
    main()
