"""10-shot unsteered ICL baseline over the full test set for every task."""
import os, sys
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from eval.utils import eval_steered

MODEL       = "meta-llama/Llama-3.1-8B-Instruct"
MODEL_SHORT = MODEL.split("/")[-1]
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
    dummy   = torch.zeros(0, device=DEVICE)

    results = {}
    for task in TASKS:
        r = eval_steered(
            task=task, model=model, tokenizer=tokenizer,
            top_heads=[], top_mlp_layers=[],
            dist_fv=dummy, dist_fv_mlp=dummy,
            n_heads=n_heads, steer_scale=0.0,
            dataset_path=DATASET_PATH, train_split=TRAIN_SPLIT, seed=SEED,
            device=DEVICE, n_shots=10,
        )
        results[task] = r["steered_topk"]

    avg = {K: sum(results[t][K] for t in TASKS) / len(TASKS) for K in (1, 2, 3)}

    print(f"\n{'Task':<30}  {'top-1':>6}  {'top-2':>6}  {'top-3':>6}")
    print("-" * 52)
    for task in TASKS:
        m = results[task]
        print(f"{task:<30}  {m[1]:>6.3f}  {m[2]:>6.3f}  {m[3]:>6.3f}")
    print("-" * 52)
    print(f"{'MACRO AVG':<30}  {avg[1]:>6.3f}  {avg[2]:>6.3f}  {avg[3]:>6.3f}")

    os.makedirs(f"results/{MODEL_SHORT}", exist_ok=True)
    save_path = f"results/{MODEL_SHORT}/icl_baseline_10shot.pt"
    torch.save({"type": "baseline", "n_shots": 10, "tasks": TASKS, "results": results, "avg": avg}, save_path)
    print(f"\nSaved to {save_path}")


if __name__ == "__main__":
    main()
