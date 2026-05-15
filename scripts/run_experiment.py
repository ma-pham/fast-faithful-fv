"""One experiment run: mean activations -> CIE -> steering sweep -> save."""
import argparse, os, yaml, torch

from src.models.loader import load_model
from src.vectors.mean_activations import compute_mean_activations
from cie import get_or_compute_cie, compute_aie_heads
from src.eval.runner import eval_with_intervention
from src.steering.intervention import Intervention, InterventionSet
from src.io.results import save_results, append_records

CONFIG = {
    "experiment_id": "antonym_llama32_1b_v1",
    "model": "meta-llama/Llama-3.2-1B-Instruct",
    "task": "antonym",

    "dataset_path": "./dataset_files",
    "train_split": 0.7,
    "seed": 42,
    "device": "cuda",
    "dtype": torch.bfloat16,

    "prefixes": {
        "input": "Q:",
        "output": "A:",
        "instructions": "",
    },
    "separators": {
        "input": "\n",
        "output": "\n\n",
        "instructions": "",
    },

    "mean_activations": {
        "n_shots": 10,
        "n_trials": 100,
    },

    "cie": {
        "n_shots_corrupted": 10,   # set to 0 for zero-shot CIE scheme
        "n_trials": 25,
        "topk_heads": 10,
    },

    "steering_sweep": {
        "scales": [0.0, 0.5, 1.0, 2.0, 4.0, 8.0],
        "n_shots_eval": 0,
        "max_examples": 200,
        "mlp_layers": [4, 8, 12],
    },

    "paths": {
        "vectors_dir": "./cache/vectors",
        "results_dir": "./results",
    },
}



def _dtype(name: str):
    return {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}[name]


def main(cfg_path: str):
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    model, tokenizer = load_model(cfg["model"], device=cfg["device"], dtype=_dtype(cfg["dtype"]))

    vectors_dir = cfg["paths"]["vectors_dir"]
    results_dir = cfg["paths"]["results_dir"]
    exp_id = cfg["experiment_id"]
    task = cfg["task"]

    # 1. Mean activations -------------------------------------------------
    mean_path = os.path.join(
    vectors_dir, exp_id,
    f"mean_acts_{task}_{cfg['mean_activations']['n_shots']}shot.pt",)
    if os.path.exists(mean_path):
        mean = torch.load(mean_path)
    else:
        mean = compute_mean_activations(
            model, tokenizer, task=task,
            n_shots=cfg["mean_activations"]["n_shots"],
            n_trials=cfg["mean_activations"]["n_trials"],
            prefixes=cfg["prefixes"], separators=cfg["separators"],
            dataset_path=cfg["dataset_path"], device=cfg["device"],
            train_split=cfg["train_split"], seed=cfg["seed"],
            save_path=mean_path,
        )

    # 2. CIE --------------------------------------------------------------
    cache_dir = os.path.join(vectors_dir, exp_id)
    cie = get_or_compute_cie(
        model, tokenizer, task=task,
        mean_head_acts=mean["heads"].to(cfg["device"]),
        n_shots_corrupted=cfg["cie"]["n_shots_corrupted"],
        n_trials=cfg["cie"]["n_trials"],
        prefixes=cfg["prefixes"], separators=cfg["separators"],
        dataset_path=cfg["dataset_path"], device=cfg["device"],
        cache_dir=cache_dir,
        train_split=cfg["train_split"], seed=cfg["seed"],)

    # 3. Pick top-k heads by CIE, plus the MLP layers from config --------
    if cfg["cie"]["selection"] == "task_cie":
        score = cie  # already per-task
    elif cfg["cie"]["selection"] == "aie":
        # Compute mean activations for each task, then AIE across them
        mean_acts_per_task = {}
        for t in cfg["cie"]["aie_tasks"]:
            p = os.path.join(vectors_dir, exp_id, f"mean_acts_{cfg['mean_activations']['n_shots']}shot_{t}.pt")
            if os.path.exists(p):
                mean_acts_per_task[t] = torch.load(p)["heads"]
            else:
                m = compute_mean_activations(
                    model, tokenizer, task=t,
                    n_shots=cfg["mean_activations"]["n_shots"],
                    n_trials=cfg["mean_activations"]["n_trials"],
                    prefixes=cfg["prefixes"], separators=cfg["separators"],
                    dataset_path=cfg["dataset_path"], device=cfg["device"],
                    train_split=cfg["train_split"], seed=cfg["seed"],
                    save_path=p,
                )
                mean_acts_per_task[t] = m["heads"]

        score = compute_aie_heads(
            model, tokenizer, tasks=cfg["cie"]["aie_tasks"],
            mean_head_acts_per_task=mean_acts_per_task,
            n_shots_corrupted=cfg["cie"]["n_shots_corrupted"],
            n_trials=cfg["cie"]["n_trials"],
            prefixes=cfg["prefixes"], separators=cfg["separators"],
            dataset_path=cfg["dataset_path"], device=cfg["device"],
            train_split=cfg["train_split"], seed=cfg["seed"],
        )
    else:
        raise ValueError(f"Unknown selection mode: {cfg['cie']['selection']}")

    flat = score.flatten()
    top_idx = torch.topk(flat, cfg["cie"]["topk_heads"]).indices
    top_heads = [(int(i) // n_heads, int(i) % n_heads) for i in top_idx]

    # 4. Steering sweep ---------------------------------------------------
    records: list[dict] = []

    shared_meta = {
        "experiment_id": exp_id, "model": cfg["model"], "task": task,
        "split": "test", "seed": cfg["seed"],
        "n_shots_clean": cfg["mean_activations"]["n_shots"],
        "n_shots_corrupted": cfg["cie"]["n_shots_corrupted"],
        "n_shots_eval": cfg["steering_sweep"]["n_shots_eval"],
    }

    # 4a. Baseline (no intervention)
    baseline = eval_with_intervention(
        model, tokenizer, task=task, iv_set=None,
        n_shots_eval=cfg["steering_sweep"]["n_shots_eval"],
        prefixes=cfg["prefixes"], separators=cfg["separators"],
        dataset_path=cfg["dataset_path"], device=cfg["device"],
        max_examples=cfg["steering_sweep"]["max_examples"],
        train_split=cfg["train_split"], seed=cfg["seed"],
    )
    append_records(records, baseline, {**shared_meta, "intervention_type": "none",
                                       "target_layer": None, "target_head": None,
                                       "scale": 0.0, "mode": "add"})

    # 4b. Per-head steering sweep
    for (L, H) in top_heads:
        vec = mean["heads"][L, H].to(cfg["device"])
        for scale in cfg["steering_sweep"]["scales"]:
            iv = Intervention(vector=vec, layer=L, component="attn_head",
                              head=H, scale=scale, mode="add", position="last")
            recs = eval_with_intervention(
                model, tokenizer, task=task, iv_set=InterventionSet([iv]),
                n_shots_eval=cfg["steering_sweep"]["n_shots_eval"],
                prefixes=cfg["prefixes"], separators=cfg["separators"],
                dataset_path=cfg["dataset_path"], device=cfg["device"],
                max_examples=cfg["steering_sweep"]["max_examples"],
                train_split=cfg["train_split"], seed=cfg["seed"],
            )
            append_records(records, recs, {**shared_meta, "intervention_type": "head",
                                           "target_layer": L, "target_head": H,
                                           "scale": float(scale), "mode": "add"})

    # 4c. MLP steering sweep
    for L in cfg["steering_sweep"]["mlp_layers"]:
        vec = mean["mlps"][L].to(cfg["device"])
        for scale in cfg["steering_sweep"]["scales"]:
            iv = Intervention(vector=vec, layer=L, component="mlp_out",
                              scale=scale, mode="add", position="last")
            recs = eval_with_intervention(
                model, tokenizer, task=task, iv_set=InterventionSet([iv]),
                n_shots_eval=cfg["steering_sweep"]["n_shots_eval"],
                prefixes=cfg["prefixes"], separators=cfg["separators"],
                dataset_path=cfg["dataset_path"], device=cfg["device"],
                max_examples=cfg["steering_sweep"]["max_examples"],
                train_split=cfg["train_split"], seed=cfg["seed"],
            )
            append_records(records, recs, {**shared_meta, "intervention_type": "mlp",
                                           "target_layer": L, "target_head": None,
                                           "scale": float(scale), "mode": "add"})

    # 5. Save -------------------------------------------------------------
    out_path = os.path.join(results_dir, exp_id, "results.parquet")
    save_results(records, out_path)
    print(f"Saved {len(records)} records to {out_path}")


if __name__ == "__main__":
    main(CONFIG)