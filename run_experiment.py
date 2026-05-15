"""One experiment run: mean activations -> CIE -> steering sweep -> save."""
import os, torch

from src.loader import load_model
from src.collect_mean_activations import collect_attn, collect_mlp
from src.cie import get_or_compute_cie

CONFIG = {
    # "model": "meta-llama/Llama-3.2-1B-Instruct",
    "model": "Qwen/Qwen2.5-7B-Instruct",

    "dataset_path": "./dataset_files_fv",
    "train_split": 0.7,
    "seed": 42,
    "device": "cuda:1",
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
        "n_tokens": 1,
    },

    "cie": {
        "n_shots_corrupted": 0,   # set to 0 for zero-shot CIE scheme
        "n_trials": 25,
        "topk_heads": 20,
        "topk_mlp": 5,
    },

    "steering_sweep": {
        "scales": [0.5, 1.0, 1.5, 2.0, 4.0, 8.0],
        "n_shots_eval": 0,
        "max_examples": 200,
        # Optionally restrict evaluation to a subset of tasks.
        # Set to None (or omit) to evaluate all tasks.
        "eval_tasks": None,
    },

    "paths": {
        "cache_dir": "./cache/",
        "results_dir": "./results",
    },
}



def main(cfg: dict):
    tasks = [
        "antonym", "capitalize", "capitalize_first_letter", "country-capital",
        "country-currency", "english-french", "english-german", "english-spanish", "landmark-country",
        "lowercase_first_letter", "national_parks", "park-country", "person-sport", "present-past",
        "product-company", "sentiment", "singular-plural", "synonym",
    ]

    model, tokenizer = load_model(cfg["model"], device=cfg["device"], dtype=(cfg["dtype"]))

    model_short = cfg["model"].split("/")[-1]
    cache_dir   = cfg["paths"]["cache_dir"]
    results_dir = cfg["paths"]["results_dir"]

    n_heads = model.config.num_attention_heads

    # 1. Mean activations -------------------------------------------------
    # mean_acts[task] = {"heads": (n_layers, n_heads, head_dim),
    #                    "mlps":  (n_layers, hidden_size)}
    mean_parent_path = os.path.join(cache_dir, model_short, "mean_acts")
    mean_acts: dict = {}

    for task in tasks:
        attn_path = os.path.join(mean_parent_path, "attn", f"{task}.pt")
        mlp_path  = os.path.join(mean_parent_path, "mlp",  f"{task}.pt")

        collect_kwargs = dict(
            model=model, tokenizer=tokenizer, task=task,
            n_shots=cfg["mean_activations"]["n_shots"],
            n_trials=cfg["mean_activations"]["n_trials"],
            n_tokens=cfg["mean_activations"]["n_tokens"],
            prefixes=cfg["prefixes"], seperators=cfg["separators"],
            dataset_path=cfg["dataset_path"],
            train_split=cfg["train_split"],
            device=cfg["device"],
            seed=cfg["seed"],
        )
        # collect_attn returns (n_layers, n_heads, head_dim)
        # collect_mlp  returns (n_layers, hidden_size)
        head_acts = collect_attn(**collect_kwargs, save_path=attn_path)
        mlp_acts  = collect_mlp(**collect_kwargs,  save_path=mlp_path)
        mean_acts[task] = {"heads": head_acts.cpu(), "mlps": mlp_acts.cpu()}

    # 2. CIE --------------------------------------------------------------
    # Compute CIE across all tasks then average to identify the most
    # causally important attention heads and MLP layers.
    cie_cache_dir = os.path.join(cache_dir, model_short, "cie")
    all_head_cie: dict = {}
    all_mlp_cie:  dict = {}

    for task in tasks:
        cie = get_or_compute_cie(
            model, tokenizer, task=task,
            mean_head_acts=mean_acts[task]["heads"].to(cfg["device"]),
            mean_mlp_acts=mean_acts[task]["mlps"].to(cfg["device"]),
            n_shots_corrupted=cfg["cie"]["n_shots_corrupted"],
            n_trials=cfg["cie"]["n_trials"],
            prefixes=cfg["prefixes"], separators=cfg["separators"],
            dataset_path=cfg["dataset_path"], device=cfg["device"],
            train_split=cfg["train_split"], seed=cfg["seed"],
            cache_dir=cie_cache_dir,
        )
        all_head_cie[task] = cie["heads"].cpu()  # (n_layers, n_heads)
        all_mlp_cie[task]  = cie["mlp"].cpu()    # (n_layers,)

    # Average CIE across tasks
    mean_head_cie = torch.stack(list(all_head_cie.values())).mean(dim=0)  # (n_layers, n_heads)
    mean_mlp_cie  = torch.stack(list(all_mlp_cie.values())).mean(dim=0)   # (n_layers,)

    # 3. Select top-k heads and MLP layers by CIE -------------------------
    flat    = mean_head_cie.flatten()
    top_idx = torch.topk(flat, cfg["cie"]["topk_heads"]).indices
    top_heads = [(int(i) // n_heads, int(i) % n_heads) for i in top_idx]

    topk_mlp = cfg["cie"]["topk_mlp"]
    top_mlp_layers = torch.topk(mean_mlp_cie, topk_mlp).indices.tolist()

    print(f"Top {len(top_heads)} attention heads (layer, head): {top_heads}")
    print(f"Top {topk_mlp} MLP layers by CIE: {top_mlp_layers}")

    # # 4. Steering sweep ---------------------------------------------------
    # # Determine which tasks to evaluate (default: all tasks).
    # eval_tasks = cfg["steering_sweep"].get("eval_tasks") or tasks
    # scales     = cfg["steering_sweep"]["scales"]

    # # Steering vectors come from each eval task's own mean activations.
    # sweep_results = {}  # task -> scale -> result dict

    # for task in eval_tasks:
    #     steering_heads, steering_mlp = build_steering_vecs(
    #         mean_acts[task], top_heads, top_mlp_layers
    #     )
    #     steering_heads = steering_heads.to(cfg["device"])
    #     steering_mlp   = steering_mlp.to(cfg["device"])

    #     task_results = {}
    #     for scale in scales:
    #         print(f"\n[Sweep] task={task}  scale={scale}")
    #         result = eval_joint_steered(
    #             task=task,
    #             model=model,
    #             tokenizer=tokenizer,
    #             top_heads=top_heads,
    #             top_mlp_layers=top_mlp_layers,
    #             steering_heads=steering_heads,
    #             steering_mlp=steering_mlp,
    #             n_heads=n_heads,
    #             steer_scale=scale,
    #             cfg=cfg,
    #         )
    #         task_results[scale] = result
    #         print(f"  top-1={result['steered_topk'][1]:.3f}  top-3={result['steered_topk'][3]:.3f}")

    #     sweep_results[task] = task_results

    # # 5. Save results -------------------------------------------------------
    # os.makedirs(os.path.join(results_dir, model_short), exist_ok=True)
    # save_path = os.path.join(results_dir, model_short, f"{exp_id}_sweep.pt")
    # torch.save(
    #     {
    #         "config": {k: v for k, v in cfg.items() if k != "dtype"},
    #         "top_heads": top_heads,
    #         "top_mlp_layers": top_mlp_layers,
    #         "sweep": sweep_results,
    #     },
    #     save_path,
    # )
    # print(f"\nResults saved to {save_path}")


if __name__ == "__main__":
    main(CONFIG)