import torch
import matplotlib.pyplot as plt
import numpy as np

# MODEL_NAME = "Qwen/Qwen3-4B-Instruct-2507"
MODEL_NAME = "meta-llama/Llama-3.2-1B-Instruct"
model_short = MODEL_NAME.split('/')[-1]

global_results = torch.load(f"results/{model_short}/global.pt", map_location="cpu")
task_results = torch.load(f"results/{model_short}/task.pt", map_location="cpu")

tasks = [k for k in global_results if k != "task_mean"]

global_acc = [global_results[t]["steered_topk"][1] for t in tasks]
task_acc   = [task_results[t]["steered_topk"][1]   for t in tasks]

global_mean = float(global_results["task_mean"])
task_mean   = float(task_results["task_mean"])

x = np.arange(len(tasks))
width = 0.35

fig, ax = plt.subplots(figsize=(14, 5))
bars_g = ax.bar(x - width / 2, global_acc, width, label=f"global (mean={global_mean:.3f})")
bars_t = ax.bar(x + width / 2, task_acc,   width, label=f"task   (mean={task_mean:.3f})")

ax.set_xticks(x)
ax.set_xticklabels(tasks, rotation=45, ha="right")
ax.set_ylabel("Top-1 accuracy")
ax.set_ylim(0, 1)
ax.legend()
ax.set_title(f"Steering accuracy: global heads vs task-specific heads ({model_short})")

plt.tight_layout()
plt.savefig(f"results/{model_short}/comparison.png", dpi=150)
print(f"Saved results/{model_short}/comparison.png")
