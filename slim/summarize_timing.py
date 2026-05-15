import json

# Llama 3.2-3B-Instruct
N_LAYERS = 28
N_HEADS = 24
PASSES_PER_TRIAL = N_LAYERS * N_HEADS + 1  # +1 for the clean run
N_INSTANCES = 5 * 5  # samples × instructions

TASKS = [
    "conll2003_person", "english-french", "capitalize", "present-past", "next_item",
    "conll2003_location", "english-german", "country-currency", "color_v_animal_3", "country-capital",
]

# MODEL_NAME = "Qwen/Qwen3-4B-Instruct-2507"
MODEL_NAME = "meta-llama/Llama-3.2-3B-Instruct"
# MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"
OUTPUT_DIR = f"timing_results/{MODEL_NAME.split('/')[-1]}"

trial_stats = {}  # task -> avg_trial_seconds
all_trial_times = []

for path in [f"{OUTPUT_DIR}/{task}_timing.json" for task in TASKS]:
    with open(path) as f:
        data = json.load(f)

    trial_times = [t["seconds"] for t in data["trials"]]
    if not trial_times:
        continue

    avg_trial = sum(trial_times) / len(trial_times)
    all_trial_times.extend(trial_times)
    trial_stats[data["task"]] = avg_trial

stats_path = f"{OUTPUT_DIR}/processing_stats.json"
try:
    with open(stats_path) as f:
        stats = json.load(f)
    proc_stats = stats["per_task_seconds"]
except FileNotFoundError:
    proc_stats = {}

def inst_per_min(secs_per_inst):
    return 60 / secs_per_inst

common_tasks = [t for t in TASKS if t in trial_stats and t in proc_stats]
solo_tasks   = [t for t in TASKS if t in trial_stats and t not in proc_stats]

if common_tasks:
    print(f"{'Task':40s}  {'inst/min (slim)':>15}  {'inst/min (proc)':>15}")
    print("-" * 74)
    for task in common_tasks:
        slim_spi = trial_stats[task]                      # already s/instance
        proc_spi = proc_stats[task] / N_INSTANCES
        print(f"{task:40s}  {inst_per_min(slim_spi):15.2f}  {inst_per_min(proc_spi):15.2f}")

if solo_tasks:
    print(f"\n{'Task':40s}  {'inst/min (slim)':>15}")
    print("-" * 58)
    for task in solo_tasks:
        print(f"{task:40s}  {inst_per_min(trial_stats[task]):15.2f}")

if all_trial_times:
    overall_spi = sum(all_trial_times) / len(all_trial_times)
    print(f"\n{'OVERALL (slim)':40s}  {inst_per_min(overall_spi):15.2f}  ({overall_spi:.2f}s/inst, {PASSES_PER_TRIAL} passes)")
    if proc_stats:
        avg_proc_spi = sum(proc_stats[t] for t in common_tasks) / len(common_tasks) / N_INSTANCES
        print(f"{'OVERALL (proc)':40s}  {inst_per_min(avg_proc_spi):15.2f}")
        print(f"{'speedup (proc / slim)':40s}  {overall_spi / avg_proc_spi:15.2f}x")

if proc_stats:
    print(f"\nProcessing stats: {stats['total_instances']} instances  "
          f"{stats['time_seconds']:.2f}s total  {stats['instances_per_second']:.2f} inst/s  "
          f"({stats['tasks_skipped']} tasks skipped)")
