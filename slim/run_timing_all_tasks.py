"""Run every task and collect indirect effect computation times.

Calls prompt_function_vector_main directly (model loaded once) and captures
TIMING log lines via a loguru sink. IE files go to timing_results/ie_scratch
so existing cached indirect effects are not overwritten.
"""
import json
import os
import re

from loguru import logger

from recipe.function_vectors.prompt_based_function_vector import prompt_function_vector_main

# ALL_TASKS = [
#     # abstractive
#     "antonym", "capitalize", "capitalize_first_letter", "capitalize_last_letter",
#     "capitalize_second_letter",
#     "country-capital", "country-currency", "english-french", "english-german", "english-spanish",
#     "landmark-country", "lowercase_first_letter", "lowercase_last_letter", "national_parks",
#     "next_capital_letter", "next_item", "park-country", "present-past", "prev_item",
#     "product-company", "singular-plural", "synonym", "word_length",
#     # extractive
#     "adjective_v_verb_3", "adjective_v_verb_5",
#     "alphabetically_last_3",
#     "animal_v_object_3", "animal_v_object_5",
#     "choose_first_of_3", "choose_first_of_5",
#     "choose_last_of_3", "choose_last_of_5",
#     "choose_middle_of_3", "choose_middle_of_5",
#     "color_v_animal_3", "color_v_animal_5",
#     "concept_v_object_3", "concept_v_object_5",
#     "conll2003_location", "conll2003_organization", "conll2003_person",
#     "fruit_v_animal_3", "fruit_v_animal_5",
#     "object_v_concept_3", "object_v_concept_5",
#     "verb_v_adjective_3", "verb_v_adjective_5",
# ]

# random.sample(ALL_TASKS, k=10, random.seed(42))
TASKS = [
    "conll2003_person", "english-french", "capitalize", "present-past", "next_item",
    "conll2003_location", "english-german", "country-currency", "color_v_animal_3", "country-capital",
]

MODEL_NAME = "meta-llama/Llama-3.2-3B-Instruct"
PROMPT_TYPE = "short"
BASELINE = "equiprobable"
OUTPUT_DIR = f"timing_results/{MODEL_NAME.split('/')[-1]}"

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.environ.setdefault("WANDB_MODE", "disabled")

failed = []

for task in TASKS:
    out_path = os.path.join(OUTPUT_DIR, f"{task}_timing.json")
    if os.path.exists(out_path):
        print(f"Skipping {task} (already exists)", flush=True)
        continue

    print(f"\n{'='*60}")
    print(f"Timing task: {task}")
    print(f"{'='*60}", flush=True)

    captured = []

    def _sink(message):
        if "TIMING" in message.record["message"]:
            captured.append(message.record["message"])

    sink_id = logger.add(_sink, format="{message}")
    os.makedirs(os.path.join(OUTPUT_DIR, "ie_scratch", task), exist_ok=True)

    try:
        prompt_function_vector_main([
            "--dataset_name", task,
            "--model_name", MODEL_NAME,
            "--prompt_type", PROMPT_TYPE,
            "--prompt_baseline", BASELINE,
            "--batch_size", "1",
            "--ie_path_root", os.path.join(OUTPUT_DIR, "ie_scratch"),
            "--skip_eval",
            "--force_indirect_effect",
        ])
    except Exception as e:
        logger.error(f"Task {task} failed: {e}")
        failed.append(task)
    finally:
        logger.remove(sink_id)

    total_match = next((re.search(r"TIMING total indirect_effect: ([\d.]+)s", m) for m in captured if "total" in m), None)
    trial_matches = [(m,) for m in captured if "TIMING trial" in m]

    timing = {
        "task": task,
        "model": MODEL_NAME,
        "total_seconds": float(total_match.group(1)) if total_match else None,
        "trials": [
            {"prompt": int(p), "trial": int(t), "seconds": float(s)}
            for m in captured
            for p, t, s in re.findall(r"TIMING trial (\d+),(\d+): ([\d.]+)s", m)
        ],
    }

    out_path = os.path.join(OUTPUT_DIR, f"{task}_timing.json")
    with open(out_path, "w") as f:
        json.dump(timing, f, indent=2)

    status = f"{timing['total_seconds']:.2f}s" if timing["total_seconds"] is not None else "NO TIMING FOUND"
    print(f"  -> {task}: {status}", flush=True)

print(f"\n{'='*60}")
if failed:
    print(f"Failed ({len(failed)}/{len(TASKS)}): {failed}")
else:
    print(f"All {len(TASKS)} tasks completed. Results in {OUTPUT_DIR}/")
