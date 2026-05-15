import os
import subprocess
import sys

TASKS = [
    # # abstractive
    "antonym", "capitalize", "capitalize_first_letter", "capitalize_last_letter", "capitalize_second_letter",
    "country-capital", "country-currency", "english-french", "english-german", "english-spanish",
    "landmark-country", "lowercase_first_letter", "lowercase_last_letter", "national_parks",
    "next_capital_letter", "next_item", "park-country", "present-past", "prev_item",
    "product-company", "singular-plural", "synonym", "word_length",
    # extractive
    "adjective_v_verb_3", "adjective_v_verb_5",
    "alphabetically_first_3", "alphabetically_first_5",
    "alphabetically_last_3", "alphabetically_last_5",
    "animal_v_object_3", "animal_v_object_5",
    "choose_first_of_3", "choose_first_of_5",
    "choose_last_of_3", "choose_last_of_5",
    "choose_middle_of_3", "choose_middle_of_5",
    "color_v_animal_3", "color_v_animal_5",
    "concept_v_object_3", "concept_v_object_5",
    "conll2003_location", "conll2003_organization", "conll2003_person",
    "fruit_v_animal_3", "fruit_v_animal_5",
    "object_v_concept_3", "object_v_concept_5",
    # "sentiment", "squad_val", 
    "verb_v_adjective_3", "verb_v_adjective_5",
]
# TASKS = [
#     "conll2003_person",
#     "fruit_v_animal_3",
#     "fruit_v_animal_5",
#     "object_v_concept_3",
#     "verb_v_adjective_3",
#     "verb_v_adjective_5",
# ]

# TASKS = [
#     "commonsense_qa",
#     "person-instrument",
#     "ag_news",
#     "person-sport",
#     "person-occupation",
# ]

# MODEL_NAME = "Qwen/Qwen3-4B-Instruct-2507"
MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"
PROMPT_TYPE = "short"
BASELINE = "equiprobable"
SKIP_EVAL = False


def main():
    for task in TASKS:
        print(f"\n{'='*60}")
        print(f"Running task: {task}")
        print(f"{'='*60}")
        cmd = [
            sys.executable, "slim/prompt_fv.py",
            "--dataset_name", task,
            "--model_name", MODEL_NAME,
            "--prompt_type", PROMPT_TYPE,
            "--prompt_baseline", BASELINE,
            "--cache_prompt_prefixes",
        ]
        if SKIP_EVAL:
            cmd.append("--skip_eval")
        env = {**os.environ, "WANDB_MODE": "disabled"}
        result = subprocess.run(cmd, env=env)
        if result.returncode != 0:
            print(f"[ERROR] Task '{task}' failed with exit code {result.returncode}", file=sys.stderr)


if __name__ == "__main__":
    main()
