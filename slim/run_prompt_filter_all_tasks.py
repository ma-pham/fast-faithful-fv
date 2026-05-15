import subprocess
import sys

TASKS = [
    "antonym", "capitalize", "capitalize_first_letter", "country-capital",
    "country-currency", "english-french", "english-german", "english-spanish", "landmark-country",
    "lowercase_first_letter", "national_parks", "park-country", "person-sport", "present-past",
    "product-company", "sentiment", "singular-plural", "synonym",
]

MODEL_NAME = "meta-llama/Llama-3.2-3B-Instruct"
PROMPT_TYPE = "short"


def main():
    for task in TASKS:
        print(f"\n{'='*60}")
        print(f"Running task: {task}")
        print(f"{'='*60}")
        cmd = [
            sys.executable, "prompt_filter.py",
            "--dataset_name", task,
            "--model_name", MODEL_NAME,
            "--prompt_type", PROMPT_TYPE,
        ]
        result = subprocess.run(cmd)
        if result.returncode != 0:
            print(f"[ERROR] Task '{task}' failed with exit code {result.returncode}", file=sys.stderr)


if __name__ == "__main__":
    main()
