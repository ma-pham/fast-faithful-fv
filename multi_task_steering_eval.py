import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from utils_v2 import load_top_heads
from steering_eval import eval_sae, load_top_heads_from_cie


class Argsclass:
    pass


def main():
    args = Argsclass()

    # args.model_name = "Qwen/Qwen3-4B-Instruct-2507"
    args.model_name = "meta-llama/Llama-3.2-1B-Instruct"
    model_short = args.model_name.split('/')[-1]
    args.dataset_path = "dataset_files_fv"
    args.head_mode = "task"       # "global" or "task"
    args.head_ids_path = f"heads_{model_short}.pt"   # used in global mode
    args.cie_path = f"cie_scores/{model_short}"      # used in task mode
    args.activations_path = f"mean_acts/{model_short}"
    args.topk_heads = 20
    args.output_dir = f"results/{model_short}"
    args.n_icl_examples = 10
    args.mean_acts_trials = 100
    args.steer_scale = 1.0
    args.seed = 42
    args.train_split = 0.7

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
    ).to(device)
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    n_heads = model.config.num_attention_heads

    tasks = [
        "antonym", "capitalize", "capitalize_first_letter", "country-capital",
        "country-currency", "english-french", "english-german", "english-spanish", "landmark-country",
        "lowercase_first_letter", "national_parks", "park-country", "person-sport", "present-past",
        "product-company", "sentiment", "singular-plural", "synonym",
    ]

    # In global mode, load the shared head set once up front.
    global_top_heads = None
    if args.head_mode == "global":
        global_top_heads = load_top_heads(args.head_ids_path, n_heads, args.topk_heads)
        print(f"Global mode: using {len(global_top_heads)} shared heads")

    results = {}
    topones = []
    for task in tasks:
        if args.head_mode == "task":
            top_heads = load_top_heads_from_cie(args.cie_path, task, n_heads, args.topk_heads, device=device)
            print(f"----- task: {task} | top heads: {top_heads[:3]}... ------")
        else:
            top_heads = global_top_heads
            print(f"----- task: {task} ------")

        args.dataset_name = task
        results[task] = eval_sae(args, model=model, tokenizer=tokenizer, top_heads=top_heads)
        print(results[task]['steered_topk'][1])
        topones.append(results[task]['steered_topk'][1])

    results['task_mean'] = torch.tensor(topones).mean()

    save_path = f"{args.output_dir}/{args.head_mode}.pt"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(results, save_path)


if __name__ == '__main__':
    main()
