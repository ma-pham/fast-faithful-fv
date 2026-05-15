import argparse
import datetime
import gc
import itertools
import json
import os
import time
import typing
from pathlib import Path

import numpy as np
import torch
from git import Repo
from loguru import logger

from recipe.function_vectors.compute_indirect_effect import (
    PromptBaseline,
    compute_prompt_based_indirect_effect,
)
from recipe.function_vectors.generate_prompts_for_dataset import (
    LONG,
    PROMPT_TYPES,
    SHORT,
)
from recipe.function_vectors.utils.eval_utils import (
    compute_dataset_baseline,
    make_valid_path_name,
    prompt_based_eval,
    prompt_based_eval_no_intervention,
)
from recipe.function_vectors.utils.extract_utils import (
    compute_function_vector,
    compute_universal_function_vector_top_heads_from_file,
    get_prompt_based_mean_head_activations,
)
from recipe.function_vectors.utils.model_utils import load_gpt_model_and_tokenizer, set_seed
from recipe.function_vectors.utils.prompt_utils import (
    filter_prompts_by_max_tokens,
    load_dataset,
)

STORAGE_ROOT = os.environ.get("STORAGE_ROOT")

SETTINGS_BY_PROMPT_TYPE = {
    SHORT: {
        "propmt_max_len_tokens": 16,
        "saved_prompts_suffix": "prompts",
    },
    LONG: {
        "propmt_max_len_tokens": 64,
        "saved_prompts_suffix": "long_prompts",
    },
}



def _protected_decorator(func):
    def protected(output_path, output_data, *args, **kwargs):
        lock_path = Path(f"{output_path}.lock")
        if lock_path.exists():
            logger.warning(f"Skipping saving {output_path} as the lock file exists")
            return

        try:
            lock_path.touch()
            func(output_path, output_data, *args, **kwargs)
        finally:
            lock_path.unlink()

    return protected


@_protected_decorator
def _protected_json_dump(output_path, output_data):
    with open(output_path, "w") as results_file:
        json.dump(output_data, results_file, indent=2)


@_protected_decorator
def _protected_torch_save(output_path, output_data):
    torch.save(output_data, output_path)


def _gc_clear_cache():
    gc.collect()
    torch.cuda.empty_cache()


class FailureException(Exception):
    pass


def _log_failure(output_path, failure_code, failure_reason):
    if not output_path.lower().endswith(".json"):
        output_path += ".json"

    logger.warning(f"Logging failure to {output_path}")

    _protected_json_dump(
        output_path,
        {
            "code": failure_code,
            "reason": failure_reason,
            "timestamp": datetime.datetime.now().isoformat(),
        },
    )

    raise FailureException(failure_reason)


def _mean_act_indirect_effect_fv(
    args,
    mean_activations_path,
    top_heads_path,
    fv_path,
    model,
    tokenizer,
    model_config,
    dataset,
    selected_prompts,
    filter_set_per_split,
    save_path_root,
    indirect_effect_path,
    baseline_generator_kwargs,
):
    dataset_name = args.dataset_name
    n_best_prompts = args.n_best_prompts
    n_icl_examples = args.n_icl_examples
    n_top_heads = args.n_top_heads
    prefixes = args.prefixes
    prompt_baseline = args.prompt_baseline
    seed = args.seed
    separators = args.separators
    universal_set = args.universal_set

    # Load or Re-Compute mean_head_activations
    if mean_activations_path is not None and os.path.exists(mean_activations_path):
        logger.info(f"Loading Mean Activations from {mean_activations_path}")
        mean_activations = torch.load(mean_activations_path)
    else:
        logger.info(
            f"Computing Mean Activations with {args.total_mean_activation_examples} = {args.mean_activation_trials_per_prompt} examples * {n_best_prompts} prompts"
        )
        set_seed(seed)
        mean_activations = get_prompt_based_mean_head_activations(
            dataset,
            prompts=selected_prompts,
            model=model,
            model_config=model_config,
            tokenizer=tokenizer,
            n_trials_per_prompt=args.mean_activation_trials_per_prompt,
            prefixes=prefixes,
            separators=separators,
            filter_set=filter_set_per_split["train"],
            n_icl_examples=n_icl_examples,
            query_dataset="train",
            batch_size=args.batch_size,
        )
        args.mean_activations_path = f"{save_path_root}/{dataset_name}_mean_head_activations.pt"

        _protected_torch_save(args.mean_activations_path, mean_activations)

    _gc_clear_cache()

    # Compute function vector
    fv = None
    top_heads = None
    if universal_set:
        if os.path.exists(fv_path):
            logger.info(f"Loading universal function vector from {fv_path}")
            fv = torch.load(fv_path)
        else:
            logger.info(
                f"Loading top heads from {top_heads_path} to compute universal function vector and saving to {fv_path}"
            )
            fv, top_heads = compute_universal_function_vector_top_heads_from_file(
                mean_activations,
                model,
                model_config=model_config,
                top_heads_path=top_heads_path,
                n_top_heads=n_top_heads,
            )
            _protected_torch_save(fv_path, fv)

    else:
        # Load or Re-Compute indirect_effect values -- only necessary in the non-universal case
        if indirect_effect_path is not None and os.path.exists(indirect_effect_path) and not args.force_indirect_effect:
            logger.info(f"Loading Indirect Effects from {indirect_effect_path}")
            indirect_effect = torch.load(indirect_effect_path)
        else:  # Only compute indirect effects if we need to
            logger.info(
                f"Computing Indirect Effects with {args.total_indirect_effect_examples} = {args.indirect_effect_trials_per_prompt} examples * {n_best_prompts} prompts"
            )
            set_seed(seed)
            args.partial_indirect_effect_path = f"{indirect_effect_path}.partial"

            indirect_effect = compute_prompt_based_indirect_effect(
                dataset,
                selected_prompts,
                mean_activations,
                baseline=prompt_baseline,
                model=model,
                model_config=model_config,
                tokenizer=tokenizer,
                n_trials_per_prompt=args.indirect_effect_trials_per_prompt,
                last_token_only=True,
                prefixes=prefixes,
                separators=separators,
                filter_set=filter_set_per_split["train"],
                n_icl_examples=n_icl_examples,
                partial_path=args.partial_indirect_effect_path,
                query_dataset="train",
                baseline_generator_kwargs=baseline_generator_kwargs,
                batch_size=args.batch_size,
                forced=args.force_indirect_effect,
            )

            _protected_torch_save(args.indirect_effect_path, indirect_effect)

        _gc_clear_cache()

        if os.path.exists(fv_path):
            fv = torch.load(fv_path)
        else:
            fv, top_heads = compute_function_vector(
                mean_activations,
                indirect_effect,
                model,
                model_config=model_config,
                n_top_heads=n_top_heads,
                prompt_based=True,
            )
            _protected_torch_save(fv_path, fv)
            _protected_torch_save(top_heads_path, top_heads)

    _gc_clear_cache()
    return fv


def _run_zs_eval(
    args,
    fv_or_fvs,
    edit_layer_or_layers,
    model,
    tokenizer,
    model_config,
    dataset,
    filter_set_per_split,
):
    set_seed(args.seed)
    results = prompt_based_eval(
        dataset=dataset,
        fv_vector_or_vectors=fv_or_fvs,
        edit_layer_or_layers=edit_layer_or_layers,
        prompts=args.evaluation_prompts,
        model=model,
        model_config=model_config,
        tokenizer=tokenizer,
        filter_set=filter_set_per_split["test"],
        prefixes=args.prefixes,
        separators=args.separators,
        query_dataset="test",
        n_icl_examples=0,
        shuffle_icl_labels=False,
        batch_size=args.batch_size,
    )
    _gc_clear_cache()
    return results


def prompt_function_vector_main(alt_args: typing.Optional[typing.List[str]] = None):
    model = None

    parser = argparse.ArgumentParser()

    # Core
    parser.add_argument("--dataset_name", type=str, required=True, help="Name of the dataset to be loaded")
    parser.add_argument("--model_name", type=str, default="meta-llama/Llama-3.2-1B", help="HuggingFace model name")
    parser.add_argument("--revision", type=str, default=None, help="Model checkpoint revision (pythia/olmo)")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch_size", type=int, default=0, help="Eval batch size (0 = auto)")
    parser.add_argument("--edit_layer", type=int, default=-1, help="Intervention layer; -1 sweeps all layers")
    parser.add_argument("--n_top_heads", type=int, default=20, help="Attention heads used to compute function vector")
    parser.add_argument("--test_split", type=float, default=0.3, help="Fraction of data used as test set")
    parser.add_argument("--split_validation", action="store_true", help="Hold out a validation split")

    # Paths
    parser.add_argument("--root_data_dir", type=str, default=f"{STORAGE_ROOT}/function_vectors/dataset_files")
    parser.add_argument("--save_path_root", type=str, default=f"{STORAGE_ROOT}/function_vectors/full_results_prompt_based")
    parser.add_argument("--save_path_suffix", default=None, help="Subdirectory within save_path_root")
    parser.add_argument("--saved_prompts_root", type=str, default=f"{STORAGE_ROOT}/function_vectors/prompts")
    parser.add_argument("--saved_prompts_file", type=str, default=None, help="Override prompts filename")
    parser.add_argument("--saved_prompts_suffix", type=str, default=None, help="Override prompts filename suffix")
    parser.add_argument("--ie_path_root", type=str, default=None, help="Override path for indirect effects")
    parser.add_argument("--mean_activations_path", type=str, default=None, help="Precomputed mean activations file")
    parser.add_argument("--indirect_effect_path", type=str, default=None, help="Precomputed indirect effects file")

    # Prompt template
    parser.add_argument("--prefixes", type=json.loads, default={"input": "Q:", "output": "A:", "instructions": ""})
    parser.add_argument("--separators", type=json.loads, default={"input": "\n", "output": "\n\n", "instructions": "\n"})

    # Prompt-based FV
    p = parser.add_argument_group("Prompt-based FV")
    p.add_argument("--prompt_type", type=str, required=True, choices=PROMPT_TYPES)
    p.add_argument("--prompt_baseline", type=PromptBaseline, required=True, choices=list(PromptBaseline))  # type: ignore
    p.add_argument("--n_best_prompts", type=int, default=5, help="Number of top prompts to use")
    p.add_argument("--propmt_max_len_tokens", type=int, default=None, help="Max token length per prompt")
    p.add_argument("--allow_different_examples_per_prompt", action="store_true")
    p.add_argument("--min_passing_examples", type=int, default=None, help="Min examples a prompt must pass filtering")
    p.add_argument("--total_mean_activation_examples", type=int, default=100)
    p.add_argument("--total_indirect_effect_examples", type=int, default=25)
    p.add_argument("--n_icl_examples", type=int, default=0, help="ICL examples appended to each prompt")
    p.add_argument("--n_eval_icl_examples", type=int, default=10, help="Shuffled-label ICL examples at eval")
    p.add_argument("--n_indirect_effect_trials", type=int, default=5)
    p.add_argument("--evaluation_prompts", nargs="+", default=[""])
    p.add_argument("--force_prompt_evaluation", action="store_true")

    class StoreDictKeyPair(argparse.Action):
        def __init__(self, option_strings, dest, *args, **kwargs):
            super().__init__(option_strings, dest, *args, **kwargs)

        def __call__(self, parser, namespace, values, option_string=None):
            my_dict = {}
            for kv in values.split(","):  # type: ignore
                k, v = kv.split("=")
                my_dict[k] = v
            setattr(namespace, self.dest, my_dict)

    p.add_argument("--baseline_generator_kwargs", action=StoreDictKeyPair, metavar="KEY1=VAL1,KEY2=VAL2...")

    # Eval control
    parser.add_argument("--compute_baseline", type=bool, default=True)
    parser.add_argument("--metric", type=str, default="f1_score")
    parser.add_argument("--cache_prompt_prefixes", action="store_true")
    parser.add_argument("--skip_eval", action="store_true", help="Stop after computing indirect effects and FV, skip evaluation")
    parser.add_argument("--force_indirect_effect", action="store_true")
    parser.add_argument("--force_evaluation", action="store_true")
    parser.add_argument("--force_compute_baseline", action="store_true")

    # Universal FV
    u = parser.add_argument_group("Universal FV")
    u.add_argument("--universal_set", action="store_true")
    u.add_argument("--top_heads_dir", type=str, default=f"{STORAGE_ROOT}/function_vectors/full_results_top_heads")
    u.add_argument("--top_heads_prompt_type", type=str, choices=PROMPT_TYPES + ["both"], default="both")
    u.add_argument("--top_heads_baseline", type=str, choices=list(PromptBaseline) + ["all"], default="all")
    u.add_argument("--joint_intervention_min_layer_depth", type=float, default=0.25)
    u.add_argument("--joint_intervention_max_layer_depth", type=float, default=0.5)
    u.add_argument("--instruct_model_suffix", default="-Instruct")
    u.add_argument("--remove_model_suffix", default=None)
    ug = u.add_mutually_exclusive_group()
    ug.add_argument("--add_prompt_fv_twice", action="store_true")
    ug.add_argument("--use_min_abs_heads_prompt", action="store_true")
    ug.add_argument("--use_bottom_heads_prompt", action="store_true")
    ug.add_argument("--use_instruct_model_fv", action="store_true")

    args = parser.parse_args(alt_args)
    prompt_type = args.prompt_type

    if args.propmt_max_len_tokens is None:
        args.propmt_max_len_tokens = SETTINGS_BY_PROMPT_TYPE[prompt_type]["propmt_max_len_tokens"]

    if args.saved_prompts_suffix is None:
        args.saved_prompts_suffix = SETTINGS_BY_PROMPT_TYPE[prompt_type]["saved_prompts_suffix"]

    dataset_name = args.dataset_name
    model_name = args.model_name
    args.short_model_name = args.model_name[args.model_name.rfind("/") + 1 :]
    root_data_dir = args.root_data_dir
    save_path_suffix = args.save_path_suffix if args.save_path_suffix is not None else args.short_model_name
    if not args.save_path_root.endswith(prompt_type):
        args.save_path_root = f"{args.save_path_root}_{prompt_type}"

    save_path_root = f"{args.save_path_root}/{save_path_suffix}/{dataset_name}"
    saved_prompts_file = (
        args.saved_prompts_file
        if args.saved_prompts_file is not None
        else f"{dataset_name}_{args.saved_prompts_suffix}.json"
    )
    saved_prompts_full_path = f"{args.saved_prompts_root}/{saved_prompts_file}"

    ie_path_root = f"{args.ie_path_root}/{dataset_name}" if args.ie_path_root else save_path_root
    seed = args.seed
    device = args.device
    prompt_baseline = args.prompt_baseline

    indirect_effect_path = args.indirect_effect_path
    if indirect_effect_path is None:
        indirect_effect_path = f"{ie_path_root}/{dataset_name}_{prompt_baseline}_indirect_effect.pt"
        args.indirect_effect_path = indirect_effect_path

    n_top_heads = args.n_top_heads
    eval_edit_layer = args.edit_layer

    test_split = float(args.test_split)
    split_validation = args.split_validation
    splits = ("train", "valid", "test") if split_validation else ("train", "test")
    n_best_prompts = args.n_best_prompts

    if args.total_mean_activation_examples % n_best_prompts:
        raise ValueError(
            f"Total mean activation examples ({args.total_mean_activation_examples}) must be divisible by the number of prompts ({n_best_prompts})"
        )
    args.mean_activation_trials_per_prompt = args.total_mean_activation_examples // n_best_prompts
    if args.total_indirect_effect_examples % n_best_prompts:
        raise ValueError(
            f"Total indirect effect examples ({args.total_indirect_effect_examples}) must be divisible by the number of prompts ({n_best_prompts})"
        )
    args.indirect_effect_trials_per_prompt = args.total_indirect_effect_examples // n_best_prompts

    if args.min_passing_examples is None:
        args.min_passing_examples = max(args.mean_activation_trials_per_prompt, args.indirect_effect_trials_per_prompt)

    if args.force_indirect_effect and not args.force_evaluation:
        logger.warning(
            "Force indirect effect flag was set, but force evaluation was not, setting force evaluation to True"
        )
        args.force_evaluation = True

    baseline_generator_kwargs = args.baseline_generator_kwargs
    if baseline_generator_kwargs is None:
        baseline_generator_kwargs = {}
    if "prompt_type" not in baseline_generator_kwargs:
        baseline_generator_kwargs["prompt_type"] = prompt_type
    if "rng" not in baseline_generator_kwargs:
        baseline_generator_kwargs["rng"] = np.random.default_rng(seed)
    if "saved_prompts_root" not in baseline_generator_kwargs:
        baseline_generator_kwargs["saved_prompts_root"] = args.saved_prompts_root
    if "saved_prompts_file" not in baseline_generator_kwargs:
        baseline_generator_kwargs["saved_prompts_file"] = saved_prompts_file
    if "propmt_max_len_tokens" not in baseline_generator_kwargs:
        baseline_generator_kwargs["propmt_max_len_tokens"] = args.propmt_max_len_tokens
    if "model_name" not in baseline_generator_kwargs:
        baseline_generator_kwargs["model_name"] = model_name
    if "saved_prompts_suffix" not in baseline_generator_kwargs:
        baseline_generator_kwargs["saved_prompts_suffix"] = args.saved_prompts_suffix

    # evaluation_prompts = args.evaluation_prompts
    n_eval_icl_examples = args.n_eval_icl_examples

    prefixes = args.prefixes
    separators = args.separators
    compute_baseline = args.compute_baseline

    metric = args.metric
    universal_set = args.universal_set
    add_prompt_fv_twice = args.add_prompt_fv_twice
    use_min_abs_heads_prompt = args.use_min_abs_heads_prompt
    use_bottom_heads_prompt = args.use_bottom_heads_prompt
    use_instruct_model_fv = args.use_instruct_model_fv

    # Baseline key setting values
    if args.mean_activations_path is None:
        args.mean_activations_path = f"{ie_path_root}/{dataset_name}_mean_head_activations.pt"
    args.fv_path = f"{save_path_root}/{dataset_name}_{prompt_baseline}_fv.pt"
    args.top_heads_path = f"{save_path_root}/{dataset_name}_{prompt_baseline}_top_heads.pt"

    n_universal_only_flags = (
        int(add_prompt_fv_twice)
        + int(use_min_abs_heads_prompt)
        + int(use_bottom_heads_prompt)
        + int(use_instruct_model_fv)
    )

    if universal_set:
        if n_universal_only_flags > 1:
            raise ValueError(
                f"Cannot use more than one universal-only flag. Please choose one. Received add_prompt_fv_twice={add_prompt_fv_twice}, use_min_abs_heads_prompt={use_min_abs_heads_prompt}, use_bottom_heads_prompt={use_bottom_heads_prompt}, use_instruct_model_fv={use_instruct_model_fv}"
            )

        args.universal_fv_type = f"{args.top_heads_prompt_type}_{args.top_heads_baseline}"
        args.top_heads_path = f"{args.top_heads_dir}/{args.short_model_name}_{args.universal_fv_type}_top_heads.json"

        if add_prompt_fv_twice:
            logger.info("Adding prompt FV twice")
            args.universal_fv_type = "prompt_fv_twice"

        if use_min_abs_heads_prompt:
            args.top_heads_path = f"{args.top_heads_dir}/{args.short_model_name}_universal_all_min_abs_heads.json"
            logger.info(f"Using min absolute top heads from {args.top_heads_path}")
            logger.info("Using min absolute heads with prompt activations")
            args.universal_fv_type = "min_abs_heads_prompt"

        if use_bottom_heads_prompt:
            args.top_heads_path = args.top_heads_path.replace("top_heads.json", "bottom_heads.json")
            logger.info(f"Using bottom prompt-based heads from {args.top_heads_path}")
            args.universal_fv_type = "bottom_prompt_heads"

        if use_instruct_model_fv:
            instruct_model_short_name = args.short_model_name
            if args.remove_model_suffix is not None:
                args.remove_model_suffix = args.remove_model_suffix.replace('"', "").replace("'", "")
                instruct_model_short_name = instruct_model_short_name.replace(args.remove_model_suffix, "")

            args.instruct_model_suffix = args.instruct_model_suffix.replace('"', "").replace("'", "")
            instruct_model_short_name = f"{instruct_model_short_name}{args.instruct_model_suffix}"
            logger.info(f"Using instruct model {instruct_model_short_name} for universal FV")
            args.top_heads_path = args.top_heads_path.replace(args.short_model_name, instruct_model_short_name)
            logger.info(f"Using instruct model top heads from {args.top_heads_path}")
            args.mean_activations_path = args.mean_activations_path.replace(
                args.short_model_name, instruct_model_short_name
            )
            logger.info(f"Using instruct model mean activations from {args.mean_activations_path}")
            args.universal_fv_type = "instruct_model"

        args.fv_path = f"{save_path_root}/{dataset_name}_{args.universal_fv_type}_{n_top_heads}_heads_universal_fv.pt"

    elif n_universal_only_flags != 0:
        raise ValueError("Universal set flags are not supported without --universal_set")

    # In the universal set case, the mean activations and top head paths must exist already
    for path, name, must_exist in zip(
        (args.mean_activations_path, args.top_heads_path, args.fv_path),
        ("mean_activations", "top_heads", "function_vector"),
        (universal_set, universal_set, False),
    ):
        if path is None:
            raise ValueError(f"args.{name} path is None. Please check the arguments.")
        if must_exist and not os.path.exists(path):
            if name == "mean_activations":
                logged_failures = list(Path(path).parent.glob("*failure.json"))
                if len(logged_failures) > 0:
                    failure_path = logged_failures[0]
                    with open(failure_path, "r") as f:
                        failure_data = json.load(f)
                    failure_code = failure_data.get("code")
                    failure_reason = failure_data.get("reason")
                    logger.error(
                        f"Mean activations file not found. Found previous failure code: {failure_code}, reason: {failure_reason}"
                    )

            raise ValueError(f"args.{name} file not found: {path}. Please generate the {name} file first.")

    args.few_shot_batch_size = None
    # model_size = extract_model_size(model_name)
    is_conll = "conll2003" in dataset_name
    if args.batch_size == 0:
        if any(name in model_name for name in ("Llama-3.1-8B", "OLMo-2-1124-7B")):
            args.batch_size = 2 if is_conll else 3
            args.few_shot_batch_size = 1
        else:
            # We can probably afford more; but since mean activations runs with N = 20/prompt
            # and indirect effect with 5/prompt, this only really slows us down in evals, which is fine for now
            args.batch_size = 5

        if any(name in model_name.lower() for name in ("llama-2", "olmo")) and is_conll:
            args.few_shot_batch_size = 1

    if args.few_shot_batch_size is None:
        args.few_shot_batch_size = args.batch_size

    args.failure_path = f"{save_path_root}/{prompt_baseline}_failure.json"

    logger.debug(str(args))

    # Load Model & Tokenizer
    torch.set_grad_enabled(False)
    logger.info("Loading Model")
    model, tokenizer, model_config = load_gpt_model_and_tokenizer(model_name, device=device, revision=args.revision)

    if "should_prepend_bos" not in baseline_generator_kwargs:
        baseline_generator_kwargs["should_prepend_bos"] = not model_config["prepend_bos"]

    try:
        if args.edit_layer == -1:  # sweep over all layers if edit_layer=-1
            eval_edit_layer = [0, model_config["n_layers"]]

        # Load the dataset
        logger.info("Loading Dataset")
        set_seed(seed)
        if not os.path.exists(root_data_dir):
            raise ValueError(f"Dataset Directory Not Found: {root_data_dir}")

        dataset = load_dataset(
            dataset_name,
            root_data_dir=root_data_dir,
            test_size=test_split,
            seed=seed,
            split_valid=split_validation,
        )
        logger.debug(
            f"Loaded dataset {dataset_name} with the following sizes: { {k: len(d) for k, d in dataset.items()} }"
        )

        # if not os.path.exists(save_path_root):
        os.makedirs(save_path_root, exist_ok=True)

        # Load the prompts
        logger.info("Loading Prompts")
        if not os.path.exists(args.saved_prompts_root):
            raise ValueError(f"Prompts Directory Not Found: {args.saved_prompts_root}")

        with open(saved_prompts_full_path, "r") as prompts_file:
            prompts_data = json.load(prompts_file)

        assert prompts_data["dataset_name"] == dataset_name, (
            f"Dataset name mismatch, found {prompts_data['dataset_name']} in file, expected {dataset_name}"
        )
        all_task_prompts = prompts_data["prompts"]
        logger.info(f"Loaded {len(all_task_prompts)} prompts")

        # n_best_prompts = n_best_prompts if n_best_prompts is not None else len(prompts)
        # logger.info(f"Loaded {len(prompts)} prompts, using the first {n_best_prompts}")
        # prompts = prompts[:n_best_prompts]

        if args.propmt_max_len_tokens:
            keep_indices = filter_prompts_by_max_tokens(
                all_task_prompts,
                tokenizer=tokenizer,
                max_length_tokens=args.propmt_max_len_tokens,
            )
            if len(keep_indices) < n_best_prompts:
                logger.error(
                    f"Prompt max length filtering with {args.propmt_max_len_tokens} left only {len(keep_indices)} prompts, with n_best_prompts = {n_best_prompts}. Aborting..."
                )
                _log_failure(args.failure_path, "prompt_length_filter", "Insufficient prompts after length filter")

            logger.info(
                f"Filtering prompts to have at most {args.propmt_max_len_tokens} tokens, went from {len(all_task_prompts)} to {len(keep_indices)} prompts"
            )

            all_task_prompts = [all_task_prompts[i] for i in keep_indices]

        # 1. Compute Model Prompt-based Baseline & 2. Filter test set to cases where model gets it correct
        per_prompt_results_file_name = f"{save_path_root}/per_prompt_results.json"
        prompt_selection_results = None

        if os.path.exists(per_prompt_results_file_name) and not args.force_prompt_evaluation:
            try:
                with open(per_prompt_results_file_name, "r") as f:
                    prompt_selection_results = json.load(f)
            except Exception as e:
                logger.exception(e)

        if prompt_selection_results is None:
            logger.info("Evaluating full prompt set")

            prompt_selection_results = {}
            for split in splits:
                split_partial_per_prompt_results_file_name = f"{per_prompt_results_file_name}.{split}.partial"

                prompt_selection_results[split] = prompt_based_eval_no_intervention(
                    dataset,
                    prompts=all_task_prompts,
                    model=model,
                    model_config=model_config,
                    tokenizer=tokenizer,
                    compute_ppl=True,
                    relevant_split=split,
                    prefixes=prefixes,
                    separators=separators,
                    partial_path=split_partial_per_prompt_results_file_name,
                    cache_prompt_prefixes=args.cache_prompt_prefixes,
                    batch_size=args.batch_size,
                )

            logger.info(f"Saving full prompt filter results to {per_prompt_results_file_name}")
            args.fs_results_file_name = per_prompt_results_file_name
            _protected_json_dump(per_prompt_results_file_name, prompt_selection_results)

        # Filter out the best `n_best_prompts` prompts
        prompts_by_accuracy = [(p, pr[0][1]) for p, pr in prompt_selection_results["train"]["clean_topk"].items()]
        prompts_by_accuracy.sort(key=lambda t: t[1], reverse=True)
        selected_prompts = prompts_by_accuracy[:n_best_prompts]
        mean_accuracy = np.mean([p[1] for p in selected_prompts])
        selected_prompts = [p[0] for p in selected_prompts]
        logger.info(
            f"Using the following {n_best_prompts} propmts, whose mean task accuracy is {mean_accuracy:.4f}:\n{selected_prompts}"
        )

        filter_set_per_split = {}
        # Select relevant examples (per-prompt or global)
        if args.allow_different_examples_per_prompt:
            for split in splits:
                filter_set_per_split[split] = {}

                for p in selected_prompts:
                    prompt_example_rank_list = np.array(prompt_selection_results[split]["clean_rank_list"][p])
                    passing_indices = np.argwhere(prompt_example_rank_list == 0).squeeze()
                    n_passing_indices = 0 if passing_indices.ndim == 0 else len(passing_indices)

                    if n_passing_indices < args.min_passing_examples:
                        if split == "train":
                            logger.error(
                                f"Found a prompt that only {n_passing_indices} examples pass in the {split} split, min was set to {args.min_passing_examples}, aborting..."
                            )
                            _log_failure(
                                args.failure_path, "prompt_train_examples", "Insufficient train examples for prompt"
                            )
                        else:
                            if n_passing_indices == 0:
                                logger.error(
                                    f"Found a prompt that only {n_passing_indices} examples pass in the {split} split, musst have at least one test example, aborting..."
                                )
                                _log_failure(args.failure_path, "prompt_test_examples", "Zero test examples for prompt")
                            else:
                                logger.warning(
                                    f"Found a prompt that only {n_passing_indices} examples pass in the {split} split, min was set to {args.min_passing_examples}, ..."
                                )

                    filter_set_per_split[split][p] = passing_indices

        else:  # same filter set across all prompts
            for split in splits:
                selected_prompt_rank_lists = [
                    np.array(prompt_selection_results[split]["clean_rank_list"][p]) for p in selected_prompts
                ]
                prompt_example_rank_list = np.sum(
                    selected_prompt_rank_lists,
                    axis=0,
                )
                passing_indices = np.argwhere(prompt_example_rank_list == 0).squeeze()
                n_passing_indices = 0 if passing_indices.ndim == 0 else len(passing_indices)
                n_passing_per_prompt = [np.sum(p == 0) for p in selected_prompt_rank_lists]

                if n_passing_indices < args.min_passing_examples:
                    if split == "train":
                        logger.error(
                            f"Found {n_passing_indices} that pass all prompts in the {split} split (with {n_passing_per_prompt} for each prompt), min was set to {args.min_passing_examples}, aborting..."
                        )
                        _log_failure(
                            args.failure_path,
                            "shared_train_examples",
                            "Insufficient shared train examples for top prompts",
                        )

                    else:
                        if n_passing_indices == 0:
                            logger.error(
                                f"Found {n_passing_indices} that pass all prompts in the {split} split, this split requires at least one example, aborting..."
                            )
                            _log_failure(
                                args.failure_path, "shared_test_examples", "Zero shared test examples for top prompts"
                            )
                        else:
                            logger.warning(
                                f"Found {n_passing_indices} that pass all prompts in the {split} split, min was set to {args.min_passing_examples}..."
                            )
                filter_set_per_split[split] = passing_indices

        _gc_clear_cache()

        # Load or Re-Compute mean_head_activations, indirect_effect, and function vector
        fv = _mean_act_indirect_effect_fv(
            args,
            args.mean_activations_path,
            args.top_heads_path,
            args.fv_path,
            model,
            tokenizer,
            model_config,
            dataset,
            selected_prompts,
            filter_set_per_split,
            save_path_root,
            indirect_effect_path,
            baseline_generator_kwargs,
        )

        joint_fvs = None
        if add_prompt_fv_twice:
            joint_fvs = [fv, fv]

        if args.skip_eval:
            logger.info("Skipping evaluation and baseline (--skip_eval)")
            return

        # Run evaluation
        eval_identifer = f"universal_{args.universal_fv_type}_{n_top_heads}_heads" if universal_set else prompt_baseline

        if n_universal_only_flags > 0:
            results_file_suffix = "mini_sweep.json"
        elif isinstance(eval_edit_layer, int):
            results_file_suffix = f"editlayer_{eval_edit_layer}.json"
        else:
            results_file_suffix = "layer_sweep.json"

        zs_results_file_name = make_valid_path_name(
            f"{save_path_root}/zs_results_{eval_identifer}_{results_file_suffix}"
        )
        args.zs_results_file_name = zs_results_file_name

        if os.path.exists(zs_results_file_name) and not args.force_evaluation:
            logger.info("Skipping evaluation since file exists and flag was not set")

        else:
            # Run a two-argument sweep
            if add_prompt_fv_twice:
                min_depth = args.joint_intervention_min_layer_depth
                min_edit_layer = (
                    int(min_depth) if min_depth > 1 else int(np.floor(min_depth * model_config["n_layers"]))
                )
                max_depth = args.joint_intervention_max_layer_depth
                max_edit_layer = int(max_depth) if max_depth > 1 else int(np.ceil(max_depth * model_config["n_layers"]))
                args.min_edit_layer = min_edit_layer
                args.max_edit_layer = max_edit_layer
                if min_edit_layer >= max_edit_layer:
                    raise ValueError(
                        f"Minimum edit layer ({min_edit_layer}) must be less than maximum edit layer ({max_edit_layer})"
                    )
                logger.info(
                    f"Running `{args.universal_fv_type}` evaluation with edit_layer=[{min_edit_layer}, {max_edit_layer}] "
                )

                zs_results = {}
                layer_range = list(range(min_edit_layer, max_edit_layer + 1))
                for layers in itertools.product(layer_range, layer_range):
                    layers_str = "_".join([str(layer) for layer in layers])
                    zs_results[layers_str] = _run_zs_eval(
                        args,
                        joint_fvs,
                        layers,
                        model,
                        tokenizer,
                        model_config,
                        dataset,
                        filter_set_per_split,
                    )

            # Evaluate single layer
            elif isinstance(eval_edit_layer, int):
                logger.info(f"Running ZS Eval with edit_layer={eval_edit_layer}")
                zs_results = _run_zs_eval(
                    args,
                    fv,
                    eval_edit_layer,
                    model,
                    tokenizer,
                    model_config,
                    dataset,
                    filter_set_per_split,
                )

            # Sweep over layers
            else:
                logger.info(f"Running sweep over layers {eval_edit_layer}")
                zs_results = {}
                for edit_layer in range(eval_edit_layer[0], eval_edit_layer[1]):
                    zs_results[edit_layer] = _run_zs_eval(
                        args,
                        fv,
                        edit_layer,
                        model,
                        tokenizer,
                        model_config,
                        dataset,
                        filter_set_per_split,
                    )

            # Save results to files
            _protected_json_dump(zs_results_file_name, zs_results)

        if compute_baseline:
            baseline_file_name = f"{save_path_root}/model_icl_baseline.json"

            if os.path.exists(baseline_file_name) and not args.force_compute_baseline:
                logger.info("Skipping baseline since file exists and force flag is off")

            else:
                baseline_file_name = make_valid_path_name(baseline_file_name)
                args.baseline_file_name = baseline_file_name
                logger.info(f"Computing model baseline results for {n_eval_icl_examples}-shots")
                baseline_results = compute_dataset_baseline(
                    dataset,
                    model,
                    model_config,
                    tokenizer,
                    n_shots=n_eval_icl_examples,
                    seed=seed,
                    prefixes=prefixes,
                    separators=separators,
                    batch_size=args.few_shot_batch_size,
                )

                _protected_json_dump(baseline_file_name, baseline_results)

        logger.debug(f"Results saved to '{save_path_root}', saving arguments and terminating")

        # write args to file
        args_file_name = make_valid_path_name(f"{save_path_root}/{eval_identifer}_prompt_fv_eval_args.txt")
        _protected_json_dump(args_file_name, vars(args))

    except FailureException:
        pass

    finally:
        if model is not None:
            del model
        _gc_clear_cache()
        time.sleep(10)

    return


if __name__ == "__main__":
    prompt_function_vector_main()
