import json

import numpy as np
import torch
from baukit import TraceDict
from loguru import logger

# from .eval_utils import (
# )
# from .intervention_utils import (
# )
# from .model_utils import (
# )
# Include prompt creation helper functions
from recipe.function_vectors.utils.prompt_utils import (
    compute_duplicated_labels,
    get_dummy_token_labels,
    get_token_meta_labels,
    should_skip_prompt,
    word_pairs_to_prompt_data,
)
from recipe.function_vectors.utils.shared_utils import (
    tokenizer_padding_side_token,
)


# Attention Activations
def gather_attn_activations(prompt_data, layers, dummy_labels, model, tokenizer, model_config):
    """
    Collects activations for an ICL prompt

    Parameters:
    prompt_data: dict containing ICL prompt examples, and template information
    layers: layer names to get activatons from
    dummy_labels: labels and indices for a baseline prompt with the same number of example pairs
    model: huggingface model
    tokenizer: huggingface tokenizer

    Returns:
    td: tracedict with stored activations
    idx_map: map of token indices to respective averaged token indices
    idx_avg: dict containing token indices of multi-token words
    """

    # Get sentence and token labels
    query = prompt_data["query_target"]["input"]
    token_labels, prompt_string = get_token_meta_labels(
        prompt_data, tokenizer, query, prepend_bos=model_config["prepend_bos"]
    )
    sentence = [prompt_string]

    inputs = tokenizer(sentence, return_tensors="pt").to(model.device)
    idx_map, idx_avg = compute_duplicated_labels(token_labels, dummy_labels)

    # Access Activations
    with TraceDict(model, layers=layers, retain_input=True, retain_output=False) as td:
        model(**inputs)  # batch_size x n_tokens x vocab_size, only want last token prediction

    return td, idx_map, idx_avg


@tokenizer_padding_side_token
def gather_attn_activations_batch(batch_prompt_data, layers, dummy_labels, *, model, tokenizer, model_config):
    """
    Collects activations for an ICL prompt

    Parameters:
    batch_prompt_data: a list of dicts containing ICL prompt examples, and template information
    layers: layer names to get activatons from
    dummy_labels: labels and indices for a baseline prompt with the same number of example pairs
    model: huggingface model
    tokenizer: huggingface tokenizer

    Returns:
    td: tracedict with stored activations
    idx_map: map of token indices to respective averaged token indices
    idx_avg: dict containing token indices of multi-token words
    """

    # Get sentence and token labels
    sentences = []
    batch_idx_map = []
    batch_idx_avg = []
    for prompt_data in batch_prompt_data:
        token_labels, prompt_string = get_token_meta_labels(
            prompt_data,
            tokenizer,
            prompt_data["query_target"]["input"],
            prepend_bos=model_config["prepend_bos"],
        )
        idx_map, idx_avg = compute_duplicated_labels(token_labels, dummy_labels)
        sentences.append(prompt_string)
        batch_idx_map.append(idx_map)
        batch_idx_avg.append(idx_avg)

    inputs = tokenizer(sentences, return_tensors="pt", padding=True).to(model.device)
    # Access Activations
    with TraceDict(model, layers=layers, retain_input=True, retain_output=False) as td:
        # batch_size x n_tokens x vocab_size, only want last token prediction
        model(**inputs)

    return td, batch_idx_map, batch_idx_avg


def split_activations_by_head(activations, model_config):
    n_heads = model_config["n_heads"]
    head_dim = activations.size(-1) // n_heads
    new_shape = activations.size()[:-1] + (n_heads, head_dim)
    activations = activations.view(*new_shape)  # (batch_size, n_tokens, n_heads, head_dim)
    return activations


def get_mean_head_activations(
    dataset,
    model,
    model_config,
    tokenizer,
    n_icl_examples=10,
    N_TRIALS=100,
    shuffle_labels=False,
    prefixes=None,
    separators=None,
    filter_set=None,
    batch_size: int = 1,
):
    """
    Computes the average activations for each attention head in the model, where multi-token phrases are condensed into a single slot through averaging.

    Parameters:
    dataset: ICL dataset
    model: huggingface model
    model_config: contains model config information (n layers, n heads, etc.)
    tokenizer: huggingface tokenizer
    n_icl_examples: Number of shots in each in-context prompt
    N_TRIALS: Number of in-context prompts to average over
    shuffle_labels: Whether to shuffle the ICL labels or not
    prefixes: ICL template prefixes
    separators: ICL template separators
    filter_set: whether to only include samples the model gets correct via ICL

    Returns:
    mean_activations: avg activation of each attention head in the model taken across n_trials ICL prompts
    """

    n_test_examples = 1
    if prefixes is not None and separators is not None:
        dummy_labels = get_dummy_token_labels(
            n_icl_examples,
            tokenizer=tokenizer,
            prefixes=prefixes,
            separators=separators,
            model_config=model_config,
        )
    else:
        dummy_labels = get_dummy_token_labels(n_icl_examples, tokenizer=tokenizer, model_config=model_config)
    activation_storage = torch.zeros(
        N_TRIALS,
        model_config["n_layers"],
        model_config["n_heads"],
        len(dummy_labels),
        # model_config["resid_dim"] // model_config["n_heads"],
        model_config["head_dim"],
    )

    if filter_set is None:
        filter_set = np.arange(len(dataset["valid"]))

    # If the model already prepends a bos token by default, we don't want to add one
    prepend_bos = False if model_config["prepend_bos"] else True
    batch_prompt_data = []

    for n in range(N_TRIALS):
        word_pairs = dataset["train"][np.random.choice(len(dataset["train"]), n_icl_examples, replace=False)]
        word_pairs_test = dataset["valid"][np.random.choice(filter_set, n_test_examples, replace=False)]
        if prefixes is not None and separators is not None:
            prompt_data = word_pairs_to_prompt_data(
                word_pairs,
                query_target_pair=word_pairs_test,
                prepend_bos_token=prepend_bos,
                shuffle_labels=shuffle_labels,
                prefixes=prefixes,
                separators=separators,
                tokenizer=tokenizer,
            )
        else:
            prompt_data = word_pairs_to_prompt_data(
                word_pairs,
                query_target_pair=word_pairs_test,
                prepend_bos_token=prepend_bos,
                shuffle_labels=shuffle_labels,
                tokenizer=tokenizer,
            )

        if batch_size > 1:
            batch_prompt_data.append(prompt_data)
            if (len(batch_prompt_data) >= batch_size) or (n == N_TRIALS - 1):
                activations_td, batch_idx_map, batch_idx_avg = gather_attn_activations_batch(
                    batch_prompt_data=batch_prompt_data,
                    layers=model_config["attn_hook_names"],
                    dummy_labels=dummy_labels,
                    model=model,
                    tokenizer=tokenizer,
                    model_config=model_config,
                )

                stack_initial = torch.stack(
                    [
                        split_activations_by_head(activations_td[layer].input, model_config)
                        for layer in model_config["attn_hook_names"]
                    ],
                    dim=1,
                ).permute(0, 1, 3, 2, 4)

                stack_filtered = stack_initial[:, :, :, list(batch_idx_map[0].keys())]
                bs = len(batch_prompt_data)
                if stack_filtered.shape != (bs, *activation_storage.shape[2:]):
                    raise ValueError(
                        f"Expected shape {(bs, *activation_storage.shape[2:])}, got {stack_filtered.shape}"
                    )

                for b, (idx_avg, idx_map) in enumerate(zip(batch_idx_avg, batch_idx_map)):
                    for i, j in idx_avg.values():
                        stack_filtered[b, :, :, idx_map[i]] = stack_initial[b, :, :, i : j + 1].mean(
                            dim=-2
                        )  # average over duplicated tokens

                n_start = n - bs + 1
                n_end = n + 1
                activation_storage[n_start:n_end] = stack_filtered
                batch_prompt_data = []

        else:
            activations_td, idx_map, idx_avg = gather_attn_activations(
                prompt_data=prompt_data,
                layers=model_config["attn_hook_names"],
                dummy_labels=dummy_labels,
                model=model,
                tokenizer=tokenizer,
                model_config=model_config,
            )

            stack_initial = torch.vstack(
                [
                    split_activations_by_head(activations_td[layer].input, model_config)
                    for layer in model_config["attn_hook_names"]
                ]
            ).permute(0, 2, 1, 3)
            stack_filtered = stack_initial[:, :, list(idx_map.keys())]
            for i, j in idx_avg.values():
                stack_filtered[:, :, idx_map[i]] = stack_initial[:, :, i : j + 1].mean(
                    dim=-2
                )  # Average activations of multi-token words across all its tokens

            activation_storage[n] = stack_filtered

    mean_activations = activation_storage.mean(dim=0)
    return mean_activations




def _build_dummy_labels_with_prompt(tokenizer, model_config, prefixes, separators, n_icl_examples=0):
    if prefixes is not None and separators is not None:
        dummy_labels = get_dummy_token_labels(
            n_icl_examples=n_icl_examples,
            tokenizer=tokenizer,
            instructions="a",
            prefixes=prefixes,
            separators=separators,
            model_config=model_config,
        )
    else:
        dummy_labels = get_dummy_token_labels(
            n_icl_examples=n_icl_examples,
            tokenizer=tokenizer,
            instructions="a",
            model_config=model_config,
        )
    return dummy_labels


def get_prompt_based_mean_head_activations(
    dataset,
    prompts,
    model,
    model_config,
    tokenizer,
    n_trials_per_prompt=20,
    # n_icl_examples = 10, N_TRIALS = 100, shuffle_labels=False,
    prefixes=None,
    separators=None,
    filter_set=None,
    n_icl_examples=0,
    shuffle_icl_labels=False,
    query_dataset="train",
    batch_size: int = 1,
):
    """
    Computes the average activations for each attention head in the model, where multi-token phrases are condensed into a single slot through averaging.

    Parameters:
    dataset: ICL dataset
    model: huggingface model
    model_config: contains model config information (n layers, n heads, etc.)
    tokenizer: huggingface tokenizer
    n_icl_examples: Number of shots in each in-context prompt
    N_TRIALS: Number of in-context prompts to average over
    -- shuffle_labels: Whether to shuffle the ICL labels or not
    prefixes: ICL template prefixes
    separators: ICL template separators
    filter_set: whether to only include samples the model gets correct with the prompt

    Returns:
    mean_activations: avg activation of each attention head in the model taken across n_trials ICL prompts
    """
    if batch_size > n_trials_per_prompt:
        logger.warning(
            f"Batch size {batch_size} is greater than n_trials_per_prompt {n_trials_per_prompt}, setting batch size to {n_trials_per_prompt}"
        )
        batch_size = n_trials_per_prompt

    n_test_examples = 1

    # We can get away without lpadding everything, but we need to do some trickery on the dummy labels
    # Specifically, we'll keep the first and last instructions tokens, and discard the rest, since
    # we wouldn't really have a meaningful way to patch over them
    dummy_labels = _build_dummy_labels_with_prompt(
        tokenizer,
        model_config,
        prefixes,
        separators,
        n_icl_examples=n_icl_examples,
    )
    activation_storage = torch.zeros(
        len(prompts),
        n_trials_per_prompt,
        model_config["n_layers"],
        model_config["n_heads"],
        len(dummy_labels),
        # model_config["resid_dim"] // model_config["n_heads"],
        model_config["head_dim"],
    )

    for p, prompt in enumerate(prompts):
        # dummy_labels = _build_dummy_labels_with_prompt(
        #     tokenizer,
        #     prompt,
        #     model_config,
        #     prefixes,
        #     separators,
        #     n_icl_examples=n_icl_examples,
        # )
        # filtered_dummy_labels, skip_indices = first_instruction_token_only(dummy_labels)

        if filter_set is None:
            logger.warning(
                f"`get_prompt_based_mean_head_activations` called with no filter set, sampling from all {query_dataset} examples"
            )
            filter_set = np.arange(len(dataset[query_dataset]))

        fs = filter_set
        if isinstance(filter_set, dict):
            if prompt not in filter_set:
                error_message = f"`get_prompt_based_mean_head_activations` called with dictionary filter set, but prompt '{prompt}' not in it"
                logger.error(error_message)
                raise ValueError(error_message)

            fs = filter_set[prompt]

        # If the model already prepends a bos token by default, we don't want to add one
        prepend_bos = False if model_config["prepend_bos"] else True
        batch_prompt_data = []

        for n in range(n_trials_per_prompt):
            skip = True
            while skip:
                if n_icl_examples == 0:
                    if shuffle_icl_labels:
                        raise ValueError(
                            "Cannot providee shuffle_icl_labels = True and n_icl_examples = 0 (meaningless)"
                        )
                    word_pairs = {"input": [], "output": []}
                    word_pairs_query = dataset[query_dataset][np.random.choice(fs, n_test_examples, replace=False)]

                else:
                    if query_dataset == "train":
                        raise ValueError("Query dataset cannot be train when providing n_icl_examples != 0")

                    word_pairs = dataset["train"][
                        np.random.choice(len(dataset["train"]), n_icl_examples, replace=False)
                    ]
                    word_pairs_query = dataset["valid"][np.random.choice(fs, n_test_examples, replace=False)]

                wpq = word_pairs_query
                if isinstance(wpq, list):
                    wpq = wpq[0]

                target = wpq["output"]
                if isinstance(target, list):
                    target = target[0]

                skip = should_skip_prompt(target, prompt)

            if prefixes is not None and separators is not None:
                prompt_data = word_pairs_to_prompt_data(
                    word_pairs,
                    query_target_pair=word_pairs_query,
                    prepend_bos_token=prepend_bos,
                    shuffle_labels=shuffle_icl_labels,
                    instructions=prompt,
                    prefixes=prefixes,
                    separators=separators,
                    tokenizer=tokenizer,
                )
            else:
                prompt_data = word_pairs_to_prompt_data(
                    word_pairs,
                    query_target_pair=word_pairs_query,
                    prepend_bos_token=prepend_bos,
                    shuffle_labels=shuffle_icl_labels,
                    instructions=prompt,
                    tokenizer=tokenizer,
                )

            if batch_size > 1:
                batch_prompt_data.append(prompt_data)
                if (len(batch_prompt_data) >= batch_size) or (n == n_trials_per_prompt - 1):
                    batch_activations_td, batch_idx_map, batch_idx_avg = gather_attn_activations_batch(
                        batch_prompt_data=batch_prompt_data,
                        layers=model_config["attn_hook_names"],
                        dummy_labels=dummy_labels,
                        model=model,
                        tokenizer=tokenizer,
                        model_config=model_config,
                    )

                    stack_initial = torch.stack(
                        [
                            split_activations_by_head(batch_activations_td[layer].input, model_config)
                            for layer in model_config["attn_hook_names"]
                        ],
                        dim=1,
                    ).permute(0, 1, 3, 2, 4)

                    batch_idxs = (
                        torch.arange(len(batch_idx_map)).unsqueeze(1).repeat_interleave(len(batch_idx_map[0]), 1)
                    )
                    batch_token_idxs = torch.tensor([list(b.keys()) for b in batch_idx_map])
                    stack_filtered = stack_initial[batch_idxs, :, :, batch_token_idxs, :].permute(0, 2, 3, 1, 4)

                    # stack_filtered = stack_initial[:, :, :, list(batch_idx_map[0].keys())]
                    bs = len(batch_prompt_data)
                    if stack_filtered.shape != (bs, *activation_storage.shape[2:]):
                        raise ValueError(
                            f"Expected shape {(bs, *activation_storage.shape[2:])}, got {stack_filtered.shape}"
                        )

                    for b, (idx_avg, idx_map) in enumerate(zip(batch_idx_avg, batch_idx_map)):
                        for i, j in idx_avg.values():
                            stack_filtered[b, :, :, idx_map[i]] = stack_initial[b, :, :, i : j + 1].mean(
                                dim=-2
                            )  # average over duplicated tokens

                    n_start = n - bs + 1
                    n_end = n + 1
                    activation_storage[p, n_start:n_end] = stack_filtered
                    batch_prompt_data = []

            else:  # batch_size == 1
                activations_td, idx_map, idx_avg = gather_attn_activations(
                    prompt_data=prompt_data,
                    layers=model_config["attn_hook_names"],
                    dummy_labels=dummy_labels,
                    model=model,
                    tokenizer=tokenizer,
                    model_config=model_config,
                )

                stack_initial = torch.vstack(
                    [
                        split_activations_by_head(activations_td[layer].input, model_config)
                        for layer in model_config["attn_hook_names"]
                    ]
                ).permute(0, 2, 1, 3)
                stack_filtered = stack_initial[:, :, list(idx_map.keys())]

                if stack_filtered.shape != activation_storage.shape[2:]:
                    raise ValueError(f"Expected shape {activation_storage.shape[2:]}, got {stack_filtered.shape}")

                for i, j in idx_avg.values():
                    stack_filtered[:, :, idx_map[i]] = stack_initial[:, :, i : j + 1].mean(
                        dim=-2
                    )  # Average activations of multi-token words across all its tokens

                activation_storage[p, n] = stack_filtered

    # Sanity check for my aggregation logic
    non_zeros = (activation_storage != 0).to(torch.int)
    non_zeros = non_zeros.sum(dim=(2, 3, -1))
    idxs = torch.where(non_zeros == 0)
    if idxs[0].numel() > 0:
        raise ValueError(f"Found at least one entry with all zeros: {idxs}")

    mean_activations = activation_storage.mean(dim=(0, 1))
    return mean_activations




# Attention Weights
def compute_function_vector(
    mean_activations,
    indirect_effect,
    model,
    model_config,
    n_top_heads=10,
    token_class_idx=-1,
    prompt_based: bool = False,
):
    """
    Computes a "function vector" vector that communicates the task observed in ICL examples used for downstream intervention.

    Parameters:
    mean_activations: tensor of size (Layers, Heads, Tokens, head_dim) containing the average activation of each head for a particular task
    indirect_effect: tensor of size (N, Layers, Heads, class(optional)) containing the indirect_effect of each head across N trials
    model: huggingface model being used
    model_config: contains model config information (n layers, n heads, etc.)
    n_top_heads: The number of heads to use when computing the summed function vector
    token_class_idx: int indicating which token class to use, -1 is default for last token computations

    Returns:
    function_vector: vector representing the communication of a particular task
    top_heads: list of the top influential heads represented as tuples [(L,H,S), ...], (L=Layer, H=Head, S=Avg. Indirect Effect Score)
    """
    model_resid_dim = model_config["resid_dim"]
    model_head_dim = model_config["head_dim"]
    attn_in_dim = model_config["n_heads"] * model_head_dim
    device = model.device

    li_dims = len(indirect_effect.shape)
    pb = int(prompt_based)
    dim = (0, 1) if prompt_based else 0

    if li_dims == (3 + pb) and token_class_idx == -1:
        mean_indirect_effect = indirect_effect.mean(dim=dim)
    else:
        assert li_dims == (4 + pb)
        mean_indirect_effect = indirect_effect[:, :, :, token_class_idx].mean(
            dim=dim
        )  # Subset to token class of interest

    # Compute Top Influential Heads (L,H)
    h_shape = mean_indirect_effect.shape
    topk_vals, topk_inds = torch.topk(mean_indirect_effect.view(-1), k=n_top_heads, largest=True)
    top_lh = list(
        zip(
            *np.unravel_index(topk_inds, h_shape),
            [round(x.item(), 4) for x in topk_vals],
        )
    )
    top_heads = top_lh[:n_top_heads]

    # Compute Function Vector as sum of influential heads
    function_vector = torch.zeros((1, 1, model_resid_dim)).to(device)
    T = -1  # Intervention & values taken from last token

    for L, H, _ in top_heads:
        if "gpt2-xl" in model_config["name_or_path"]:
            out_proj = model.transformer.h[L].attn.c_proj
        elif "gpt-j" in model_config["name_or_path"]:
            out_proj = model.transformer.h[L].attn.out_proj
        elif any(name in model_config["name_or_path"].lower() for name in ("llama", "gemma", "olmo", "qwen")):
            out_proj = model.model.layers[L].self_attn.o_proj
        elif "gpt-neox" in model_config["name_or_path"] or "pythia" in model_config["name_or_path"]:
            out_proj = model.gpt_neox.layers[L].attention.dense
        else:
            raise ValueError(f"Unsupported model in `compute_function_vector`: {model_config['name_or_path']}")

        x = torch.zeros(attn_in_dim)
        x[H * model_head_dim : (H + 1) * model_head_dim] = mean_activations[L, H, T]
        d_out = out_proj(x.reshape(1, 1, attn_in_dim).to(device).to(model.dtype))

        function_vector += d_out

    function_vector = function_vector.to(model.dtype)
    function_vector = function_vector.reshape(1, model_resid_dim)

    return function_vector, top_heads




def compute_universal_function_vector_top_heads_from_file(
    mean_activations,
    model,
    model_config,
    top_heads_path,
    n_top_heads,
):
    """
    Computes a "function vector" vector that communicates the task observed in ICL examples used for downstream intervention
    using the set of heads with universally highest causal effect computed across a set of ICL tasks

    Parameters:
    mean_activations: tensor of size (Layers, Heads, Tokens, head_dim) containing the average activation of each head for a particular task
    model: huggingface model being used
    model_config: contains model config information (n layers, n heads, etc.)
    n_top_heads: The number of heads to use when computing the function vector

    Returns:
    function_vector: vector representing the communication of a particular task
    top_heads: list of the top influential heads represented as tuples [(L,H,S), ...], (L=Layer, H=Head, S=Avg. Indirect Effect Score)
    """
    model_resid_dim = model_config["resid_dim"]
    model_head_dim = model_config["head_dim"]
    attn_in_dim = model_config["n_heads"] * model_head_dim
    device = model.device

    # Universal Set of Heads
    with open(top_heads_path, "r") as f:
        top_heads_info = json.load(f)

    file_top_heads = top_heads_info["top_heads"]
    if len(file_top_heads) < n_top_heads:
        raise ValueError(
            f"Top heads file contains only {len(file_top_heads)} heads, but n_top_heads is set to {n_top_heads}"
        )

    top_heads = file_top_heads[:n_top_heads]

    # Compute Function Vector as sum of influential heads
    function_vector = torch.zeros((1, 1, model_resid_dim)).to(device)
    T = -1  # Intervention & values taken from last token

    for L, H in top_heads:  # we don't include the effects here, they're in another key in the JSON
        if "gpt2-xl" in model_config["name_or_path"]:
            out_proj = model.transformer.h[L].attn.c_proj
        elif "gpt-j" in model_config["name_or_path"]:
            out_proj = model.transformer.h[L].attn.out_proj
        elif any(name in model_config["name_or_path"].lower() for name in ("llama", "gemma", "olmo", "qwen")):
            out_proj = model.model.layers[L].self_attn.o_proj
        elif "gpt-neox" in model_config["name_or_path"]:
            out_proj = model.gpt_neox.layers[L].attention.dense

        x = torch.zeros(attn_in_dim)
        x[H * model_head_dim : (H + 1) * model_head_dim] = mean_activations[L, H, T]
        d_out = out_proj(x.reshape(1, 1, attn_in_dim).to(device).to(model.dtype))

        function_vector += d_out
        function_vector = function_vector.to(model.dtype)
    function_vector = function_vector.reshape(1, model_resid_dim)

    return function_vector, top_heads
