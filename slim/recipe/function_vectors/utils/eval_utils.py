import copy
import itertools
import os
import typing

import numpy as np
import torch
from loguru import logger
from torch.nn import functional as F
from tqdm import tqdm
from transformers import DynamicCache
from transformers.tokenization_utils_base import BatchEncoding

from recipe.function_vectors.utils.intervention_utils import (
    batch_function_vector_intervention,
    function_vector_intervention,
)
from recipe.function_vectors.utils.model_utils import set_seed
from recipe.function_vectors.utils.prompt_utils import (
    create_prompt,
    should_skip_prompt,
    word_pairs_to_prompt_data,
)
from recipe.function_vectors.utils.shared_utils import (
    EvalDataResults,
    tokenizer_padding_side_token,
)


def compute_top_k_accuracy(target_token_ranks, k=10) -> float:
    """
    Evaluation to compute topk accuracy.

    Parameters:
    target_token_ranks: the distribution of output token ranks
    k: how many tokens we're looking at (top K)

    Return:
    The accuracy of the token in the top k of tokens
    """

    target_token_ranks = np.array(target_token_ranks)
    return (target_token_ranks < k).sum(axis=0) / len(target_token_ranks)


def compute_individual_token_rank(prob_dist, target_id) -> int:
    """
    Individual computation of token ranks across a single distribution.

    Parameters:
    prob_dist: the distribution of scores for a single output
    target_id: the target id we care about

    Return:
    A single value representing the token rank for that single token
    """
    if isinstance(target_id, list):
        target_id = target_id[0]

    return torch.where(  # type: ignore
        torch.argsort(prob_dist.squeeze(), descending=True) == target_id
    )[0].item()


def get_answer_id(query, answer, tokenizer):
    """
    Parameters:
    query (str): query as a string
    answer (str): expected answer as a string
    tokenizer: huggingface tokenizer

    Returns:
    answer_ids (list): A list of the contextualized tokens of the answer
    """
    source = tokenizer(query, truncation=False, padding=False).input_ids
    target = tokenizer(query + answer, truncation=False, padding=False).input_ids
    assert len(source) < len(target) < tokenizer.model_max_length
    answer_ids = target[len(source) :]
    return answer_ids


def compute_dataset_baseline(
    dataset,
    model,
    model_config,
    tokenizer,
    n_shots=10,
    seed=42,
    prefixes=None,
    separators=None,
    batch_size=1,
) -> dict:
    """
    Computes the ICL performance of the model on the provided dataset for a varying number of shots.

    Parameters:
    dataset: ICL dataset
    model: huggingface model
    model_config: contains model config information (n layers, n heads, etc.)
    tokenizer: huggingface tokenizer
    n_shots: The upper bound of ICL examples to be used when evaluating the ICL performance of the model
    seed: seed for determining dataset split

    Returns:
    results_dict: dictionary containing the ICL performance results as the number of shots in ICL prompts varies.
    """
    results_dict = {}
    for n in range(n_shots + 1):
        set_seed(seed)
        results_dict[n] = n_shot_eval_no_intervention(
            dataset,
            n_shots=n,
            model=model,
            model_config=model_config,
            tokenizer=tokenizer,
            prefixes=prefixes,
            separators=separators,
            batch_size=batch_size,
        )

    results_dict[f"shuffled_{n_shots}"] = n_shot_eval_no_intervention(
        dataset,
        n_shots=n_shots,
        model=model,
        model_config=model_config,
        tokenizer=tokenizer,
        prefixes=prefixes,
        separators=separators,
        shuffle_labels=True,
        batch_size=batch_size,
    )

    return results_dict


# Evaluate a sentence
@tokenizer_padding_side_token
def sentence_eval(
    eval_data: EvalDataResults,
    *,
    model,
    tokenizer,
    compute_nll=True,
    past_key_values=None,
    prefix_length_tokens=0,
) -> EvalDataResults:
    """
    Evaluate a single sentence completion for a model, comparing to the given target.

    Parameters:
    sentence: sentence to have the model process and predict
    target: expected response of the model
    model: huggingface model
    tokenizer: huggingface tokenizer
    compute_nll: whether to compute the negative log likelihood of a teacher-forced answer prompt (used for computing PPL)
    past_key_values: for compatability with the batch method, should never be provided
    prefix_length_tokens: for compatability with the batch method, should never be provided

    Returns:
    model output on the provided sentence
    """
    # Clean Run, No Intervention:
    if len(eval_data) != 1:
        raise ValueError("Expected a single sentence to evaluate.")

    if past_key_values is not None or prefix_length_tokens != 0:
        raise ValueError(f"Base `sentence_eval` does not support `past_key_values`, received {past_key_values}")

    sentence = eval_data.sentences
    target = eval_data.targets

    device = model.device
    inputs = tokenizer(sentence, return_tensors="pt").to(device)
    original_pred_idx = len(inputs.input_ids.squeeze()) - 1
    target_completion = "".join(sentence + target)
    nll_inputs = tokenizer(target_completion, return_tensors="pt").to(device)

    if not (nll_inputs.input_ids[:, : inputs.input_ids.shape[1]] == inputs.input_ids).all():
        logger.warning(
            "Adding the target completion changes at least one input token. This may lead to incorrect NLL computation."
        )

    if compute_nll:
        nll_targets = nll_inputs.input_ids.clone()
        input_length = len(inputs.input_ids.squeeze())

        while inputs.input_ids[:, input_length - 1] != nll_targets[:, input_length - 1]:
            input_length -= 1

        nll_targets[
            :, :input_length
        ] = -100  # This is the accepted value to skip indices when computing loss in nn.CrossEntropyLoss

        output = model(**nll_inputs, labels=nll_targets)
        eval_data.nlls = [output.loss.item()]
        eval_data.logits = output.logits[:, original_pred_idx, :]

    else:
        eval_data.logits = model(**inputs).logits[:, -1, :]

    return eval_data


@tokenizer_padding_side_token
def batch_sentence_eval(
    eval_data: EvalDataResults,
    *,
    model,
    tokenizer,
    compute_nll=True,
    past_key_values=None,
    prefix_length_tokens=0,
) -> EvalDataResults:
    """
    Evaluate a batch of sentence completions for a model, comparing to the given targets.
    Supports using cached prefix computations for efficiency.

    Parameters:
    sentences: list of sentences to have the model process and predict
    targets: list of expected responses of the model
    model: huggingface model
    tokenizer: huggingface tokenizer
    compute_nll: whether to compute the negative log likelihood of teacher-forced answer prompts (used for computing PPL)
    past_key_values: cached key/values from a previously computed prefix
    prefix_length_tokens: the length of the prefix in tokens (not characters)

    Returns:
    list of model outputs on the provided sentences
    """
    if len(eval_data) == 0:
        raise ValueError("No sentences provided for evaluation.")

    sentences = eval_data.sentences
    targets = eval_data.targets

    device = model.device

    if past_key_values is not None:
        batch_size = len(sentences)
        _cache_key = past_key_values.layers[0].keys if isinstance(past_key_values, DynamicCache) else past_key_values[0][0]
        cache_batch_size = _cache_key.size(0)
        if batch_size != cache_batch_size:
            logger.warning(f"Batch size {batch_size} does not match cache size {cache_batch_size}, so skipping cache.")
            past_key_values = None
        else:
            past_key_values = copy.deepcopy(past_key_values)

    full_tokenized_inputs = tokenizer(sentences, return_tensors="pt", padding=True).to(device)

    # Combine each sentence with its target
    target_completions = [sentence + target for sentence, target in zip(sentences, targets)]

    # Tokenize everything first
    full_tokenized_completions = tokenizer(target_completions, return_tensors="pt", padding=True).to(device)

    full_matches_input = (
        full_tokenized_completions.input_ids[:, : full_tokenized_inputs.input_ids.shape[1]]
        == full_tokenized_inputs.input_ids
    ) | (full_tokenized_inputs.input_ids == tokenizer.pad_token_id)

    if not full_matches_input.all():
        logger.warning(
            "Adding the target completion changes at least one input token. This may lead to incorrect NLL computation."
        )

    # For NLL computation with cached prefixes
    if past_key_values is not None:
        # Slice off the prefix tokens to get just the new tokens we need to process
        suffix_input_ids = full_tokenized_completions.input_ids[:, prefix_length_tokens:]
        suffix_attention_mask = full_tokenized_completions.attention_mask[:, prefix_length_tokens:]

        prefix_attention_mask = torch.ones(
            # Get length from KV cache
            (suffix_input_ids.size(0), _cache_key.size(2)),
            dtype=suffix_attention_mask.dtype,
            device=suffix_attention_mask.device,
        )

        full_attention_mask = torch.cat([prefix_attention_mask, suffix_attention_mask], dim=1)

        # Create new inputs dictionary with just the suffixes but with the correct full attention mask
        nll_inputs = BatchEncoding(
            {
                "input_ids": suffix_input_ids,
                "attention_mask": full_attention_mask,
            }
        )

        # Do the same for the input sentences (without targets)
        input_suffix_ids = full_tokenized_inputs.input_ids[:, prefix_length_tokens:]
        input_suffix_mask = full_tokenized_inputs.attention_mask[:, prefix_length_tokens:]

        # Create equivalent structure for original sliced inputs
        input_tokenized = BatchEncoding({"input_ids": input_suffix_ids, "attention_mask": input_suffix_mask})
    else:
        # Standard case without caching - use the full tokenized inputs
        nll_inputs = full_tokenized_completions
        input_tokenized = full_tokenized_inputs

    # Find pad positions in the input tokenized data
    pad_positions = torch.argmax((input_tokenized.input_ids == tokenizer.pad_token_id).to(int), dim=1)

    # Grab the output positions
    output_positions = [
        (input_tokenized.input_ids.shape[1] if pos.item() == 0 else pos.item()) - 1 for pos in pad_positions
    ]

    # Adjust for no padding case
    seq_lengths = []
    for i, pos in enumerate(pad_positions):
        if pos.item() == 0 and input_tokenized.input_ids[i, 0] != tokenizer.pad_token_id:
            # No padding found, use full length
            seq_lengths.append(input_tokenized.input_ids.shape[1])
        else:
            seq_lengths.append(pos.item())

    # Create labels tensor with -100 for input tokens (to be ignored in loss)
    nll_targets = nll_inputs.input_ids.clone()

    # Mask out the original sentence portions
    for i, input_length in enumerate(seq_lengths):
        valid_length = input_length
        # Ensure we're at the right boundary by checking token IDs
        while (
            valid_length > 0 and nll_targets[i, valid_length - 1] != input_tokenized.input_ids[i, valid_length - 1]
        ):
            valid_length -= 1
        nll_targets[i, :valid_length] = -100

    # Set pad tokens to ignore index
    nll_targets[nll_targets == tokenizer.pad_token_id] = -100

    # Forward pass with labels and cached prefix if available
    outputs = model(**nll_inputs, past_key_values=past_key_values, labels=nll_targets)

    # Get predictions at the end of input sequences
    eval_data.logits = outputs.logits[torch.arange(len(output_positions)), torch.tensor(output_positions), :]

    if compute_nll:
        # Calculate NLL for each example
        batch_nlls = []
        for i in range(len(sentences)):
            example_nll = F.cross_entropy(
                outputs.logits[i, :-1],
                nll_targets[i, 1:],
                ignore_index=-100,
                reduction="mean",
            )
            batch_nlls.append(example_nll.item())

        eval_data.nlls = batch_nlls

    return eval_data


# Evaluate few-shot dataset w/o intervention
def n_shot_eval_no_intervention(
    dataset,
    n_shots,
    model,
    model_config,
    tokenizer,
    compute_ppl=True,
    shuffle_labels=False,
    prefixes=None,
    separators=None,
    test_split="test",
    batch_size: int = 1,
):
    """
    Evaluate a model (without any interventions) on the provided ICL dataset.

    Parameters:
    dataset: ICL dataset
    n_shots: the number of ICL examples in each in-context prompt
    model: huggingface model
    model_config: contains model config information (n layers, n heads, etc.)
    tokenizer: huggingface tokenizer
    compute_ppl: whether to compute perplexity of teacher-forced correct completion for base model & intervened model
    shuffle_labels: Whether to shuffle the ICL labels or not
    prefixes: dict of ICL template prefixes for each ICL component (input, output, instructions)
    separators: dict of ICL template separators for each ICL component (input, output, instructions)
    test_split: the dataset test split to use as the "test" dataset, typically set to 'test' or 'valid'

    Returns:
    results: dict of topk (k=1,2,3) accuracy on the test_split dataset, for both the model's n-shot
    """
    clean_rank_list = []

    if compute_ppl:
        clean_nll_list = []

    def _process_batch(batch):
        if len(batch) > 0:
            eval_result = eval_method(
                batch,
                model=model,
                tokenizer=tokenizer,
                compute_nll=compute_ppl,
            )

            for i in range(len(batch)):
                if eval_result.logits is None:
                    raise ValueError("Logits not computed when they should be")

                target_token_id = get_answer_id(batch.sentences[i], batch.targets[i], tokenizer)
                clean_rank = compute_individual_token_rank(eval_result.logits[i], target_token_id)
                clean_rank_list.append(clean_rank)

                if compute_ppl:
                    if eval_result.nlls is None:
                        raise ValueError("NLLs not computed when they should be")

                    clean_nll_list.append(eval_result.nlls[i])

    # If the model already prepends a bos token by default, we don't want to add one
    prepend_bos = False if model_config["prepend_bos"] else True

    eval_method = batch_sentence_eval if (batch_size > 1) else sentence_eval
    current_batch = EvalDataResults()
    dataset_size = len(dataset[test_split])

    for j in tqdm(range(dataset_size), total=dataset_size):
        if n_shots == 0:
            word_pairs = {"input": [], "output": []}
        else:
            word_pairs = dataset["train"][np.random.choice(len(dataset["train"]), n_shots, replace=False)]
        word_pairs_test = dataset[test_split][j]
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

        # Get relevant parts of the Prompt
        query, target = (
            prompt_data["query_target"]["input"],
            prompt_data["query_target"]["output"],
        )
        query = query[0] if isinstance(query, list) else query
        target = target[0] if isinstance(target, list) else target

        # Figure out tokens of interest
        current_batch.append(create_prompt(prompt_data), target)

        if len(current_batch) >= batch_size:
            _process_batch(current_batch)
            current_batch = EvalDataResults()

    _process_batch(current_batch)
    current_batch = EvalDataResults()

    results = {
        "clean_topk": [(K, compute_top_k_accuracy(clean_rank_list, K)) for K in range(1, 4)],
        "clean_rank_list": clean_rank_list,
    }
    if compute_ppl:
        results["clean_ppl"] = np.exp(clean_nll_list).mean()

    return results


def prompt_based_eval(
    dataset,
    fv_vector_or_vectors,
    edit_layer_or_layers: int | typing.Sequence[int],
    prompts,
    model,
    model_config,
    tokenizer,
    filter_set=None,
    prefixes=None,
    separators=None,
    n_icl_examples: int = 0,
    shuffle_icl_labels: bool = False,
    query_dataset: str = "test",
    prepend_space_to_prefix=False,
    generic_input="TEST_INPUT",
    generic_output="TEST_OUTPUT",
    batch_size: int = 1,
):
    """
    Evaluate a model and FV intervention on the model using the provided ICL dataset.

    Parameters:
    dataset: ICL dataset
    function_vector: torch vector that triggers execution of a task when added to a particular layer
    edit_layer: layer index
    n_shots: the number of ICL examples in each in-context prompt
    prompts: the list of prompts to evaluate with
    model: huggingface model
    model_config: contains model config information (n layers, n heads, etc.)
    tokenizer: huggingface tokenizer
    shuffle_labels: Whether to shuffle the ICL labels or not
    filter_set: whether to only include samples the model gets correct via ICL
    prefixes: dict of ICL template prefixes for each ICL component (input, output, instructions)
    separators: dict of ICL template separators for each ICL component (input, output, instructions)

    Returns:
    results: dict of topk accuracy on the test dataset, for both the model's n-shot, and n-shot + FV intervention, as well as the token rank of each prediction
    """
    clean_ranks = {p: [] for p in prompts}
    intervention_ranks = {p: [] for p in prompts}

    # If the model already prepends a bos token by default, we don't want to add one
    prepend_bos = False if model_config["prepend_bos"] else True

    if filter_set is None:
        logger.warning(f"`prompt_based_eval` called with no filter set, sampling from all {query_dataset} examples")
        filter_set = np.arange(len(dataset[query_dataset]))

    word_pairs = {"input": [], "output": []}

    if n_icl_examples == 0:
        prefix = "Zero shot eval"
    else:
        prefix = f"{n_icl_examples}-shot{' shuffled ' if shuffle_icl_labels else ' '}eval"

    pbar = tqdm(total=len(prompts) * len(filter_set), desc=f"{prefix} edit layer {edit_layer_or_layers}")

    intervention_method = batch_function_vector_intervention if batch_size > 1 else function_vector_intervention
    dataset_size = len(dataset["test"])
    skip_count = 0

    def _process_batch(batch, prompt, terminal=False):
        if len(batch) > 0:
            clean_result, intervention_result = intervention_method(
                batch,
                edit_layer_or_layers=edit_layer_or_layers,
                function_vector_or_vectors=fv_vector_or_vectors,
                model=model,
                model_config=model_config,
                tokenizer=tokenizer,
                compute_nll=False,
            )

            for i in range(len(batch)):
                n_skipped = batch.skipped_indices.get(i, 0)
                for _ in range(n_skipped):
                    clean_ranks[prompt].append(np.nan)
                    intervention_ranks[prompt].append(np.nan)

                target_token_id = get_answer_id(clean_result.sentences[i], clean_result.targets[i], tokenizer)

                if clean_result.logits is None:
                    raise ValueError("Logits not computed in clean results when they should be")

                current_clean_rank = compute_individual_token_rank(clean_result.logits[i], target_token_id)

                if intervention_result.logits is None:
                    raise ValueError("Logits not computed in intervention results when they should be")

                current_intervention_rank = compute_individual_token_rank(
                    intervention_result.logits[i], target_token_id
                )

                clean_ranks[prompt].append(current_clean_rank)
                intervention_ranks[prompt].append(current_intervention_rank)

        n_final_skipped = batch.skipped_indices.get(len(batch), 0)
        if n_final_skipped > 0:
            if terminal:
                for _ in range(n_final_skipped):
                    clean_ranks[prompt].append(np.nan)
                    intervention_ranks[prompt].append(np.nan)
            else:
                # I _think_ this should never happen, but I want a sanity check
                raise ValueError("Skipped indices should never be at the end of a non-empty non-terminal batch")

    for p, prompt in enumerate(prompts):
        pbar.set_postfix({"Prompt #": p + 1})

        fs = filter_set
        if isinstance(filter_set, dict):
            if prompt not in filter_set:
                error_message = f"`compute_prompt_based_indirect_effect` called with dictionary filter set, but prompt '{prompt}' not in it"
                logger.error(error_message)
                raise ValueError(error_message)

            fs = filter_set[prompt]

        current_batch = EvalDataResults()

        for j in range(dataset_size):
            if j not in fs:
                continue

            pbar.update()

            if n_icl_examples == 0:
                word_pairs = {"input": [], "output": []}
            else:
                word_pairs = dataset["train"][np.random.choice(len(dataset["train"]), n_icl_examples, replace=False)]
            word_pairs_test = dataset[query_dataset][j]

            if prefixes is not None and separators is not None:
                prompt_data = word_pairs_to_prompt_data(
                    word_pairs,
                    instructions=prompt,
                    query_target_pair=word_pairs_test,
                    prepend_bos_token=prepend_bos,
                    shuffle_labels=shuffle_icl_labels,
                    prefixes=prefixes,
                    separators=separators,
                    prepend_space_to_prefix=prepend_space_to_prefix,
                    tokenizer=tokenizer,
                )
            else:
                prompt_data = word_pairs_to_prompt_data(
                    word_pairs,
                    instructions=prompt,
                    query_target_pair=word_pairs_test,
                    prepend_bos_token=prepend_bos,
                    shuffle_labels=shuffle_icl_labels,
                    tokenizer=tokenizer,
                )

            query, target = (
                prompt_data["query_target"]["input"],
                prompt_data["query_target"]["output"],
            )

            skip = len(prompt.strip()) > 0 and should_skip_prompt(target, prompt)
            skip_count += 1
            if skip:
                current_batch.skipped_indices[len(current_batch)] += 1
                continue

            query = query[0] if isinstance(query, list) else query
            target = str(target[0]) if isinstance(target, list) else str(target)

            sentence = create_prompt(prompt_data)
            current_batch.append(sentence, target)

            if len(current_batch) >= batch_size:
                _process_batch(current_batch, prompt)
                current_batch = EvalDataResults()

        _process_batch(current_batch, prompt, terminal=True)
        current_batch = EvalDataResults()

        phase = "zero_shot_eval" if n_icl_examples == 0 else "few_shot_shuffled_eval"
        log_data = {
            f"{phase}_edit_layer": edit_layer_or_layers,
            f"{phase}_prompt_index": p,
            f"{phase}_accuracy": compute_top_k_accuracy(intervention_ranks[prompt], k=1),
        }

    results = {
        "clean_topk": {
            prompt: [(K, compute_top_k_accuracy(prompt_clean_rank_list, K)) for K in range(1, 4)]
            for (prompt, prompt_clean_rank_list) in clean_ranks.items()
        },
        "clean_ranks": clean_ranks,
        "intervention_topk": {
            prompt: [(K, compute_top_k_accuracy(prompt_intervention_rank_list, K)) for K in range(1, 4)]
            for (
                prompt,
                prompt_intervention_rank_list,
            ) in intervention_ranks.items()
        },
        "intervention_ranks": intervention_ranks,
    }

    return results


# Evaluate few-shot dataset w/o intervention
def prompt_based_eval_no_intervention(
    dataset,
    prompts,
    model,
    model_config,
    tokenizer,
    compute_ppl=True,
    shuffle_labels=False,
    prefixes=None,
    separators=None,
    relevant_split="train",
    prepend_space_to_prefix=False,
    generic_input="TEST_INPUT",
    generic_output="TEST_OUTPUT",
    partial_path=None,
    ignore_overlap_skip_check=False,
    cache_prompt_prefixes=False,
    batch_size: int = 1,
):
    """
    Evaluate a model (without any interventions) on the provided ICL dataset.

    Parameters:
    dataset: ICL dataset
    prompts: the prompts to test against
    model: huggingface model
    model_config: contains model config information (n layers, n heads, etc.)
    tokenizer: huggingface tokenizer
    compute_ppl: whether to compute perplexity of teacher-forced correct completion for base model & intervened model
    shuffle_labels: Whether to shuffle the ICL labels or not
    prefixes: dict of ICL template prefixes for each ICL component (input, output, instructions)
    separators: dict of ICL template separators for each ICL component (input, output, instructions)
    relevant_split: the dataset test split to use as the "test" dataset, typically set to 'test' or 'valid'
    prepend_space_to_prefix: whether to prepend a space to the prefix
    generic_input: the generic input token to replace in the prompt (when generating a generic prefix to cache)
    generic_output: the generic output token to replace in the prompt (when generating a generic prefix to cache)
    partial_path: path to a partial results file to save/load and continue from
    cache_prompt_prefixes: whether to cache the prompt prefixes for each prompt
    batch_size: batch size for evaluation

    Returns:
    results: dict of topk (k=1,2,3) accuracy on the test_split dataset, for both the model's n-shot
    """
    if partial_path is not None and os.path.exists(partial_path):
        partial_results = torch.load(partial_path)
    else:
        partial_results = {
            "clean_ranks": {p: [] for p in prompts},
            "clean_nlls": {p: [] for p in prompts},
            "last_prompt_index": -1,
        }

    # If the model already prepends a bos token by default, we don't want to add one
    prepend_bos = False if model_config["prepend_bos"] else True

    word_pairs = {"input": [], "output": []}

    use_batch_method = (batch_size > 1) or cache_prompt_prefixes
    eval_method = batch_sentence_eval if use_batch_method else sentence_eval

    def _process_batch(batch, prompt, terminal=False):
        if len(batch) > 0:
            eval_result = eval_method(
                batch,
                model=model,
                tokenizer=tokenizer,
                compute_nll=compute_ppl,
                past_key_values=prompt_cache,  # type: ignore
                prefix_length_tokens=prefix_length_tokens,
            )

            for i in range(len(batch)):
                n_skipped = batch.skipped_indices.get(i, 0)
                for _ in range(n_skipped):
                    partial_results["clean_ranks"][prompt].append(np.nan)
                    if compute_ppl:
                        partial_results["clean_nlls"][prompt].append(np.nan)

                if eval_result.logits is None:
                    raise ValueError("Logits not computed when they should be")

                target_token_id = get_answer_id(batch.sentences[i], batch.targets[i], tokenizer)
                clean_rank = compute_individual_token_rank(eval_result.logits[i], target_token_id)
                partial_results["clean_ranks"][prompt].append(clean_rank)

                if compute_ppl:
                    if eval_result.nlls is None:
                        raise ValueError("NLLs not computed when they should be")

                    partial_results["clean_nlls"][prompt].append(eval_result.nlls[i])

        n_final_skipped = batch.skipped_indices.get(len(batch), 0)
        if n_final_skipped > 0:
            if terminal:
                for _ in range(n_final_skipped):
                    partial_results["clean_ranks"][prompt].append(np.nan)
                    if compute_ppl:
                        partial_results["clean_nlls"][prompt].append(np.nan)
            else:
                # I _think_ this should never happen, but I want a sanity check
                raise ValueError("Skipped indices should never be at the end of a non-empty non-terminal batch")

    for p, prompt in tqdm(enumerate(prompts), desc="Prompt", position=0, total=len(prompts)):
        if p <= partial_results["last_prompt_index"]:
            continue

        prefix_word_pairs = {"input": generic_input, "output": generic_output}
        if prefixes is not None and separators is not None:
            prompt_data = word_pairs_to_prompt_data(
                word_pairs,
                instructions=prompt,
                query_target_pair=prefix_word_pairs,
                prepend_bos_token=prepend_bos,
                shuffle_labels=shuffle_labels,
                prefixes=prefixes,
                separators=separators,
                prepend_space_to_prefix=prepend_space_to_prefix,
                tokenizer=tokenizer,
            )
        else:
            prompt_data = word_pairs_to_prompt_data(
                word_pairs,
                instructions=prompt,
                query_target_pair=prefix_word_pairs,
                prepend_bos_token=prepend_bos,
                shuffle_labels=shuffle_labels,
                tokenizer=tokenizer,
            )

        # Important in case we prepend a space here
        preprocessed_generic_output = prompt_data["query_target"]["output"]
        prompt_sentence = create_prompt(prompt_data)
        prefix_end_index = prompt_sentence.find(" " + generic_input)
        prefix_to_cache = prompt_sentence[:prefix_end_index]
        prefix_length_tokens = 0

        prompt_cache = None
        if cache_prompt_prefixes:
            with torch.no_grad():
                inputs_prefix_to_cache = tokenizer([prefix_to_cache] * batch_size, return_tensors="pt").to(model.device)
                prompt_cache = DynamicCache()
                prompt_cache = model(**inputs_prefix_to_cache, past_key_values=prompt_cache).past_key_values
                prefix_length_tokens = inputs_prefix_to_cache.input_ids.shape[-1]

        skip_count = 0
        current_batch = EvalDataResults()

        dataset_size = len(dataset[relevant_split])
        for j in tqdm(
            range(dataset_size),
            total=dataset_size,
            desc="Example",
            position=1,
            leave=False,
        ):
            # Get relevant parts of the Prompt
            query, target = (
                dataset[relevant_split][j]["input"],
                dataset[relevant_split][j]["output"],
            )

            skip = (not ignore_overlap_skip_check) and should_skip_prompt(target, prompt)
            if skip:
                skip_count += 1
                current_batch.skipped_indices[len(current_batch)] += 1
                continue

            query = query[0] if isinstance(query, list) else query
            target = target[0] if isinstance(target, list) else target

            if isinstance(target, (str, int)):
                target = preprocessed_generic_output.replace(generic_output, str(target))
            else:
                target = [preprocessed_generic_output.replace(generic_output, str(t)) for t in target]

            sentence = prompt_sentence.replace("TEST_INPUT", query)
            current_batch.append(sentence, target)

            if len(current_batch) >= batch_size:
                _process_batch(current_batch, prompt)
                current_batch = EvalDataResults()

        _process_batch(current_batch, prompt, terminal=True)
        current_batch = EvalDataResults()

        log_data = {
            "prompt_eval_prompt_index": p,
            f"prompt_eval_{relevant_split}_accuracy": compute_top_k_accuracy(
                partial_results["clean_ranks"][prompt], k=1
            ),
            f"prompt_eval_{relevant_split}_skip_count": skip_count,
        }

        partial_results["last_prompt_index"] += 1
        if partial_path is not None:
            torch.save(partial_results, partial_path)

    results = {
        "clean_topk": {
            prompt: [(K, compute_top_k_accuracy(prompt_clean_rank_list, K)) for K in range(1, 4)]
            for (prompt, prompt_clean_rank_list) in partial_results["clean_ranks"].items()
        },
        "clean_rank_list": partial_results["clean_ranks"],
    }
    if compute_ppl:
        results["clean_ppl"] = {
            prompt: np.exp(prompt_clean_nll_list).mean()
            for (prompt, prompt_clean_nll_list) in partial_results["clean_nlls"].items()
        }

    return results



def make_valid_path_name(path: str):
    """
    Returns an updated path name if given name already exists
    """
    file_name, extension = os.path.splitext(path)
    counter = 1

    while os.path.exists(path):
        path = file_name + "_" + str(counter) + extension
        counter += 1

    return path

