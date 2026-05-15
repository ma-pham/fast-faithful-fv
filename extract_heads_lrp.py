import os
import random
import torch
from transformers import AutoTokenizer
import time
import json

from eval.utils import load_dataset, word_pairs_to_prompt_data, create_prompt,  top_prompts, train_filter_set
from lxt.efficient import monkey_patch
from prompt_utils import get_answer_id, rank_of_token

# ==========================================
# 1. Config & Setup
# ==========================================

# todo param and implelemtation check w paper and repo

STORAGE_ROOT = "storage"
INTERMEDIATE_PATH = os.path.join(STORAGE_ROOT, "function_vectors", "full_results_prompt_based_short")
DATASET_PATH = os.path.join(STORAGE_ROOT, "function_vectors", "dataset_files")
RESULTS_PATH =  os.path.join(STORAGE_ROOT, "eval")

# MODEL_SHORT = "Llama-3.2-3B-Instruct"
# MODEL_SHORT = "Llama-3.1-8B-Instruct"
MODEL_SHORT="Qwen3-4B-Instruct-2507"

#TRAIN_SPLIT = 0.7
TEST_SPLIT = 0.3
top_k_filter = 5


PREFIXES = {"input": "Q:", "output": "A:", "instructions": ""}
SEPARATORS = {"input": "\n", "output": "\n\n", "instructions": ""}
PROMPT_SEPARATORS = {"input": "\n", "output": "\n\n", "instructions": "\n"}

RANDOM_SEED = 42
random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

device = torch.device("cuda:0" if torch.cuda.is_available() else ("mps" if torch.mps.is_available() else "cpu"))

if MODEL_SHORT == "Llama-3.2-3B-Instruct":
    model_name= "llama3b"
    from transformers.models.llama import modeling_llama
    hf_model_id = "meta-llama/Llama-3.2-3B-Instruct"
    model = modeling_llama.LlamaForCausalLM.from_pretrained(
        hf_model_id,
        device_map=device,
        torch_dtype=torch.bfloat16,
        attn_implementation="eager",
    )
    monkey_patch(modeling_llama, verbose=True)


elif MODEL_SHORT == "Llama-3.1-8B-Instruct" :
    model_name= "llama8b"
    from transformers.models.llama import modeling_llama
    hf_model_id = "meta-llama/Llama-3.1-8B-Instruct"
    model = modeling_llama.LlamaForCausalLM.from_pretrained(
        hf_model_id,
        device_map=device,
        torch_dtype=torch.bfloat16,
        attn_implementation="eager",
    )
    monkey_patch(modeling_llama, verbose=True)


else:
    model_name= "qwen4b"
    from transformers.models.qwen3 import modeling_qwen3
    hf_model_id = "Qwen/Qwen3-4B-Instruct-2507"
    model = modeling_qwen3.Qwen3ForCausalLM.from_pretrained(
        hf_model_id,
        device_map=device,
        torch_dtype=torch.bfloat16,
        attn_implementation="eager",
    )
    monkey_patch(modeling_qwen3, verbose=True)


tokenizer = AutoTokenizer.from_pretrained(hf_model_id, fix_mistral_regex=True)

for param in model.parameters():
    param.requires_grad = False

n_heads = model.config.num_attention_heads
num_layers = model.config.num_hidden_layers
head_dim = getattr(model.config, "head_dim", model.config.hidden_size // n_heads)


# TODO deal with nonsense tasks, where only one is nonsensne
# TODO check intentional frame


tasks = [
    # abstractive
    "antonym", "capitalize", "capitalize_first_letter", "capitalize_last_letter", "capitalize_second_letter",
    "country-capital", "country-currency", "english-french", "english-german", "english-spanish",
    "landmark-country", "lowercase_first_letter", "lowercase_last_letter", "national_parks",
    "next_capital_letter", "next_item", "park-country", "present-past", "prev_item",
    "product-company", "singular-plural", "synonym", "word_length",
    # extractive
    "adjective_v_verb_3", "adjective_v_verb_5",
    #"alphabetically_first_3", # removed
    #"alphabetically_first_5", # removed
    "alphabetically_last_3",
    #"alphabetically_last_5", #removed
    "animal_v_object_3", "animal_v_object_5",
    "choose_first_of_3", "choose_first_of_5",
    "choose_last_of_3", "choose_last_of_5",
    "choose_middle_of_3", "choose_middle_of_5",
    "color_v_animal_3", "color_v_animal_5",
    "concept_v_object_3", "concept_v_object_5",
    "conll2003_location", "conll2003_organization", "conll2003_person",
    "fruit_v_animal_3", "fruit_v_animal_5",
    "object_v_concept_3", "object_v_concept_5",
    #"squad_val",
    "verb_v_adjective_3", "verb_v_adjective_5",
]


formatted_tasks = {
    "antonym": {
        "llama3b": ["opposite", "antonym"],
        "qwen4b": ["opposite", "contrasting"],
        "llama8b": ["opposite", "opposing", "contrasting"],
        "omit": False
    },
    "capitalize": {
        "llama3b": ["title case"],
        "qwen4b": ["capitalize", "capitalization"],
        "llama8b": ["title case", "Capitalize", "Uppercase", "capital"],
        "omit": False
    },
    "capitalize_first_letter": {
        "llama3b": ["initial uppercase letter", "letter that is at the beginning", "first letter", "starting letter"],
        "qwen4b": ["initial uppercase letter", "initial letter", "first letter"],
        "llama8b": ["initial uppercase letter", "first letter", "initial letter"],
        "omit": False
    },
    "capitalize_last_letter": {
        "llama3b": ["last letter", "ending letter", "final letter"],
        "qwen4b": ["last letter", "terminal letter", "last alphabe"],
        "llama8b": ["last letter", "letter", "end"],
        "omit": False
    },
    "capitalize_second_letter": {
        "llama3b": ["f"],
        "qwen4b": ["f"],
        "llama8b": ["f"],
        "omit": True
    },
    "country-capital": {
        "llama3b": ["capital"],
        "qwen4b": ["capital", "capitals"],
        "llama8b": ["capital"],
        "omit": False
    },
    "country-currency": {
        "llama3b": ["currency"],
        "qwen4b": ["currency", "currencies"],
        "llama8b": ["currency", "currencies"],
        "omit": False
    },
    "english-french": {
        "llama3b": ["French"],
        "qwen4b": ["french"],
        "llama8b": ["French"],
        "omit": False
    },
    "english-german": {
        "llama3b": ["German"],
        "qwen4b": ["german"],
        "llama8b": ["German"],
        "omit": False
    },
    "english-spanish": {
        "llama3b": ["Spanish"],
        "qwen4b": ["spanish"],
        "llama8b": ["Spanish"],
        "omit": False
    },
    "landmark-country": {
        "llama3b": ["country"],
        "qwen4b": ["country"],
        "llama8b": ["country"],
        "omit": False
    },
    "lowercase_first_letter": {
        "llama3b": ["f"],
        "qwen4b": ["f"],
        "llama8b": ["f"],
        "omit": True
    },
    "lowercase_last_letter": {
        "llama3b": ["f"],
        "qwen4b": ["f"],
        "llama8b": ["f"],
        "omit": True
    },
    "national_parks": {
        "llama3b": ["state"],
        "qwen4b": ["state"],
        "llama8b": ["state"],
        "omit": False
    },
    "next_capital_letter": {
        "llama3b": ["f"],
        "qwen4b": ["f"],
        "llama8b": ["f"],
        "omit": True
    },
    "next_item": {
        "llama3b": ["Next", "next position", "Advance", "one position", "forward"],
        "qwen4b": ["next", "next item", "Next step", "Advance"],
        "llama8b": ["Next", "next position", "Next step"],
        "omit": False
    },
    "park-country": {
        "llama3b": ["country"],
        "qwen4b": ["country"],
        "llama8b": ["country", "countries"],
        "omit": False
    },
    "present-past": {
        "llama3b": ["past tense"],
        "qwen4b": ["past form", "past tense"],
        "llama8b": ["past tense", "past form"],
        "omit": False
    },
    "prev_item": {
        "llama3b": ["item that precedes", "one position backward", "before", "preceding"],
        "qwen4b": ["previous item", "before", "one less", "predecessor", "precedes"],
        "llama8b": ["before", "back one step", " previous item"],
        "omit": False
    },
    "product-company": {
        "llama3b": ["company", "Who developed"],
        "qwen4b": ["company", "developer", "creator"],
        "llama8b": ["company", "developer"],
        "omit": False
    },
    "singular-plural": {
        "llama3b": ["plural"],
        "qwen4b": ["plural"],
        "llama8b": ["plural"],
        "omit": False
    },
    "synonym": {
        "llama3b": ["synonym", "comparable word", "same meaning", "means the same"],
        "qwen4b": ["same-meaning", "similar meaning", "same meaning"],
        "llama8b": ["synonym", "same meaning", "same idea", "similar", "meaning"],
        "omit": False
    },
    "word_length": {
        "llama3b": ["word length", "number of letters", "Count characters"],
        "qwen4b": ["number of letters", "length", "number of characters"],
        "llama8b": ["word length", "number of letters", "Count characters", "Count", "letters"],
        "omit": False
    },
    "adjective_v_verb_3": {
        "llama3b": ["adjective"],
        "qwen4b": ["adjective"],
        "llama8b": ["adjective", "characteristic"],
        "omit": False
    },
    "adjective_v_verb_5": {
        "llama3b": ["adjective"],
        "qwen4b": ["adjective"],
        "llama8b": ["adjective"],
        "omit": False
    },
    "alphabetically_last_3": {
        "llama3b": ["concludes the alphabet order", "alphabetically last word", "alphabetically highest", "last word if sorted alphabetically", "Alphabetical order","last word"],
        "qwen4b": ["last alphabetically", "Alphabetical", "last word", "alphabetically last", "last in the dictionary", "last when alphabetized"],
        "llama8b": ["concludes the alphabet order", "alphabetically last", "last alphabetically", "last word if sorted alphabetically", "Alphabetical order","last word"],
        "omit": False
    },
    "animal_v_object_3": {
        "llama3b": ["animal","living being"],
        "qwen4b": ["animal", "living creature", "fauna"],
        "llama8b": ["animal", "living creature"],
        "omit": False
    },
    "animal_v_object_5": {
        "llama3b": ["animal","living being","creature"],
        "qwen4b": ["animal"],
        "llama8b": ["animal", "creature"],
        "omit": False
    },
    "choose_first_of_3": {
        "llama3b": ["first"],
        "qwen4b": ["first word", "first element"],
        "llama8b": ["first"],
        "omit": False
    },
    "choose_first_of_5": {
        "llama3b": ["first"],
        "qwen4b": ["first word", "first item"],
        "llama8b": ["first"],
        "omit": False
    },
    "choose_last_of_3": {
        "llama3b": ["final", "last"],
        "qwen4b": ["last word", "final item", "last item", "last in the sequence"],
        "llama8b": ["final", "last"],
        "omit": False
    },
    "choose_last_of_5": {
        "llama3b": ["final", "last"],
        "qwen4b": ["last word", "last element"],
        "llama8b": ["final", "last"],
        "omit": False
    },
    "choose_middle_of_3": {
        "llama3b": ["second"],
        "qwen4b": ["second item", "middle component", "second word"],
        "llama8b": ["second", "middle"],
        "omit": False
    },
    "choose_middle_of_5": {
        "llama3b": ["third", "middle"],
        "qwen4b": ["third item", "after the second word", "second to last from the start", "third word"],
        "llama8b": ["third", "position three", "after the second"],
        "omit": False
    },
    "color_v_animal_3": {
        "llama3b": ["color"],
        "qwen4b": ["color", "tint"],
        "llama8b": ["color"],
        "omit": False
    },
    "color_v_animal_5": {
        "llama3b": ["color"],
        "qwen4b": ["color"],
        "llama8b": ["color"],
        "omit": False
    },
    "concept_v_object_3": {
        "llama3b": ["not a concrete object", "state of being", "not an object", "not a physical object"],
        "qwen4b": ["not a concrete object", "isn't a thing", "not an object", "not a thing"],
        "llama8b": ["adverb", "not a tangible thing", "not an object", "not a tangible item", "isn't a tangible item"],
        "omit": False
    },
    "concept_v_object_5": {
        "llama3b": ["not a concrete object", "describes a feeling", "not an object", "emotion, action, or descriptor"],
        "qwen4b": ["not a tangible item", "not a tangible thing", "not a thing"],
        "llama8b": ["not a tangible item", "not a noun", "not a thing", "not a thing or animal"],
        "omit": False
    },
    "conll2003_location": {
        "llama3b": ["city", "country", "place", "location"],
        "qwen4b": ["location", "city", "country"],
        "llama8b": ["city", "country", "place", "location"],
        "omit": False
    },
    "conll2003_organization": {
        "llama3b": ["company", "name", "title", "organization", "team"],
        "qwen4b": ["entity name", "company", "institutional name", "corporation", "business"],
        "llama8b": ["company name", "noun"],
        "omit": False
    },
    "conll2003_person": {
        "llama3b": ["person"],
        "qwen4b": ["personal name", "person", "individual"],
        "llama8b": ["person", "name"],
        "omit": False
    },
    "fruit_v_animal_3": {
        "llama3b": ["fruit", "plant product", "eat", "edible plant"],
        "qwen4b": ["fruit"],
        "llama8b": ["fruit", "grows on a tree or bush", "edible item"],
        "omit": False
    },
    "fruit_v_animal_5": {
        "llama3b": ["fruit"],
        "qwen4b": ["produce", "fruit", "plants"],
        "llama8b": ["fruit"],
        "omit": False
    },
    "object_v_concept_3": {
        "llama3b": ["object", "thing"],
        "qwen4b": ["object", "thing", "item"],
        "llama8b": ["thing", "item"],
        "omit": False
    },
    "object_v_concept_5": {
        "llama3b": ["object", "thing"],
        "qwen4b": ["object", "thing"],
        "llama8b": ["object", "thing"],
        "omit": False
    },
    "verb_v_adjective_3": {
        "llama3b": ["verb"],
        "qwen4b": ["verb", "not an adjective"],
        "llama8b": ["imperative form", "not an adjective", "not a descriptive word"],
        "omit": False
    },
    "verb_v_adjective_5": {
        "llama3b": ["verb"],
        "qwen4b": ["not a descriptive word", "not an adjective"],
        "llama8b": ["not an adjective", "verb"],
        "omit": False
    }
}



# ==========================================
# 2. Helpers for Attribution
# ==========================================


def get_target_token_variants(tokenizer, target_word):
    variants = [
        target_word.lower(),
        target_word.capitalize(),
        target_word.upper(),
        " " + target_word.lower(),
        " " + target_word.capitalize(),
        " " + target_word.upper()
    ]
    token_ids = set()
    for variant in variants:
        encoded = tokenizer.encode(variant, add_special_tokens=False)
        if encoded:
            token_ids.add(encoded[0])
    return token_ids


def get_substring_indices(text, tokenizer, search_strings):
    """
    Finds token indices corresponding to any occurrence of the search string(s).
    """
    if isinstance(search_strings, str):
        search_strings = [search_strings]

    encoded = tokenizer(text, return_offsets_mapping=True, add_special_tokens=True)
    offsets = encoded['offset_mapping']

    key_indices = set()
    text_lower = text.lower()

    for search_string in search_strings:
        search_lower = search_string.lower()
        start_char = 0

        while True:
            start_char = text_lower.find(search_lower, start_char)
            if start_char == -1:
                break
            end_char = start_char + len(search_string)

            for idx, (start, end) in enumerate(offsets):
                if start == end: continue
                # Add token if it overlaps with the found substring
                if start < end_char and end > start_char:
                    key_indices.add(idx)

            start_char += 1  # advance to allow finding overlapping/subsequent matches

    return list(key_indices)

# ==========================================
# 3. Extraction Loop
# ==========================================

instances_processed = 0
tasks_skipped = 0
task_timings = {}
start_time = time.perf_counter()  # excludes model load, measures extraction loop only

for index, task in enumerate(tasks):
    task_folder = os.path.join(INTERMEDIATE_PATH,MODEL_SHORT, str(task))

    # 3. Check for the specific failure file within that task folder
    failure_file = os.path.join(task_folder, "equiprobable_failure.json")
    res_name = task+"_lrp.pt"
    res_path = os.path.join(task_folder, res_name)


    if os.path.exists(failure_file):
        tasks_skipped += 1
        continue

    if os.path.exists(res_path):
        pass  # result exists but not skipping


    prompts = top_prompts(INTERMEDIATE_PATH,MODEL_SHORT,task)


    filter_set = train_filter_set(INTERMEDIATE_PATH,MODEL_SHORT,task, prompts)

    all_relevances = []
    o_proj_cache = {l: [] for l in range(num_layers)}
    valid_extractions = 0

    dataset = load_dataset(task, root_data_dir=DATASET_PATH, test_size=TEST_SPLIT, seed=RANDOM_SEED,
                           split_valid=(filter_set is None))

    task_start = time.perf_counter()
    for prompt in prompts:
        # print(f"\n--- Processing Prompt: '{prompt}' ---")

        if filter_set is not None:
            # Only keep indices that actually exist in the loaded split
            max_idx = len(dataset["train"])
            indices = [idx for idx in filter_set if idx < max_idx]
        else:
            indices = range(len(dataset["train"]))

        indices_list = list(indices)
        # Determine the sample size (protects against lists smaller than 5)
        random.shuffle(indices_list)

        # Determine the target sample size (protects against lists smaller than 5)
        target_sample_size = min(5, len(indices_list))

        successful_samples = 0  # Track how many we've successfully processed

        search_strings = formatted_tasks[task][model_name]

        for j in indices_list:
            word_pair_test = dataset["train"][int(j)]
            prompt_data = word_pairs_to_prompt_data(
                word_pairs={"input": [], "output": []},
                query_target_pair=word_pair_test,
                instructions=prompt,
                prepend_bos_token=False,
                shuffle_labels=False,
                prefixes=PREFIXES,
                separators=PROMPT_SEPARATORS,
            )

            target = prompt_data["query_target"]["output"]
            target = target[0] if isinstance(target, list) else target
            sentence = create_prompt(prompt_data)

             #1. Use context-aware tokenization to get the EXACT target token ID
            try:
                target_token_id = get_answer_id(sentence, target, tokenizer)[0]
            except IndexError:
                continue

            toks = tokenizer(sentence, return_tensors="pt").to(device)
            # 2. Identify instruction tokens to track attention towards
            key_indices = get_substring_indices(sentence, tokenizer, search_strings)
            if not key_indices:
                raise ValueError(f"No substring found.")

            # 3. Hook setup for Function Vector activations
            temp_cache = {}

            def get_hook(layer_id):
                def hook(_, inp, __):
                    # cache input to o_proj: shape (batch_size, seq_len, hidden_size)
                    temp_cache[layer_id] = inp[0].detach().clone()

                return hook


            handles = [model.model.layers[l].self_attn.o_proj.register_forward_hook(get_hook(l)) for l in
                       range(num_layers)]

            # 4. Forward pass with embed gradients & attention tracking
            embedding_layer = model.get_input_embeddings()
            inputs_embeds = embedding_layer(toks["input_ids"])
            inputs_embeds.requires_grad = True

            outputs = model(inputs_embeds=inputs_embeds, use_cache=False, output_attentions=True)
            next_token_logits = outputs.logits[0, -1, :]


            # 5. Filter: Ensure target is in top-K predictions using soft ranking
            # (unsqueeze is needed because rank_of_token expects shape [1, vocab_size])
            token_rank = rank_of_token(next_token_logits.unsqueeze(0), target_token_id)

            if token_rank > top_k_filter:
                for h in handles: h.remove()
                continue

            # 6. Backward pass for Attribution
            for attn_layer in outputs.attentions:
                attn_layer.retain_grad()

            target_score = next_token_logits[target_token_id]
            model.zero_grad()
            target_score.backward()

            # 7. Calculate Relevance (Activation * Gradient)
            attn_relevance = []
            for attn_layer in outputs.attentions:
                relevance = attn_layer * attn_layer.grad
                attn_relevance.append(relevance.float().detach().cpu())

            concatenated_relevance = torch.cat(attn_relevance)  # [num_layers, num_heads, seq_len, seq_len]

            # Sum relevance from the query token (-1) to all instruction target tokens
            head_scores = concatenated_relevance[:, :, -1, key_indices].clamp(min=0).sum(dim=-1)
            #head_scores = concatenated_relevance[:, :, :, key_indices].clamp(min=0).sum(dim=(2, 3)) #this is for summing over all qp
            all_relevances.append(head_scores)

            # 8. Cache activations for the FV (taking only the last token)
            for l in range(num_layers):
                o_proj_cache[l].append(temp_cache[l][:, -1:, :].cpu().clone())

            for h in handles: h.remove()
            valid_extractions += 1

            instances_processed += 1
            successful_samples += 1

            if successful_samples >= target_sample_size:
                break  # We found enough valid datapoints, stop searching

        """
        
            target = prompt_data["query_target"]["output"]
            target = target[0] if isinstance(target, list) else target
            sentence = create_prompt(prompt_data)
            target_token_ids = get_target_token_variants(tokenizer, target)

            # --- ADD THIS DEBUG BLOCK ---
            if not target_token_ids.intersection(top_preds):
                decoded_preds = [tokenizer.decode([t]) for t in top_preds]
                print(f"\n[DEBUG] Skipped '{target}'.")
                print(f"Target variants: {[tokenizer.decode([t]) for t in target_token_ids]}")
                print(f"Top 5 predicted tokens: {repr(decoded_preds)}")
            # ----------------------------

            matching_targets = target_token_ids.intersection(top_preds)

            if not matching_targets:
                #print(
                #    f"Target token filter failed: No tokens from {target_token_ids} "
                 #   f"appeared in the top-{top_k_filter} predictions.")
                for h in handles: h.remove()
                continue


            primary_target_id = list(matching_targets)[0]

            target_score = next_token_logits[primary_target_id]
        """

    if valid_extractions == 0:
        raise ValueError("No valid combinations found where the target was in the Top-K predictions.")

    task_timings[task] = round(time.perf_counter() - task_start, 4)
    print(task, "time:", task_timings[task])

    # ==========================================
    # 4. Format tensors
    # ==========================================



    # 1. Format Indirect Effect (Relevances)
    # Eval script expects: (n_prompts, n_trials, num_layers, num_heads) and calls .mean(dim=0).mean(dim=0)
    # We provide: (1, 1, num_layers, num_heads) so the dual mean operations naturally result in (num_layers, num_heads)
    stacked_rel = torch.stack(all_relevances)  # [N, num_layers, num_heads]
    mean_rel = stacked_rel.mean(dim=0)  # [num_layers, num_heads]
    eval_formatted_aie = mean_rel.view(1, 1, num_layers, n_heads)

    # 2. Format Mean Activations (for Steering / FVs)
    # Eval script expects: (num_layers, num_heads, n_dummy_labels, head_dim) and slices [:, :, -1, :]
    # We construct: (num_layers, num_heads, 1, head_dim)
    mean_activations_per_layer = []
    for l in range(num_layers):
        # o_proj_cache[l] contains N tensors of shape (1, 1, hidden_dim)
        layer_acts = torch.cat(o_proj_cache[l], dim=0)  # [N, 1, hidden_dim]
        layer_mean = layer_acts.mean(dim=0)  # [1, hidden_dim]

        # Reshape from (1, hidden_dim) -> (n_heads, 1, head_dim)
        layer_mean_heads = layer_mean.view(n_heads, 1, head_dim)
        mean_activations_per_layer.append(layer_mean_heads)

    eval_formatted_mean_acts = torch.stack(mean_activations_per_layer)  # [num_layers, n_heads, 1, head_dim]


if torch.cuda.is_available():
    torch.cuda.synchronize()
end_time = time.perf_counter()
elapsed_time_seconds = end_time - start_time

results = {
    "model": MODEL_SHORT,
    "total_instances": instances_processed,
    "tasks_skipped": tasks_skipped,
    "time_seconds": round(elapsed_time_seconds, 4),
    "instances_per_second": round(instances_processed / elapsed_time_seconds, 2) if elapsed_time_seconds > 0 else None,
    "per_task_seconds": task_timings,
}

timing_dir = os.path.join("timing_results", MODEL_SHORT)
os.makedirs(timing_dir, exist_ok=True)
stats_path = os.path.join(timing_dir, "processing_stats.json")
with open(stats_path, "w") as file:
    json.dump(results, file, indent=4)