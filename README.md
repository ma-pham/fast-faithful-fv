# Fast and Faithful Function Vectors

## Setup

### Environment

Set `STORAGE_ROOT` to the directory where cached data and results will be saved:

```bash
export STORAGE_ROOT=/path/to/LRP_dist_function_vectors/storage
```

### Installing dependencies

```bash
uv pip install -r requirements.txt
```

> **Note:** `transformers==4.52.4` is pinned because `lxt==2.1` (tested against 4.52.4) imports `find_pruneable_heads_and_indices` from `transformers.pytorch_utils`, which was removed in transformers 5.x.

## Usage

### Cache texts

Create baseline. Results are saved to `$STORAGE_ROOT/function_vectors/short_real_text_caches/`.

```bash
STORAGE_ROOT=/path/to/LRP_dist_function_vectors/storage \
python slim/cache_texts.py \
  --model_name meta-llama/Llama-3.2-3B-Instruct \
  --max_length_tokens 16
```

Output file: `$STORAGE_ROOT/function_vectors/short_real_text_caches/llama-3.2-3b-instruct_wikitext-103-v1.csv.gz`
(appends `_long` to the filename if `--max_length_tokens > 16`)

### Extract attention heads via LRP

Runs LRP-based attribution to extract the most relevant attention heads per task. Model is configured at the top of the script.

```bash
python extract_heads_lrp.py
```

### Run prompt function vectors across all tasks

Run indirect effect calcuation for every head

```bash
python slim/run_prompt_fv_all_tasks.py
```

### evaluate 

```bash
python eval/run_eval_simple.py
```

## Thanks

This project builds on [Davidson et al. (2025), "Do different prompting methods yield a common task representation in language models?"](https://arxiv.org/abs/2505.12075) (Guy Davidson, Todd M. Gureckis, Brenden M. Lake, Adina Williams). Their code is available at https://github.com/guydav/prompting-methods-task-representations.