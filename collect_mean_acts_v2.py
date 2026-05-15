from tqdm import tqdm
import torch

from utils_v2 import *
from prompt_utils import *


def collect(task, args, model, tokenizer, activations_file, device):

    n_test_examples = 1
    if not hasattr(args, 'seed'):
        raise ValueError("args.seed is required but not provided")
    seed = args.seed
    test_size = 1 - args.train_split

    # Model configuration
    n_heads = model.config.num_attention_heads
    n_layers = model.config.num_hidden_layers
    resid_dim = model.config.hidden_size
    model_head_dim = model.config.head_dim
    model_config_prepend_bos = True
    prepend_bos = False if model_config_prepend_bos else True

    layers = np.repeat(np.arange(0, n_layers), n_heads)
    heads = np.tile(np.arange(0, n_heads), n_layers)
    all_heads_ids = np.stack((layers, heads), axis=1)

    activations, hooks = setup_hooks(model, all_heads_ids, last_token=True)

    generator = torch.Generator()
    generator.manual_seed(seed)
    dataset = load_dataset(task, root_data_dir=args.dataset_path, test_size=test_size, seed=seed)

    activation_storage = torch.zeros(args.mean_acts_trials, len(all_heads_ids), model_head_dim, device=device)
    for i in tqdm(range(args.mean_acts_trials), desc=f"Generating activations for {task}"):
        train_perm = torch.randperm(len(dataset['train']), generator=generator)[:args.n_icl_examples].tolist()
        word_pairs = dataset['train'][train_perm]

        val_perm = torch.randperm(len(dataset['valid']), generator=generator)[:n_test_examples]
        word_pairs_test = dataset['valid'][val_perm.tolist()]

        prompt_data = word_pairs_to_prompt_data(word_pairs, query_target_pair=word_pairs_test, prepend_bos_token=prepend_bos, shuffle_labels=False, prefixes=args.prefixes, separators=args.separators)

        query = prompt_data['query_target']['input']
        _, prompt_string = get_token_meta_labels(prompt_data, tokenizer, query, prepend_bos=model_config_prepend_bos)

        inputs = tokenizer([prompt_string], return_tensors='pt').to(device)
        acts = extract_activations_attn(model, inputs, activations, all_heads_ids, n_heads, resid_dim, model_head_dim, last_token=True, device=device)
        activation_storage[i] = acts.squeeze()

    if args.activations_path:
        os.makedirs(os.path.dirname(activations_file), exist_ok=True)
        print(f"Saving activations to {activations_file}")
        torch.save(activation_storage, activations_file)

    for hook in hooks:
        hook.remove()

    return activation_storage
