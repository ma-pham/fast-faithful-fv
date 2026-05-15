"""Steer a string in a task direction and compare head selection configs.

Usage:
    python eval/steer_string.py
"""
import os, sys
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from eval.utils import (
    load_cie, load_aie, load_mean_attn, load_mean_mlp,
    select_modules, build_steering_vecs,
)
from utils_v2 import split_activations_by_head

# ── Config ────────────────────────────────────────────────────────────────────

MODEL       = "Qwen/Qwen3-4B-Instruct-2507"
MODEL_SHORT = MODEL.split("/")[-1]
CACHE_DIR   = "./cache"
DEVICE      = "cuda:0"
N_TOKENS    = 1
GEN_TOKENS  = 100

# Head selection configs to compare
CONFIGS = [
    # dict(label="per_task/attn/0shot",  selection="per_task",  modules="attn",  cie_shot="0shot",  topk_attn=20, topk_mlp=6,  topk_joint=26),
    # dict(label="task_aie/attn/0shot",  selection="task_aie",  modules="attn",  cie_shot="0shot",  topk_attn=20, topk_mlp=6,  topk_joint=26),
    # dict(label="task_aie/attn/10shot",  selection="task_aie",  modules="attn",  cie_shot="10shot",  topk_attn=20, topk_mlp=6,  topk_joint=26),
    # dict(label="per_task/attn/10shot",  selection="per_task",  modules="attn",  cie_shot="10shot",  topk_attn=20, topk_mlp=6,  topk_joint=26),
    dict(label="task_aie/joint/10shot",  selection="task_aie",  modules="joint",  cie_shot="10shot",  topk_attn=20, topk_mlp=6,  topk_joint=26),
    dict(label="task_aie/joint/0shot",  selection="task_aie",  modules="joint",  cie_shot="0shot",  topk_attn=20, topk_mlp=6,  topk_joint=26),
    dict(label="per_task/joint/10shot",  selection="per_task",  modules="joint",  cie_shot="10shot",  topk_attn=20, topk_mlp=6,  topk_joint=26),
    dict(label="per_task/joint/0shot",  selection="per_task",  modules="joint",  cie_shot="0shot",  topk_attn=20, topk_mlp=6,  topk_joint=26),

]

ALL_TASKS = [
    "antonym", "capitalize", "capitalize_first_letter", "country-capital",
    "country-currency", "english-french", "english-german", "english-spanish",
    "landmark-country", "lowercase_first_letter", "national_parks", "park-country",
    "person-sport", "present-past", "product-company", "sentiment",
    "singular-plural", "synonym",
]

SCALE = 0.5


# ── Steering ──────────────────────────────────────────────────────────────────

def get_modules(cfg: dict, task: str, n_heads: int):
    if cfg["selection"] == "per_task":
        attn_cie, mlp_cie = load_cie(CACHE_DIR, MODEL_SHORT, cfg["cie_shot"], task)
    else:
        attn_cie, mlp_cie = load_aie(CACHE_DIR, MODEL_SHORT, cfg["cie_shot"], ALL_TASKS)

    top_heads, top_mlp_layers = select_modules(
        attn_cie, mlp_cie, cfg["modules"],
        cfg["topk_attn"], cfg["topk_mlp"], cfg["topk_joint"],
    )

    mean_attn = load_mean_attn(CACHE_DIR, MODEL_SHORT, task, N_TOKENS)
    mean_mlp  = load_mean_mlp(CACHE_DIR, MODEL_SHORT, task, N_TOKENS)
    dist_fv, dist_fv_mlp = build_steering_vecs(mean_attn, top_heads, n_heads, mean_mlp=mean_mlp, top_mlp_layers=top_mlp_layers)
    return top_heads, top_mlp_layers, dist_fv.to(DEVICE), dist_fv_mlp.to(DEVICE)


def _register_hooks(model, top_heads, top_mlp_layers, dist_fv, dist_fv_mlp, n_heads, scale):
    heads_by_layer = {}
    for gid, (L, H) in enumerate(top_heads):
        heads_by_layer.setdefault(L, []).append((gid, H))

    hooks = []
    for L, head_list in heads_by_layer.items():
        def attn_hook(module, inputs, _hl=head_list):
            x = inputs[0]
            B, S, D = x.shape
            x_heads = split_activations_by_head(x, n_heads=n_heads)
            for gid, H in _hl:
                for t in range(N_TOKENS):
                    pos = -(N_TOKENS - t)
                    x_heads[:, pos, H] += scale * dist_fv[gid, t]
            return (x_heads.view(B, S, D),)
        hooks.append(model.model.layers[L].self_attn.o_proj.register_forward_pre_hook(attn_hook, with_kwargs=False))

    for mlp_idx, L in enumerate(top_mlp_layers):
        def mlp_hook(module, inp, out, _i=mlp_idx):
            out = out.clone()
            for t in range(N_TOKENS):
                pos = -(N_TOKENS - t)
                out[:, pos, :] += scale * dist_fv_mlp[_i, t]
            return out
        hooks.append(model.model.layers[L].mlp.register_forward_hook(mlp_hook))

    return hooks


@torch.inference_mode()
def steer_string(prompt: str, task: str, cfg: dict, model, tokenizer, n_heads: int, scale: float = SCALE):
    top_heads, top_mlp_layers, dist_fv, dist_fv_mlp = get_modules(cfg, task, n_heads)
    hooks = _register_hooks(model, top_heads, top_mlp_layers, dist_fv, dist_fv_mlp, n_heads, scale)

    toks = tokenizer(prompt, return_tensors="pt").to(DEVICE)
    out = model.generate(
        **toks, max_new_tokens=GEN_TOKENS,
        do_sample=False, temperature=None, top_p=None,
        pad_token_id=tokenizer.eos_token_id,
    )

    for h in hooks:
        h.remove()

    return tokenizer.decode(out[0][toks["input_ids"].shape[1]:], skip_special_tokens=True)


@torch.inference_mode()
def baseline_string(prompt: str, model, tokenizer):
    toks = tokenizer(prompt, return_tensors="pt").to(DEVICE)
    out = model.generate(
        **toks, max_new_tokens=GEN_TOKENS,
        do_sample=False, temperature=None, top_p=None,
        pad_token_id=tokenizer.eos_token_id,
    )
    return tokenizer.decode(out[0][toks["input_ids"].shape[1]:], skip_special_tokens=True)


# ── Main ──────────────────────────────────────────────────────────────────────

def compare(prompt: str, task: str, model, tokenizer, n_heads: int, scale: float = SCALE, configs=None):
    if configs is None:
        configs = CONFIGS

    print(f"\nPrompt : {repr(prompt)}")
    print(f"Task   : {task}  |  scale={scale}")
    print("=" * 60)

    # base = baseline_string(prompt, model, tokenizer)
    # print(f"\n[BASELINE]\n  {base}")

    for cfg in configs:
        gen = steer_string(prompt, task, cfg, model, tokenizer, n_heads, scale)
        print(f"\n[{cfg['label']}]\n  {gen}")


def main():
    print(f"Loading {MODEL}...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, trust_remote_code=True, torch_dtype=torch.bfloat16,
    ).to(DEVICE).eval()
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    n_heads = model.config.num_attention_heads

    # ── Edit examples below ───────────────────────────────────────────────────
    examples = [
        ("Albert Einstein[a] (14 March 1879 – 18 April 1955) was a German-born theoretical physicist best known for developing the theory of relativity. Einstein also made important contributions to quantum theory.[1][5] His mass–energy equivalence formula E = mc2, which arises from special relativity, has been called 'the world's most famous equation#.[6] He received the 1921 Nobel Prize in Physics for 'his services to theoretical physics, and especially for his discovery of the law of the photoelectric effect'.[7] ", "english-spanish"),
        ("Albert Einstein[a] (14 March 1879 – 18 April 1955) was a German-born theoretical physicist best known for developing the theory of relativity. Einstein also made important contributions to quantum theory.[1][5] His mass–energy equivalence formula E = mc2, which arises from special relativity, has been called 'the world's most famous equation#.[6] He received the 1921 Nobel Prize in Physics for 'his services to theoretical physics, and especially for his discovery of the law of the photoelectric effect'.[7] ", "english-french"),
        ("Over 80 percent of project cost was for building and operating the fissile material production plants. The project proposed both highly enriched uranium and plutonium as fuel for nuclear weapons. Enriched uranium was produced at the Clinton Engineer Works in Tennessee. Plutonium was produced in the world's first industrial-scale nuclear reactors at the Hanford Engineer Works in Washington. These sites were supported by dozens of other facilities across Canada, the US and the UK. ", "english-spanish"),
        ("Over 80 percent of project cost was for building and operating the fissile material production plants. The project proposed both highly enriched uranium and plutonium as fuel for nuclear weapons. Enriched uranium was produced at the Clinton Engineer Works in Tennessee. Plutonium was produced in the world's first industrial-scale nuclear reactors at the Hanford Engineer Works in Washington. These sites were supported by dozens of other facilities across Canada, the US and the UK. ", "english-french"),
        ("Berkeley students compete in thirty varsity athletic sports, and the university is one of eighteen full-member institutions in the Atlantic Coast Conference (ACC). Berkeley's athletic teams, the California Golden Bears, have also won 107 national championships, 196 individual national titles, and 223 Olympic medals (including 121 gold).[19][20] Berkeley's alumni, faculty, and researchers include 63 Nobel laureates[21] and 19 Academy Award winners,[22] and the university is also a producer of Rhodes Scholars,[23] Marshall Scholars,[24] and Fulbright Scholars.", "english-spanish"),
        ("Berkeley students compete in thirty varsity athletic sports, and the university is one of eighteen full-member institutions in the Atlantic Coast Conference (ACC). Berkeley's athletic teams, the California Golden Bears, have also won 107 national championships, 196 individual national titles, and 223 Olympic medals (including 121 gold).[19][20] Berkeley's alumni, faculty, and researchers include 63 Nobel laureates[21] and 19 Academy Award winners,[22] and the university is also a producer of Rhodes Scholars,[23] Marshall Scholars,[24] and Fulbright Scholars.", "english-french"),
        ("The college later disincorporated and merged with the state of California's Agricultural, Mining, and Mechanical Arts College to create the University of California in 1868. Durant was elected the first president of the University of California on August 16, 1870, and resigned only two years later in order to relinquish the position to a younger man (Daniel Coit Gilman). In 1873, the University of California moved to its new Berkeley campus.", "english-spanish"),
        ("The college later disincorporated and merged with the state of California's Agricultural, Mining, and Mechanical Arts College to create the University of California in 1868. Durant was elected the first president of the University of California on August 16, 1870, and resigned only two years later in order to relinquish the position to a younger man (Daniel Coit Gilman). In 1873, the University of California moved to its new Berkeley campus.", "english-french"),
        ("The city of Oakland, California, was founded in 1852 and incorporated in 1854. The city uses a strong mayor form of government. Until the early 20th century, all Oakland mayors served terms of only one or two years each. Oakland mayors now serve 4-year terms and are limited to two terms. ", "english-spanish"),
        ("The city of Oakland, California, was founded in 1852 and incorporated in 1854. The city uses a strong mayor form of government. Until the early 20th century, all Oakland mayors served terms of only one or two years each. Oakland mayors now serve 4-year terms and are limited to two terms. ", "english-french"),
        # ("Q: cat\nA:", "singular-plural"),
        # ("Q: Paris\nA:", "country-capital"),
    ]

    for prompt, task in examples:
        compare(prompt, task, model, tokenizer, n_heads)


if __name__ == "__main__":
    main()
