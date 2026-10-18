# Per-token prompt weighting for MOSS-Image, KJNodes Krea2PromptWeight syntax: (phrase:weight).
# The Qwen3.5 tokenizer drops ComfyUI's weights, so instead every attention call of the weighted prompt gets:
#   weight > 0:  a bias added to the token's attention logits, (x:2) boosts, (x:0.5) weakens. "linear" adds
#                2 * (weight - 1) like KJNodes, "log" adds log(weight), which multiplies the share of attention the
#                token gets by exactly weight before the softmax renormalizes.
#   weight <= 0: the token's value vector is scaled by weight instead (0 mutes it, -1 subtracts the concept).
# Weights follow the conditioning tensor rather than being global, so the negative prompt is left alone and CFG works.
import logging
import math
import re

import torch

import comfy.patcher_extension

WEIGHT_RE = re.compile(r"\(([^():]+):(-?\d*\.?\d+)\)")
KEY = "moss_prompt_weight"


def common_prefix(a, b):
    n = 0
    while n < min(len(a), len(b)) and a[n] == b[n]:
        n += 1
    return n


def find_all(seq, sub, lo, hi):
    return [i for i in range(lo, hi - len(sub) + 1) if seq[i:i + len(sub)] == sub]


def weighted_attention(func, q, k, v, heads, *args, transformer_options={}, **kwargs):
    # text tokens are always the last L positions: the whole sequence in the text embedder, after the image in the blocks
    bias, v_scale, prev = transformer_options[KEY]
    n = k.shape[1]
    if v_scale is not None:
        v = v * v_scale[:, -n:]
    if bias is not None:
        kwargs["mask"] = bias[..., -n:]
    if prev is not None:
        return prev(func, q, k, v, heads, *args, transformer_options=transformer_options, **kwargs)
    return func(q, k, v, heads, *args, transformer_options=transformer_options, **kwargs)


class PromptWeightWrapper:
    def __init__(self, cond, bias, v_scale):
        self.cond = cond
        self.bias = bias  # {position: logit bias}
        self.v_scale = v_scale  # {position: value multiplier}

    def __call__(self, executor, x, timestep, context, transformer_options, **kwargs):
        B, L = context.shape[:2]
        rows = []
        if L == self.cond.shape[0]:
            ref = self.cond.to(context)
            rows = [b for b in range(B) if torch.equal(context[b], ref)]
        if not rows:
            return executor(x, timestep, context, transformer_options, **kwargs)

        n = x.shape[-2] * x.shape[-1] + L
        transformer_options = transformer_options.copy()
        bias, v_scale, prev = transformer_options.get(KEY, (None, None, transformer_options.get("optimized_attention_override")))
        if self.bias:
            add = torch.zeros((B, 1, 1, n), device=x.device, dtype=x.dtype)
            add[rows, 0, 0, torch.tensor([n - L + p for p in self.bias], device=x.device)[:, None]] = torch.tensor(list(self.bias.values()), device=x.device, dtype=x.dtype)[:, None]
            bias = add if bias is None else bias + add
        if self.v_scale:
            mul = torch.ones((B, n, 1), device=x.device, dtype=x.dtype)
            mul[torch.tensor(rows, device=x.device)[:, None], torch.tensor([n - L + p for p in self.v_scale], device=x.device), 0] = torch.tensor(list(self.v_scale.values()), device=x.device, dtype=x.dtype)
            v_scale = mul if v_scale is None else v_scale * mul
        transformer_options[KEY] = (bias, v_scale, prev)
        transformer_options["optimized_attention_override"] = weighted_attention
        return executor(x, timestep, context, transformer_options, **kwargs)


class MOSSImagePromptWeight:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL",),
                "clip": ("CLIP",),
                "text": ("STRING", {"multiline": True, "dynamicPrompts": True, "tooltip": "Prompt with per-phrase weights, e.g. (red hat:1.5) boosts, (red hat:0.5) weakens, (red hat:0) mutes, (red hat:-1) subtracts. Plain text is 1.0."}),
                "strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 4.0, "step": 0.05, "tooltip": "Global multiplier on the weighting. The effect compounds over all 44 attention layers, lower it if the image breaks up."}),
                "mapping": (["linear", "log"], {"default": "linear", "tooltip": "How a positive weight becomes an attention bias. linear: 2 * (weight - 1), like KJNodes, hits harder ((x:2) is about 7x the attention). log: log(weight), (x:2) gets exactly 2x the attention, finer control near 1."}),
            },
        }

    RETURN_TYPES = ("MODEL", "CONDITIONING")
    FUNCTION = "encode"
    CATEGORY = "conditioning/moss_image"
    DESCRIPTION = "Per-phrase prompt weighting for MOSS-Image via attention biasing, (phrase:weight) syntax. Weights only the conditioning it outputs, so it works with CFG; use a second node to weight the negative prompt."
    EXPERIMENTAL = True

    def encode(self, model, clip, text, strength, mapping="linear"):
        terms = [(m.group(1).strip(), float(m.group(2))) for m in WEIGHT_RE.finditer(text)]
        tokens = clip.tokenize(WEIGHT_RE.sub(lambda m: m.group(1), text))
        cond = clip.encode_from_tokens_scheduled(tokens)
        if not terms:
            return (model, cond)

        key = next(iter(tokens))
        ids = [t[0] for t in tokens[key][0]]
        empty = [t[0] for t in clip.tokenize("")[key][0]]
        # the prompt sits between the template's prefix and suffix
        start = common_prefix(ids, empty)
        end = max(start, len(ids) - common_prefix(ids[::-1], empty[::-1]))
        offset = len(ids) - cond[0][0].shape[1]
        hf = getattr(clip.tokenizer, clip.tokenizer.clip).tokenizer

        bias, v_scale = {}, {}
        for phrase, w in terms:
            positions = []
            for variant in (" " + phrase, phrase):  # mid-sentence words carry a leading space token
                sub = hf(variant)["input_ids"]
                positions = [i + j - offset for i in find_all(ids, sub, start, end) for j in range(len(sub))]
                if positions:
                    break
            if not positions:
                logging.warning(f"MOSSImagePromptWeight: '{phrase}' not found in the tokenized prompt, skipped.")
                continue
            for p in positions:
                if w > 0:
                    bias[p] = strength * (math.log(w) if mapping == "log" else 2.0 * (w - 1.0))
                else:
                    v_scale[p] = 1.0 + strength * (w - 1.0)
        if not bias and not v_scale:
            return (model, cond)

        model = model.clone()
        model.add_wrapper_with_key(comfy.patcher_extension.WrappersMP.DIFFUSION_MODEL, KEY, PromptWeightWrapper(cond[0][0][0].cpu(), bias, v_scale))
        return (model, cond)


NODE_CLASS_MAPPINGS = {"MOSSImagePromptWeight": MOSSImagePromptWeight}
NODE_DISPLAY_NAME_MAPPINGS = {"MOSSImagePromptWeight": "MOSS-Image Prompt Weight"}
