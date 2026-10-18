# Per-token prompt weighting for MOSS-Image, KJNodes Krea2PromptWeight syntax: (phrase:weight).
# The Qwen3.5 tokenizer drops ComfyUI's weights, so instead every attention call of the weighted prompt gets:
#   weight > 0:  a bias added to the token's attention logits, (x:2) boosts, (x:0.5) weakens. "linear" adds
#                2 * (weight - 1) like KJNodes, "log" adds log(weight), which multiplies the share of attention the
#                token gets by exactly weight before the softmax renormalizes.
#   weight <= 0: the token's value vector is scaled by weight instead (0 mutes it, -1 subtracts the concept).
# Weights follow the conditioning tensor rather than being global, so the negative prompt is left alone and CFG works.
import codecs
import logging
import math
import re

import torch

import comfy.ldm.modules.attention
import comfy.patcher_extension
from comfy.sd1_clip import escape_important, unescape_important
from comfy.ldm.modules.attention import AttentionTensorContainer

WEIGHT_RE = re.compile(r"-?\d*\.?\d+")
KEY = "moss_prompt_weight"

# GPT-2 byte-level BPE alphabet, built here because transformers 5 tokenizers no longer expose byte_decoder
_printable = [*range(ord("!"), ord("~") + 1), *range(ord("¡"), ord("¬") + 1), *range(ord("®"), ord("ÿ") + 1)]
BYTE_DECODER = {chr(b): b for b in _printable}
BYTE_DECODER.update({chr(256 + i): b for i, b in enumerate(b for b in range(256) if b not in _printable)})


def split_groups(text):
    # top level text and balanced (...) groups, unmatched parentheses stay text, e.g. "smile :)"
    out, depth, start = [], 0, 0
    for i, c in enumerate(text):
        if c == "(":
            if depth == 0:
                out.append(text[start:i])
                start = i
            depth += 1
        elif c == ")" and depth:
            depth -= 1
            if depth == 0:
                out.append(text[start:i + 1])
                start = i + 1
    if depth:
        return out + ["("] + split_groups(text[start + 1:])
    return out + [text[start:]]


def strip_weights(text, terms):
    # like ComfyUI's token_weights: weight after the last colon, so a phrase may hold colons and nested groups,
    # e.g. JSON. Groups without a numeric weight stay literal text since the Qwen3.5 tokenizer keeps them.
    out = ""
    for x in split_groups(text):
        if len(x) < 2 or x[0] != "(" or x[-1] != ")":
            out += x
            continue
        phrase, sep, w = x[1:-1].rpartition(":")
        if not phrase.strip() or not WEIGHT_RE.fullmatch(w.strip()):
            out += "(" + strip_weights(x[1:-1], terms) + ")"
            continue
        i = len(terms)
        terms.append(None)  # outer group first so nested weights override it
        phrase = strip_weights(phrase, terms)
        terms[i] = (unescape_important(phrase.strip()), float(w))
        out += phrase
    return out


def common_prefix(a, b):
    n = 0
    while n < min(len(a), len(b)) and a[n] == b[n]:
        n += 1
    return n


def apply_weights(k_len, v, kwargs, transformer_options):
    # text tokens are always the last L positions: the whole sequence in the text embedder, after the image in the blocks
    bias, v_scale, _ = transformer_options[KEY]
    if v_scale is not None:
        v = v * v_scale[:, -k_len:]
    if bias is not None:
        kwargs["mask"] = bias[..., -k_len:]
    return v


def weighted_attention(func, q, k, v, heads, *args, transformer_options={}, **kwargs):
    # only used when another node already set an attention override, which then replaces the per-layer attention anyway
    v = apply_weights(k.shape[1], v, kwargs, transformer_options)
    prev = transformer_options[KEY][2]
    return prev(func, q, k, v, heads, *args, transformer_options=transformer_options, **kwargs)


class WeightedComfyAttention(comfy.ldm.modules.attention.ComfyAttention):
    # replaces a layer's ComfyAttention so weighted steps still run the checkpoint's own kernel, e.g. int8 in _attn models
    def __init__(self, orig):
        super().__init__()
        self.config = orig.config
        self.orig = orig.function or comfy.ldm.modules.attention.optimized_attention
        self.function = self.weighted

    def weighted(self, q, k, v, heads, *args, transformer_options={}, **kwargs):
        if KEY in transformer_options:
            v = AttentionTensorContainer(apply_weights(k.peek().shape[1], v.take(), kwargs, transformer_options))
        return self.orig(q, k, v, heads, *args, transformer_options=transformer_options, **kwargs)


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
        if prev is not None:
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
        terms = []
        tokens = clip.tokenize(unescape_important(strip_weights(escape_important(text), terms)))
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
        # match phrases on the decoded prompt text: in context BPE can merge a phrase's edge into a neighbour, e.g. "),"
        decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
        prompt, spans = "", []
        for t in hf.convert_ids_to_tokens(ids[start:end]):
            lo = len(prompt)
            prompt += decoder.decode(bytes(BYTE_DECODER[c] for c in t))
            spans.append((lo, max(len(prompt), lo + 1)))  # a token holding only part of a character still covers it

        bias, v_scale = {}, {}
        for phrase, w in terms:
            matches = [(m.start(), m.end()) for m in re.finditer(r"(?<!\w)" + re.escape(phrase) + r"(?!\w)", prompt)]
            positions = [start + i - offset for a, b in matches for i, (lo, hi) in enumerate(spans) if lo < b and hi > a]
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
        for name, m in model.get_model_object("diffusion_model").named_modules():
            if isinstance(m, comfy.ldm.modules.attention.ComfyAttention):
                path = f"diffusion_model.{name}"
                current = model.get_model_object(path)
                if not isinstance(current, WeightedComfyAttention):
                    model.add_object_patch(path, WeightedComfyAttention(current))
        model.add_wrapper_with_key(comfy.patcher_extension.WrappersMP.DIFFUSION_MODEL, KEY, PromptWeightWrapper(cond[0][0][0].cpu(), bias, v_scale))
        return (model, cond)


NODE_CLASS_MAPPINGS = {"MOSSImagePromptWeight": MOSSImagePromptWeight}
NODE_DISPLAY_NAME_MAPPINGS = {"MOSSImagePromptWeight": "MOSS-Image Prompt Weight"}
