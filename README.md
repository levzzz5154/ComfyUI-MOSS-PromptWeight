# ComfyUI-MOSS-PromptWeight

Per-phrase prompt weighting for MOSS-Image in ComfyUI. The Qwen3.5 text encoder ignores ComfyUI's normal `(word:1.2)` weights, so this node applies them inside the attention of the diffusion model and the text encoder instead.

## Install

Clone into `ComfyUI/custom_nodes` and restart ComfyUI:

```
cd ComfyUI/custom_nodes
git clone https://github.com/levzzz5154/ComfyUI-MOSS-PromptWeight
```

No extra dependencies. Needs a ComfyUI version with MOSS-Image support.

## Usage

Use **MOSS-Image Prompt Weight** (`conditioning/moss_image`) in place of CLIP Text Encode for the positive prompt, and use its `model` output for sampling.

```
a ginger cat wearing a (red hat:-1), a vase of (yellow flowers:1.5) in the background
```

- `(x:1.5)` boosts a phrase and `(x:0.5)` weakens it, by biasing how much attention its tokens get.
- `(x:0)` mutes a phrase and `(x:-1)` subtracts it, by scaling its value vectors.
- Like ComfyUI, the weight is the number after the last colon, so groups can nest and hold colons, e.g. a JSON chunk `({"caption":"hakurei reimu"}:2)`. Parentheses without a weight stay in the prompt as text, e.g. `girl (touhou)`; escape them only when the text itself ends in `:number)`, e.g. `\(12:30\)`.
- `strength` multiplies the whole effect. Lower it if the image breaks up.
- `mapping` controls how a positive weight becomes a bias:
  - `linear` (default, same as KJNodes' Krea2 node): `2 * (w - 1)`. Hits harder.
  - `log`: `log(w)`. `(x:2)` gets exactly 2x the attention.
- `encoder` (default on) also applies the weights inside the Qwen3.5 text encoder. It is causal, so otherwise the words after a phrase carry it on unweighted: a muted phrase still reaches them at full strength. With `encoder` on, about 13% gets through. Negative weights mute the phrase in the encoder rather than subtract it.

Weights apply only to the conditioning this node outputs, so CFG and the negative prompt work normally. To weight the negative prompt, chain a second node.

![example](example.png)
Same seed, left to right: baseline, `(yellow flowers:3)` log, `(yellow flowers:3)` linear, `(red hat:0.3)` log, `(red hat:0.3)` linear.

## Notes

- If you edit the conditioning afterwards (averaging, zeroing, concatenating), the weights stop applying. Combining conditionings is fine.
- Weighted steps keep the attention kernel stored in the checkpoint (e.g. int8 in `_attn` models). If another node sets an attention override (Sage, Flash, ...), that one is used instead.

Based on the approach of KJNodes' `Krea2PromptWeight`.
