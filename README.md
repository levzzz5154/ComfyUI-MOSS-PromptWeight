# ComfyUI-MOSS-PromptWeight

Per-phrase prompt weighting for MOSS-Image in ComfyUI. The Qwen3.5 text encoder ignores ComfyUI's normal `(word:1.2)` weights, so this node applies them inside the diffusion model's attention instead.

## Usage

Use **MOSS-Image Prompt Weight** (`conditioning/moss_image`) in place of CLIP Text Encode for the positive prompt, and use its `model` output for sampling.

```
a ginger cat wearing a (red hat:-1), a vase of (yellow flowers:1.5) in the background
```

- `(x:1.5)` boosts a phrase and `(x:0.5)` weakens it, by biasing how much attention its tokens get.
- `(x:0)` mutes a phrase and `(x:-1)` subtracts it, by scaling its value vectors.
- `strength` multiplies the whole effect. Lower it if the image breaks up.
- `mapping` controls how a positive weight becomes a bias:
  - `linear` (default, same as KJNodes' Krea2 node): `2 * (w - 1)`. Hits harder.
  - `log`: `log(w)`. `(x:2)` gets exactly 2x the attention.

Weights apply only to the conditioning this node outputs, so CFG and the negative prompt work normally. To weight the negative prompt, chain a second node.

![example](example.png)
Same seed, left to right: baseline, `(yellow flowers:3)` log, `(yellow flowers:3)` linear, `(red hat:0.3)` log, `(red hat:0.3)` linear.

## Notes

- If you edit the conditioning afterwards (averaging, zeroing, concatenating), the weights stop applying. Combining conditionings is fine.
- Weighted steps use the global attention backend with a mask, not the per-layer attention setting stored in the checkpoint.

Based on the approach of KJNodes' `Krea2PromptWeight`.
