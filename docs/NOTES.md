# Study Notes

One section per module: what it is, why it exists, the key formula, tensor
shapes, and one common bug. Design decisions come first so you can see why
things are the way they are.

Notation: `B` batch, `S` sequence length, `D` d_model, `H` heads, `d_k = D / H`,
`V` vocab size, `F` d_ff.

## Design decisions

- **Linear**: weight stored as `(out, in)`, no bias. Init is truncated normal with
  `std = sqrt(2 / (in + out))`, cut at +-3 std. `nn.init.trunc_normal_` does only
  the sampling; it is an init helper, not a layer. The forward pass uses
  `einops.einsum` with named dims so the contraction axis is explicit.
- **Embedding**: truncated normal, std 1, cut at +-3. Lookup is plain indexing
  (`weight[ids]`), not a one-hot matmul.
- **RMSNorm**: input upcast to float32 for the mean square, gain applied in
  float32, result cast back to the input dtype. Gain is `weight`, initialized to
  ones. `eps = 1e-5`.
- **softmax**: subtract the per-row max before `exp`. A row that is entirely
  `-inf` gives NaN (`inf - inf`). The causal mask never produces such a row
  because a token can always attend to itself.
- **SwiGLU**: `d_ff = round(8/3 * d_model / 64) * 64`, the nearest multiple of 64
  with a minimum of 64. Examples: 64 -> 192, 256 -> 704, 512 -> 1344, 768 -> 2048.
  SiLU is written by hand as `x * sigmoid(x)`. Sub-layers are named `w1` (gate),
  `w2` (output), `w3` (value).
- **RoPE**: features are paired as (0,1), (2,3), ... (interleaved layout, not the
  half-split layout some libraries use). cos/sin tables are non-persistent
  buffers of shape `(max_seq_len, d_k/2)`. `token_positions` must broadcast
  against `x.shape[:-1]`; with a head dimension pass a `(seq,)` tensor.
- **Attention mask**: boolean, `True` = may attend (same as
  `F.scaled_dot_product_attention`). Blocked scores become `-inf` before softmax.
  The softmax itself runs in at least float32 even under bf16/fp16 autocast.
- **Multi-head attention**: separate q/k/v/output `Linear` layers (not one fused
  matrix). Fused is slightly faster on GPU, same math. RoPE touches q and k only.
  RoPE is passed in as a module, so `rope=None` means NoPE.
- **Transformer**: pre-norm blocks, final RMSNorm, untied LM head. One RoPE module
  is shared by all layers. Weight decay is applied to 2D weights only; norm
  gains are not decayed.
- **Ablation switches** live in `TransformerLM` (`use_rmsnorm`, `post_norm`,
  `use_rope`, `ffn_type`) and as CLI flags in `train.py`. With post-norm or no
  norm there is no final norm. The SiLU FFN uses `d_ff = 4 * d_model` with two
  matrices, which matches the parameter count of SwiGLU with `8/3 * d_model`.
- **cross_entropy**: logits upcast to float32; log-sum-exp with max subtraction;
  mean over every position.
- **AdamW**: same operation order as `torch.optim.AdamW` (decay first, then the
  Adam step with `eps` added after dividing by the bias-corrected root of `v`).
  That is why the test can compare results tightly.
- **LR schedule**: `cosine_cycle_iters` is the step where decay ends. After it
  the rate stays at `min_lr`. `train.py` sets it to `max_iters`.
- **Gradient clipping**: one global norm over all parameters, `eps = 1e-6`, only
  ever shrinks gradients. Returns the pre-clip norm.
- **get_batch**: window starts are drawn from `[0, N - context_length - 1]` so
  the shifted target always fits. Returns int64 tensors.
- **Checkpoints**: model + optimizer + iteration, written to a temp file and
  renamed so an interrupted save cannot corrupt the last good checkpoint.
- **BPE vocab layout**: ids 0-255 are bytes, then special tokens in the order
  given, then merges in learned order.
- **BPE tie-breaking**: highest count wins; ties go to the lexicographically
  greater pair of byte strings, compared as `(first, second)`.
- **BPE special tokens**: the text is split on them first, so no merge ever crosses
  one. Longest special token matches first.
- **Tokenizer files**: `save_bpe` writes `vocab.json` (`{"id": "hex"}`) and
  `merges.txt` (`hex hex` per line). Hex is lossless for arbitrary bytes. These
  are not the GPT-2 file formats.
- **encode_iterable** encodes one input string (one line for a file) at a time.
  A whitespace run that spans two lines, such as a blank line, is tokenized per
  line instead of as one token. Round-trip decoding is unaffected.
- **Sampling**: temperature <= 0 means greedy. Top-p keeps a token while the mass
  of the tokens ranked before it is below `top_p`, so the top token always stays.
- **Training resume**: `last.pt` holds weights, optimizer and step. The batch
  sampler is reseeded on resume. The fp16 GradScaler state is not saved.

## Modules

### Linear (`layers.py`)
- **What / why**: a matrix multiply with no bias. Every projection in the model is one.
- **Formula**: `y = x W^T`
- **Shapes**: `x: (..., in)`, `W: (out, in)`, `y: (..., out)`
- **Bug to watch**: storing `W` as `(in, out)` and forgetting the transpose, or
  initializing with the wrong fan (use `in + out`).

### Embedding (`layers.py`)
- **What / why**: learned vector per token id; the model's only view of tokens.
- **Formula**: `e = E[token_id]`
- **Shapes**: `E: (V, D)`, `ids: (...)` -> `(..., D)`
- **Bug to watch**: passing float ids, or ids >= V (index error on CPU, silent
  garbage or a device-side assert on GPU).

### RMSNorm (`layers.py`)
- **What / why**: rescales each token vector to unit RMS so activations stay in a
  stable range. Cheaper than LayerNorm (no mean, no bias).
- **Formula**: `y_i = x_i / sqrt(mean(x^2) + eps) * g_i`
- **Shapes**: `x: (..., D)`, `g: (D,)`
- **Bug to watch**: computing the square in fp16/bf16 (overflow or lost precision).
  Normalize over the last dim only.

### softmax (`layers.py`)
- **What / why**: turns scores into probabilities.
- **Formula**: `softmax(x)_i = exp(x_i - max) / sum_j exp(x_j - max)`
- **Shapes**: same as input; sums to 1 over `dim`
- **Bug to watch**: skipping the max subtraction (overflow to inf/NaN for logits
  above about 88 in float32), or softmax over the wrong dim.

### SiLU and SwiGLU (`layers.py`)
- **What / why**: the feed-forward sublayer, where most parameters live. The gate
  lets the network switch features on and off per token.
- **Formula**: `FFN(x) = W2 (SiLU(W1 x) * W3 x)`, `SiLU(z) = z * sigmoid(z)`
- **Shapes**: `x: (..., D)`, `W1, W3: (F, D)`, `W2: (D, F)`
- **Bug to watch**: forgetting the third matrix when comparing parameter counts, or
  multiplying the SiLU branch by itself instead of by `W3 x`.

### RoPE (`layers.py`)
- **What / why**: encodes position by rotating q and k, so attention scores depend
  on relative distance. No learned parameters.
- **Formula**: pair `i` at position `m` rotates by `m * theta^(-2i/d_k)`:
  `x'_2i = x_2i cos a - x_2i+1 sin a`, `x'_2i+1 = x_2i sin a + x_2i+1 cos a`
- **Shapes**: `x: (..., S, d_k)`, positions `(..., S)`, tables `(max_seq_len, d_k/2)`
- **Bug to watch**: rotating v, mixing the interleaved and half-split pairing
  conventions, or positions that do not broadcast over the head dim.

### Scaled dot-product attention (`attention.py`)
- **What / why**: each query takes a weighted average of values, with weights from
  query-key similarity.
- **Formula**: `softmax(Q K^T / sqrt(d_k)) V`
- **Shapes**: `Q: (..., Sq, d_k)`, `K: (..., Sk, d_k)`, `V: (..., Sk, d_v)`,
  mask `(..., Sq, Sk)`
- **Bug to watch**: omitting `1/sqrt(d_k)`, using the mask the wrong way round
  (here `True` means attend), or a fully masked row (NaN).

### Causal multi-head self-attention (`attention.py`)
- **What / why**: several attention heads in parallel, each looking at a different
  `d_k`-sized slice; the causal mask hides future tokens.
- **Formula**: `concat_h(softmax(q_h k_h^T / sqrt(d_k) + mask) v_h) Wo`
- **Shapes**: `x: (B, S, D)` -> heads `(B, H, S, d_k)` -> `(B, S, D)`
- **Bug to watch**: reshaping `(D)` into `(d_k, H)` instead of `(H, d_k)`, which mixes
  heads silently. Use `rearrange("... s (h d) -> ... h s d")`.

### TransformerBlock (`model.py`)
- **What / why**: one layer = attention then FFN, each wrapped in a residual.
- **Formula**: `x = x + Attn(RMSNorm(x))`, then `x = x + FFN(RMSNorm(x))`
- **Shapes**: `(B, S, D)` in and out
- **Bug to watch**: putting the norm on the residual path (post-norm) by accident;
  adding the residual after the norm output instead of to the input.

### TransformerLM (`model.py`)
- **What / why**: embeds tokens, runs N blocks, normalizes, projects to vocab logits.
- **Formula**: `logits = RMSNorm(Blocks(Embed(ids))) W_head^T`
- **Shapes**: `(B, S) -> (B, S, V)`
- **Bug to watch**: leaking future information. The causality test (change a later
  token, earlier logits must not move) catches this.

### cross_entropy (`utils.py`)
- **What / why**: the training loss; negative log probability of the true next token.
- **Formula**: `loss = log sum_j exp(l_j) - l_target`, computed with the max shifted out
- **Shapes**: `logits (..., V)`, `targets (...)` -> scalar
- **Bug to watch**: applying softmax first and then log (underflow to log 0), or
  averaging over the wrong number of positions.

### AdamW (`optim.py`)
- **What / why**: per-parameter adaptive step sizes with decoupled weight decay.
- **Formula**: see the docstring; `m, v` are moving averages of `g` and `g^2`, step
  size `lr * sqrt(1 - b2^t) / (1 - b1^t)`, and `theta *= 1 - lr * wd`.
- **Shapes**: `m`, `v` match the parameter
- **Bug to watch**: adding weight decay to the gradient (that is Adam with L2, not
  AdamW), or forgetting bias correction (the first steps become far too small).

### LR schedule (`optim.py`)
- **What / why**: warmup avoids unstable early updates; cosine decay settles training.
- **Formula**: warmup `lr = t / T_w * max`; then `min + 0.5 (1 + cos(pi (t - T_w)/(T_c - T_w))) (max - min)`
- **Shapes**: scalar function of the step
- **Bug to watch**: off-by-one at the phase boundaries, and not updating the lr
  in `param_groups` every step.

### Gradient clipping (`optim.py`)
- **What / why**: caps the update size when a batch produces a huge gradient.
- **Formula**: `g <- g * min(1, max_norm / (||g||_2 + eps))` with a single global norm
- **Shapes**: all gradients treated as one long vector
- **Bug to watch**: clipping each tensor separately (changes direction), or clipping
  before `scaler.unscale_` under fp16.

### get_batch (`data.py`)
- **What / why**: next-token training examples from a long token stream.
- **Formula**: `x = t[s : s+T]`, `y = t[s+1 : s+T+1]`
- **Shapes**: `dataset (N,)` -> two `(B, T)` int64 tensors
- **Bug to watch**: an off-by-one that lets `s + T + 1 > N`, or feeding uint16
  tensors to the embedding (convert to int64).

### Checkpoints (`utils.py`)
- **What / why**: survive interruptions (Kaggle timeouts) and resume exactly.
- **Shapes**: n/a. Stores model state, optimizer state (including `m`, `v`, step), iteration.
- **Bug to watch**: saving the model but not the optimizer; Adam restarts cold and
  the loss spikes after resume.

### BPE training (`tokenizer.py`)
- **What / why**: learns a subword vocabulary from data so common strings are single tokens.
- **Formula**: repeat: `pair* = argmax count(pair)`, merge it everywhere
- **Shapes**: output `vocab: {id: bytes}`, `merges: [(bytes, bytes)]`
- **Bug to watch**: letting merges cross pre-token or special-token boundaries, or an
  unstable tie-break (results must be deterministic).

### Tokenizer (`tokenizer.py`)
- **What / why**: text to ids and back, using the learned merges.
- **Formula**: per pre-token, repeatedly apply the lowest-rank merge present
- **Shapes**: `str -> list[int]`, `list[int] -> str`
- **Bug to watch**: applying merges in vocab order instead of merge-rank order, and
  splitting a special token into pieces.

### Generation (`model.py`)
- **What / why**: sample text one token at a time from the model.
- **Formula**: `p = softmax(logits / T)`, keep the top-p nucleus, renormalize, sample
- **Shapes**: logits for the last position `(V,)`
- **Bug to watch**: feeding more than `context_length` tokens (crop the window), or
  using the logits of the first position instead of the last.

### Training loop (`train.py`)
- **What / why**: ties everything together with eval, checkpoints and resume.
- **Bug to watch**: evaluating without `model.eval()`/`no_grad`, or restarting the
  LR schedule from step 0 after a resume (here the schedule uses the restored step).

## Worked example: BPE merges by hand

Corpus (one document, then the special token `<|endoftext|>`):

```
low low low low low lower lower widest widest widest newest newest newest newest newest newest
```

With the GPT-2 pre-tokenizer, spaces attach to the word that follows, so the
pre-tokens are `low` x1, ` low` x4, ` lower` x2, ` widest` x3, ` newest` x6.
`_` below stands for the space byte. Start from single bytes.

| step | top pairs (count) | chosen | why |
|------|-------------------|--------|-----|
| 1 | `(e,s)` 9, `(s,t)` 9 | `(s,t)` | tie, and `s` > `e` |
| 2 | `(e,st)` 9 | `(e,st)` | unique max (next is `(w,e)` 8) |
| 3 | `(l,o)` 7, `(o,w)` 7 | `(o,w)` | tie, and `o` > `l` |
| 4 | `(l,ow)` 7 | `(l,ow)` | unique max |
| 5 | `(_,low)`, `(_,n)`, `(n,e)`, `(e,w)`, `(w,est)` all 6 | `(w,est)` | tie, `w` is the greatest first byte string |
| 6 | `(_,low)`, `(_,n)`, `(n,e)`, `(e,west)` all 6 | `(n,e)` | tie, `n` > `e` > `_` |
| 7 | `(_,low)`, `(_,ne)`, `(ne,west)` all 6 | `(ne,west)` | tie, `ne` > `_` |
| 8 | `(_,low)`, `(_,newest)` both 6 | `(_,newest)` | tie on first, `newest` > `low` |
| 9 | `(_,low)` 6 | `(_,low)` | unique max |
| 10 | `(_,w)`, `(w,i)`, `(i,d)`, `(d,est)` all 3 | `(w,i)` | tie, `w` is the greatest first byte string |

Resulting merges, in order:

```
(s, t)  (e, st)  (o, w)  (l, ow)  (w, est)  (n, e)  (ne, west)  (" ", newest)  (" ", low)  (w, i)
```

With `vocab_size = 256 + 1 + 10 = 267` the vocab is the 256 bytes, then
`<|endoftext|>` at id 256, then `st`, `est`, `ow`, `low`, `west`, `ne`, `newest`,
` newest`, ` low`, `wi` at ids 257 to 266. `tests/test_tokenizer.py` checks exactly
this list for both trainers.

## Worked example: parameter and FLOPs count

Config: `V = 10000`, `S = 256`, `L = 4` layers, `D = 256`, `H = 4`, `F = 704`
(the default for `D = 256`), SwiGLU, pre-norm, untied head.

Parameters:

| part | formula | count |
|------|---------|-------|
| token embedding | `V * D` | 2,560,000 |
| attention per layer | `4 * D^2` | 262,144 |
| SwiGLU per layer | `3 * D * F` | 540,672 |
| two RMSNorms per layer | `2 * D` | 512 |
| one layer total | | 803,328 |
| 4 layers | | 3,213,312 |
| final RMSNorm | `D` | 256 |
| LM head | `D * V` | 2,560,000 |
| **total** | | **8,333,568** |

Without the token embedding table: 5,773,568. `tests/test_model.py` checks this
formula against `num_parameters()` on a small config.

Forward FLOPs per token (a multiply-add counts as 2 FLOPs):

- Matmul weights excluding the embedding: `4 * (262,144 + 540,672) + 2,560,000 = 5,771,264`,
  so the matmuls cost `2 * 5,771,264 = 11,542,528`.
- Attention scores and weighted values: `4 * L * S * D = 4 * 4 * 256 * 256 = 1,048,576`.
- Forward total: `12,591,104` FLOPs per token.
- Training is about 3x the forward pass: `37,773,312` FLOPs per token.
- One step with `32 x 256 = 8192` tokens: about `3.1e11` FLOPs.

Divide by the FLOP/s your GPU actually achieves (usually a fraction of the
datasheet peak) to estimate seconds per step. Memory for weights, gradients and
the two AdamW moments in float32 is about `4 * 8.3M * 4 bytes = 133 MB`;
activations dominate at larger batch sizes and context lengths.
`TransformerLM.flops_per_token()` computes the same formula.

## Ablations

The four switches, what each tests, and the command lines are in
[scripts/run_ablations.md](../scripts/run_ablations.md).
