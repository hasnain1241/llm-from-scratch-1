# Study Notes

One section per module: concept, math, tensor shapes, common bugs.

## Design decisions

(Running list. Add an entry whenever a non-obvious choice is made.)

- Linear: weight stored as (out, in), no bias. Init is truncated normal with
  std = sqrt(2 / (in + out)), cut at +-3 std. `nn.init.trunc_normal_` is used
  for the sampling only, since it is an init helper and not a layer.
- Linear forward uses `einops.einsum` with named dims so the contraction
  axis is explicit.
- Embedding: truncated normal, std 1, cut at +-3. Lookup is plain indexing
  (`weight[ids]`), not a one-hot matmul.
- RMSNorm: input is upcast to float32 for the mean square, gain is applied in
  float32, then the result is cast back to the input dtype. Gain is stored as
  `weight`, initialized to ones. eps defaults to 1e-5.
- softmax: subtract the per-row max before exp. A row that is entirely -inf
  would give NaN (inf - inf); the causal mask never produces such a row
  because each token can always attend to itself.

## Modules

- layers
- attention
- model
- optim
- data
- tokenizer
- train
- utils
