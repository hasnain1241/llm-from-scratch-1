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

## Modules

- layers
- attention
- model
- optim
- data
- tokenizer
- train
- utils
