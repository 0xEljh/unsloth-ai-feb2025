# unsloth-ai-feb2025
Taking on the unsloth puzzle

Overview
--------

[Colab Notebook for A](https://colab.research.google.com/drive/1SSFSmimdWNmkwZxO1y59XP9V11jwPs49?usp=sharing)
- The main optimizations over a "naive dequantization" are:
  - using `fma` to calculate block absmax (fuses multiplication and addition)
  - reading less absmax values (1 per block instead of 1 per element); we can use a matmul to "broadcast" back to the correct shape anyways
    > this reduces two reads by a factor of `weight_blocksize`, of which one of these reads is non-contiguous
  - using tl.where and the raw code values instead of an index lookup (similar to bnb nf4 implementation)
  - cache `offset` and `absmax2` values
  - cache `code` values so it can be "gathered" from cache memory (we don't want a non-contiguous read from global memory)
  - ensure contiguouity of output tensor
  - interleave the final two sets dequantized weights so we can perform a single contiguous memory write (instead of two strided ones). This interleave is done after the matmul with absmax rather than before.

Overall, this leads to a colab timing of `~3.8s` for 1k iterations. Which is a ~1.25+x speedup over the original implementation.

[Notebook for C](/Unsloth_Puzzles_C.ipynb)
- Reused kernel from A
- Compiled loss, mlp, attention, and layernorms.
- Patching out linear4bit solves most of the graph break issues.
- The last source of recompiles is due to the cache. But swapping to the static cache seems to consume lots of memory. Not dealing with this only adds about ~8 recompiles which seems acceptable.
- Added flex attention, enabling dynamic shapes for this requires quite a few patches to handle the symbolic shape parameters: size hints, swapping in sym math, etc.

[Notebook for D](/Unsloth_Puzzles_D.ipynb)
- Implemented the linear efficient backprop without explicitly defining gradient calculation (left to autograd; turns out that's the actual intention of the challenge and that definitely makes it easier)
- Added a wrapper to support awareness of the reduction method (sum or mean), which need to be handled differently (esp. for unequal chunk sizes)
- Using 4 chunks provides >50% vram reduction
- Using a larger number of chunks can cause some drift due to more floating point ops + the nature of the values handled being very small
- Tested with both CE and MSE losses, as well as the llama causal loss function
- Tested with llama, which requires some adaptation of the sequence of operations because of the label shifting
- Used it in GRPO training: Exposed the logit calculation and used `memory_efficient_linear` to calculate both the `loss` and `mean_kl` terms. This requires some "hacking" to put together an appropriate `label` (i.e. the parameters that need to be chunked along with the logits) and also appropriate handling of the reduction.