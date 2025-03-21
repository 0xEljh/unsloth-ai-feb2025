# unsloth-ai-feb2025
Taking on the unsloth puzzle

Overview
--------

[Colab Notebook for A](https://colab.research.google.com/drive/1MnNocyMRQrBL529hdnNJ5myZD5ka_xPl?usp=sharing)
- Implemented nf4 dequantization in a single Triton kernel. It's faster than the unsloth reference impelmentation for fp16, but slower for bf16 on an NVIDIA T4 in Colab. The speedup is more significant at larger tensor sizes, suggesting that there is significant overhead in this implementation.
- Might need to revisit some of the cache eviction
- The main optimizations over a "naive dequantization" are:
  - using `fma` to calculate block absmax (fuses multiplication and addition)
  - reading less absmax values (1 per block instead of 1 per element); we can use a matmul to "broadcast" back to the correct shape anyways
  - using tl.where and the raw code values instead of an index lookup (similar to bnb nf4 implementation)
- What doesn't work:
  - Interleaving the code indexes (the individual 4-bit values extracted from the byte) to re-obtain a single vector is slower than operating on these sequentially. Similar reshapes and joins/concats are also slower.
  - extracting all 8 bits from a byte via asm seems to be subjective.

[Notebook for C](/Unsloth_Puzzles_C.ipynb)
- Reused kernel from A, with small modifications to autotune params, and some constant fixing for torch compile (also a minor sidestep of `triton.cdiv`)
- Compiled loss, mlp, attention, and layernorms.
- Patching out linear4bit solves most of the graph break issues.
- The sdpa attention and llama attention are further simplified to prevent breaks/recompiles. Might revist to swap with flex attention.
- The last source of recompiles is due to the cache. But swapping to the static cache seems to consume lots of memory. Not dealing with this only adds about ~8 recompiles which seems acceptable.

[Notebook for D](/Unsloth_Puzzles_D.ipynb)
- Implemented the linear efficient backprop without explicitly defining gradient calculation (left to autograd; turns out that's the actual intention of the challenge and that definitely makes it easier)
- Using 4 chunks provides >50% vram reduction
- This implementation still needs to be made aware of the reduction method, in a manual way (changing the variable within the function). Can definitely be improved here.
- Tested with both CE and MSE losses.
- Patching it into the Llama head is still a WIP.