# unsloth-ai-feb2025
Taking on the unsloth puzzle

Overview
--------

[Colab Notebook for A](https://colab.research.google.com/drive/1MnNocyMRQrBL529hdnNJ5myZD5ka_xPl?usp=sharing)
- The main optimizations over a "naive dequantization" are:
  - using `fma` to calculate block absmax (fuses multiplication and addition)
  - reading less absmax values (1 per block instead of 1 per element); we can use a matmul to "broadcast" back to the correct shape anyways
    > this reduces two reads by a factor of `weight_blocksize`, one of which is non-contiguous
  - using tl.where and the raw code values instead of an index lookup (similar to bnb nf4 implementation)
  - cache `offset` and `absmax2` values
  - use custom asm to extract the 4-bit codes from each byte
  - ensure contiguouity of output
  - interleave the dequantized weights for contiguous memory writes (instead of strided). This interleave is done after the matmul with absmax rather than before for performance stability on T4 (affected lower bound of bench sometimes)

Overall, the performance of A seems to be a bit unstable (and from checking auto-tune, its sometimes a little inconsistent with blocksize choice; best `TL_BLOCKSIZE` on T4 is ~128).
Re-running `test_dequantize` can yield a variety of results, albeit slightly bounded.

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