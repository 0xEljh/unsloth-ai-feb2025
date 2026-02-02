# Unsloth Challenge 2025 Solutions

My solutions to the [Unsloth Challenge](https://x.com/danielhanchen/status/1891194528931209644), a set of open puzzles focused on optimizing LLM fine-tuning through Triton kernels, distributed training, compilation, and memory-efficient techniques.

## Table of Contents

- [Overview](#overview)
- [Puzzle A: NF4 Triton Dequantization](#puzzle-a-nf4-triton-dequantization)
- [Puzzle B: QLoRA + FSDP2](#puzzle-b-qlora--fsdp2)
- [Puzzle C: torch.compile without Graph Breaks](#puzzle-c-torchcompile-without-graph-breaks)
- [Puzzle E: Memory Efficient Backpropagation](#puzzle-e-memory-efficient-backpropagation)
- [Results Summary](#results-summary)
- [Running the Code](#running-the-code)
- [Requirements](#requirements)

---

## Overview

The Unsloth challenge consists of five problems (A-E), with A, B, and C being slightly related due to their use in/of QLoRA/NF4 quantization. D referred to a set of opensource bounties in Unsloth's repo.

| Puzzle | Focus Area | Difficulty | Max Points |
|--------|-----------|------------|------------|
| A | Custom Triton kernel for NF4 dequantization | Hard | 14 |
| B | FSDP2 distributed training with QLoRA | Medium-Hard | 10 |
| C | torch.compile optimization for QLoRA | Easy-Medium | 9 |
| D | (Not included) | - | - |
| E | Memory-efficient logit computation | Medium-Hard | 10 |

---

## Puzzle A: NF4 Triton Dequantization

### Problem Statement

Convert NF4 (4-bit NormalFloat) quantized tensors to fp16/bf16 in a **single Triton kernel** that:
- Performs double dequantization (absmax + weight) in one kernel
- Is **≥1.15x faster** than Unsloth's `fast_dequantize`
- Works on Tesla T4 GPUs
- Does not use large intermediate memory buffers
- Works with `torch.compile`

### Background

NF4 quantization (used by bitsandbytes/QLoRA) stores weights in a 4-bit format with block-wise scaling factors. Dequantization requires:
1. Looking up the 4-bit code values from a 16-element codebook
2. Multiplying by the block's absmax (scaling factor)
3. The absmax itself is quantized (double quantization), requiring a second lookup

### Solution Approach

My implementation in [`Unsloth_Puzzles_A_colab_copy.ipynb`](Unsloth_Puzzles_A_colab_copy.ipynb) uses the following optimizations:

#### 1. Fused Multiply-Add for Absmax Calculation
```python
absmax = tl.fma(absmax1_val, absmax2_val, offset)  # fuses multiplication and addition
```

#### 2. Reduced Memory Reads
Instead of reading one absmax per element, we read one per block and use the block structure to broadcast:
```python
# naive approach -> returns a vector like (0, ..., 0, 1, ...); many repeats
weight_block_index = index // (weight_blocksize // 2)
# Read absmax once per weight block, not per element
weight_block_index = start // (weight_blocksize // 2) + tl.arange(0, BLOCK_SIZE // (weight_blocksize // 2))
```
This reduces two reads by a factor of `weight_blocksize` (64), with one being non-contiguous.

#### 3. Cache-Friendly Code Lookup
The NF4 codebook (16 values) is pulled into cache contiguously, then gathered:
```python
code_index_range = tl.arange(0, 16)
_ = tl.load(code_ptr + code_index_range, eviction_policy="evict_last")  # pull codebook into cache
weight1 = gather_nf4_code(code_index1, code_ptr)  # gather from cache
```

#### 4. Interleaved Output Writes
Each byte contains two 4-bit values. Rather than writing to odd/even indices (strided writes), we interleave after computation for a single contiguous write:
```python
weight = tl.interleave(weight1, weight2)
tl.store(out_ptr + weight_index, tl.ravel(weight), mask=weight_index < total_elements)
```

#### 5. Strategic Eviction Policies
- `evict_last` for frequently reused data (code values, offset, absmax)
- `evict_first` for data used once (raw weight bytes)

### Results

| Metric | Unsloth Reference | This Implementation | Speedup |
|--------|-------------------|---------------------|---------|
| Colab timing | ~4.7s | ~3.8s | **~1.25x** |

<img width="645" height="236" alt="image" src="https://github.com/user-attachments/assets/7e5e2a5a-cb5e-4cbc-918d-bc6fdcf5938b" />

The kernel also works with `torch.compile` (with some expected recompilations due to dynamic shapes).

**Links:**
- [Colab Notebook](https://colab.research.google.com/drive/1SSFSmimdWNmkwZxO1y59XP9V11jwPs49?usp=sharing)

---

## Puzzle B: QLoRA + FSDP2

### Problem Statement

Fine-tune Llama 3.1 8B with QLoRA using **PyTorch FSDP2** on 2+ GPUs, demonstrating:
- Full FSDP2 features (offloading, checkpointing, mixed precision)
- `torch.compile` compatibility
- Equivalent loss to single-GPU training
- HuggingFace Trainer/TRL compatibility

### Background

FSDP2 (Fully Sharded Data Parallel) shards model parameters, gradients, and optimizer states across GPUs. The challenge is making bitsandbytes' `Linear4bit` layers work with:
- FSDP's parameter sharding
- Mixed precision policies
- torch.compile tracing
- Checkpoint saving with DTensors

### Solution Approach

The implementation in [`Unsloth_Puzzles_B.ipynb`](Unsloth_Puzzles_B.ipynb) addresses each challenge:

#### 1. Compilable Linear4bit Wrapper
Wraps bitsandbytes layers with explicit dequantization using Puzzle A's kernel:
```python
class Linear4bitCompilable(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        W = triton_dequantize_nf4(self.weight, self.quant_state).to(x.dtype)
        return torch.matmul(x, W.t())
```

#### 2. Mixed Precision Compatibility
Cast weights to match input dtype for proper mixed precision policy support:
```python
W = triton_dequantize_nf4(self.weight, self.quant_state).to(x.dtype)
```

#### 3. CPU Offloading Backend
Add gloo backend for CPU operations:
```python
init_process_group("cuda:nccl,cpu:gloo", timeout=timedelta(seconds=1_800))
```

#### 4. torch.compile Pattern Matching Fix
Patch the weight-only quantized (WOQ) optimization pattern to handle traced subgraphs:
```python
def patch_is_valid_woq_optimization_pattern():
    def fn(match):
        try:
            # ... validation logic ...
        except AttributeError:
            return False  # Handle missing 'scales' with 'meta'
```

#### 5. DTensor-Aware Checkpointing
Patch peft's `save_pretrained` to localize DTensors before saving/checkpointing:
```python
for name, tensor in output_state_dict.items():
    if isinstance(tensor, DTensor):
        tensor = tensor.to_local()
        output_state_dict[name] = tensor
```

#### 6. Bottom-Up Sharding Strategy
Following FSDP2 best practices, shard from the bottom up:
1. First shard `Linear4bitCompilable` (frozen base layers) with `reshard_after_forward=False`
2. Then shard LoRA modules with `reshard_after_forward=True`
3. Finally shard layernorms (outside LoRA tree)

We compile all modules after sharding.

We get to use `reshard_after_forward=False` on all base layers because they don't get updated during training (they are frozen!)

### Results

Loss curves between single GPU and FSDP2 training align closely, demonstrating equivalent training dynamics.

**Links:**
- [Kaggle Notebook (2x T4)](https://www.kaggle.com/code/mxksowie/fsdp2-with-qlora)
- [Colab Notebook (1x T4)](https://colab.research.google.com/drive/19Xag1tT7KM8vZ5FgiHuKLu9dEhFdMtjI#scrollTo=upoSvQmMEJMp)

---

## Puzzle C: torch.compile without Graph Breaks

### Problem Statement

Make `torch.compile` work for QLoRA training with:
- No graph breaks
- Fewer than 30 compilations (60 is definitely wrong)
- Loss matching non-compiled training
- All modules compiled: MLP, attention, layernorms, loss function

### Background

`torch.compile` traces PyTorch operations into optimized graphs. Graph breaks occur when:
- Dynamic control flow is encountered
- Unsupported operations are used
- Data-dependent shapes appear

QLoRA has several problematic areas:
- bitsandbytes `Linear4bit` with custom CUDA kernels
- Dynamic KV cache sizes
- Symbolic shape handling in attention

### Solution Approach

The implementation in [`Unsloth_Puzzles_C.ipynb`](Unsloth_Puzzles_C.ipynb) systematically addresses each issue by patching them out:

#### 1. Replace Linear4bit with Compilable Version
Same wrapper as Puzzle B, using Triton kernel instead of bitsandbytes CUDA:
```python
@torch.compile(fullgraph=False, dynamic=True, options=torch_compile_options)
def forward(self, x):
    W = triton_dequantize_nf4(self.weight, self.quant_state)
    return torch.matmul(x, W.t())
```

#### 2. Flex Attention with Dynamic Shapes
Extensive patches to `torch._inductor.kernel.flex_attention` for symbolic shape handling:

```python
def get_split_k(B, H, Mk, SM=128):
    """Fix handling of symbolic values"""
    B_sym = sympy.sympify(B)
    H_sym = sympy.sympify(H)
    bh = sympy.Max(B_sym * H_sym, 1)
    split_k = SM // bh
    return sympy.Max(split_k, 1)

def next_power_of_2(n_sym):
    """Symbolic next power of 2"""
    k = sympy.integer_log(n_sym, 2)[0]
    is_exact_power = sympy.Eq(sympy.Pow(2, k), n_sym)
    return sympy.Piecewise(
        (sympy.Pow(2, k), is_exact_power),
        (sympy.Pow(2, k + 1), True)
    )
```

#### 3. Size Hints for Symbolic Dimensions
Add guards and hints to help the compiler reason about shapes:
```python
V.graph.sizevars.guard_leq(1, seq_len_q)
kernel_options.setdefault("QK_HEAD_DIM", V.graph.sizevars.size_hint(qk_head_dim, fallback=HEAD_DIM))
```

#### 4. Compiled Attention Forward
```python
@torch.compile(fullgraph=False, dynamic=True, options=torch_compile_options)
def patched_llama_attention_forward(self, hidden_states, position_embeddings, ...):
    # ... standard attention logic ...
    block_mask = create_block_mask(causal_mask, B=None, H=None, Q_LEN=q_len, KV_LEN=kv_len)
    return flex_attention(query, key, value, score_mod=causal_score_mod, ...)
```

#### 5. Regional Compilation
Compile at appropriate granularity to balance graph optimization vs. compilation overhead:
```python
# Compile each layer's components
for child in model.layers:
    child.self_attn = torch.compile(child.self_attn, ...)
    child.mlp = torch.compile(child.mlp, ...)
    child.input_layernorm = torch.compile(child.input_layernorm, ...)
```

### Results

| Metric | Original | Compiled |
|--------|----------|----------|
| Step 1 Loss | 1.5196 | 1.5199 |
| Step 10 Loss | 2.6758 | 2.7632 |

<img width="567" height="453" alt="image" src="https://github.com/user-attachments/assets/7a6a61f8-d66f-465c-ad0c-e54ff78f1b05" />

Losses remain closely aligned with minor divergence due to flex attention tuning. The remaining recompiles (~8) are from dynamic cache shapes, which is acceptable given the memory cost of static caching.

---

## Puzzle E: Memory Efficient Backpropagation

### Problem Statement

Reduce VRAM usage for the final projection layer (`hidden_dim → vocab_size`) by:
- Avoiding full logit materialization (e.g., 4GB for bsz=4, qlen=4096, vocab=128K)
- NOT hard-coding gradients (use autograd)
- Generalizing beyond cross-entropy loss
- Achieving ≥50% VRAM reduction
- Working with GRPO training

### Background

The projection to vocabulary space creates massive intermediate tensors:
```
Memory = batch_size × seq_len × vocab_size × dtype_bytes
       = 4 × 4096 × 128000 × 2 = 4GB (fp16)
```

The key insight is that we can chunk the computation along the sequence dimension:

$$\frac{dL}{dW} = X_1^T \frac{dL_1}{dy_1} + X_2^T \frac{dL_2}{dy_2}$$

### Solution Approach

The implementation in [`Unsloth_Puzzles_E.ipynb`](Unsloth_Puzzles_E.ipynb) creates a custom autograd function:

#### 1. Chunked Forward Pass
```python
class MemoryEfficientLinear(torch.autograd.Function):
    @staticmethod
    def forward(ctx, X, linear, labels, forward_function, reduction):
        X_chunks = torch.chunk(X, num_chunks, dim=seq_dim)
        labels_chunks = torch.chunk(labels, num_chunks, dim=seq_dim)

        outputs = [forward_function(x_chunk, linear, label_chunk)
                   for x_chunk, label_chunk in zip(X_chunks, labels_chunks)]

        # Aggregate based on reduction method
        if reduction == "mean":
            return sum(out * size for out, size in zip(outputs, chunk_sizes)) / sum(chunk_sizes)
        else:  # sum
            return sum(outputs)
```

#### 2. Chunked Backward Pass
```python
@staticmethod
def backward(ctx, dY):
    dX = torch.zeros_like(X)

    for chunk_idx, (X_chunk, labels_chunk) in enumerate(zip(X_chunks, labels_chunks)):
        X_chunk.requires_grad = True

        with torch.enable_grad():
            output_chunk = forward_function(X_chunk, linear, labels_chunk)

            # Scale gradient for mean reduction
            scale_factor = chunk_size / total_size if reduction == "mean" else 1.0
            output_chunk.backward(dY * scale_factor)

        dX[..., start:start+chunk_size, :] = X_chunk.grad

    return dX, None, None, None, None
```

#### 3. Reduction-Aware Aggregation
Handle both `mean` and `sum` reductions correctly:
```python
# For mean reduction, scale gradients by chunk proportion
scale_factor = curr_chunk_size / sum(chunk_sizes) if reduction == "mean" else 1.0
output_chunk.backward(dY * scale_factor)
```

#### 4. Label Shift Handling for Causal LM
Pre-shift labels before chunking to avoid losing tokens:
```python
labels = nn.functional.pad(labels, (0, 1), value=ignore_index)
shift_labels = labels[..., 1:].contiguous()
loss = memory_efficient_linear(hidden_states, lm_head, shift_labels, forward_fn, "sum")
```

#### 5. GRPO Integration
Patch GRPO's loss computation to use memory-efficient linear for both policy loss and KL divergence:
```python
def compute_loss(self, model, inputs, ...):
    # Stack references: input_ids, ref_logps, completion_mask
    ref_stack = torch.cat([input_ids, ref_per_token_logps, completion_mask], dim=0)

    def forward_function(chunk, linear, ref):
        logits = linear(chunk)
        log_softmax = selective_log_softmax(logits, ref_input_ids)
        per_token_kl = torch.exp(ref_logps - log_softmax) - (ref_logps - log_softmax) - 1
        per_token_loss = torch.exp(log_softmax - log_softmax.detach()) * advantages
        return (-(per_token_loss - self.beta * per_token_kl) * mask).sum()

    return memory_efficient_linear(hidden_states, lm_head, ref_stack, forward_function, "sum")
```

### Results

| Test Configuration | VRAM Saved |
|-------------------|------------|
| CE Loss (mean) | 57.0% |
| CE Loss (sum) | 63.8% |
| MSE Loss | 64.9% |
| Llama ForCausalLMLoss | 31.6% |

All tests pass gradient matching assertions (`rtol=1e-3, atol=1e-5`).

**GRPO Training**: Successfully trained Llama 3.1 8B with memory-efficient loss computation, producing coherent reasoning outputs.

---

## Results Summary

| Puzzle | Goal | Achievement |
|--------|------|-------------|
| A | ≥1.15x speedup | **~1.17-1.25x** speedup |
| B | FSDP2 + QLoRA + compile | **Working** with equivalent loss |
| C | No graph breaks | **~8 recompiles** (from cache dynamics) |
| E | ≥50% VRAM reduction | **50-65%** reduction |

---

## Running the Code

### Prerequisites

```bash
pip install -r requirements.txt
```

### Running Notebooks

Each puzzle has a self-contained Jupyter notebook:

```bash
# Puzzle A - Run locally or on Colab
jupyter notebook Unsloth_Puzzles_A.ipynb

# Puzzle B - Requires 2 GPUs (use Kaggle)
# See: https://www.kaggle.com/code/mxksowie/fsdp2-with-qlora

# Puzzle C
jupyter notebook Unsloth_Puzzles_C.ipynb

# Puzzle E
jupyter notebook Unsloth_Puzzles_E.ipynb
```

### Running the Script Version (Puzzle B)

```bash
python Unsloth_Puzzles_B_script.py
```

---

## Requirements

- Python 3.10+
- PyTorch 2.6+
- Triton 3.0+
- CUDA 12.x
- bitsandbytes
- transformers
- peft
- trl
- accelerate

**Hardware:**
- Puzzles A, C, E: Single GPU (tested on T4, RTX 3080 Ti)
- Puzzle B: 2+ GPUs (tested on 2x T4)

---

## License

Apache License 2.0 - See [LICENSE](LICENSE) for details.

## Acknowledgments

- [Unsloth](https://github.com/unslothai/unsloth) for the challenge and reference implementations
- [bitsandbytes](https://github.com/bitsandbytes-foundation/bitsandbytes) for quantization foundations
- [PyTorch](https://pytorch.org/) for FSDP2 and compile infrastructure
