+++
title = "Flash Attention"
date = "2024-07-22"
tags = ["Sys", "AI", "LLM"]
description = "Flash Attention in a nutshell: tiling + selective gradient checkpointing."
showFullContent = false
+++

Since `self-attention` has time and space complexity that are both quadratic in the sequence length, long-sequence LLMs and high-resolution ViTs are extremely memory-hungry.

Most prior optimizations of `self-attention` were approximate computations focused on reducing FLOPs — for example, bringing the theoretical time complexity down to O(N) — but this does not effectively speed up `self-attention`, because the real bottleneck of this operation (and of most operations in the transformer) is memory access — more precisely, the IO between HBM and SRAM.

The principle of `FlashAttention` is based on `tiling`: it ensures that the inner-loop computation fits in SRAM, reducing the frequency of IO between HBM and SRAM, thereby genuinely and effectively improving transformer performance and unlocking longer contexts, and it was quickly adopted by various high-performance frameworks.

## The Traditional Self-Attention Implementation
The attention layer has the ability to combine local information with information at distant positions in the tensor — by computing an attention score between every component of the resulting tensor and every component of the input tensor, averaging features across the entire tensor without any locality constraint.

Given an $N^Q\times D^{QK}$-dimensional queries tensor $Q$, an $N^{KV}\times D^{QK}$-dimensional keys tensor $K$, and an $N^{KV}\times D^V$-dimensional values tensor $V$, the ```Attention operation``` $att(K,Q,V)$ produces the $N^Q\times D^V$-dimensional tensor Y:

$$Y = att(K,Q,V) = \underbrace{softargmax(\frac{QK^T}{\frac{1}{\sqrt{D^{QK}}}})}_A V$$

The whole process has two steps. The first step computes the attention score between each query index $q$ and each key index $k$, i.e., the ```softargmax``` of the dot product of queries and keys: $A_{q,k} = \frac{exp(\frac{1}{\sqrt{D^{QK}}} Q_q \cdot K_k ) }{\sum_l exp(\frac{1}{\sqrt{D^{QK}}} Q_q \cdot K_l)}$, where $\frac{1}{\sqrt{D^{QK}}}$ is a scaling parameter that keeps the range of values roughly unchanged as $D^{QK}$ grows.

![att](https://jipeng4974.github.io/img/attention.png)

After obtaining the attention scores $A_{q,k}$, the second step computes: $Y_q = \sum_k A_{q,k}V_k$. An attention score is the degree of match between a query and a key: the better the match, the higher the weight. If a query and a key match to the extreme, the attention score is close to 1, and the value corresponding to that key is taken directly. If the query matches several keys at a moderate level, the result is a weighted average by attention score.

## FlashAttention's IO Optimization
`FlashAttention` observes that the full inputs K, V, Q are not actually needed all at once: they can be read in batches, computed in batches, and the results written back to O in batches — the so-called `tiling`:
- In the outer loop, iterate over the K and V matrices. Each iteration only needs to load one block of $K^T$ and $V$ into on-chip SRAM.
- In the inner loop, iterate over the blocks of Q, load them into SRAM, perform the $att(K,Q,V)$ computation, and write the partial results back to the $N\times d$ result matrix on HBM[^3].
- By adjusting the softmax normalization factor accordingly, the final summed result is guaranteed to be equivalent to the standard implementation; see the appendix of [^1] for the detailed algebra.
- Set the block size of K and V to $\lceil \frac{M}{4d} \rceil$, and the block size of Q and O to $min(\lceil \frac{M}{4d} \rceil, d)$[^4].

![fast_att](https://jipeng4974.github.io/img/fast_attention.png)

Furthermore, for training workloads, `FlashAttention` also reuses the softmax normalization factor $\frac{1}{\sqrt{D^{QK}}}$ cached during the forward pass in backpropagation, which is much faster than reading the huge $N\times N$ intermediate attention matrix from HBM. This can be viewed as selective gradient checkpointing.

## FlashAttention2: Better Parallelism and Work Partitioning
Compared with GEMM, `FlashAttention` only achieves 25~40% of the theoretical FLOPs, leaving huge room for optimization. `FlashAttention2`[^2] improves parallelism and work partitioning on top of the original version.
- Reduce non-matrix-multiplication operators, because GPU matrix multiplication is highly optimized and other operators lag far behind it.
    - Avoid rescaling every time O is computed in the loop; instead, apply the softmax normalization factor when computing the final result.
    - Streamline the state maintained during backpropagation.
- Parallelize the computation across different thread blocks[^8] to make full use of GPU resources.
    - In the original version, one thread block handles one head / one batch, and each thread block runs on one SM. However, in long-sequence scenarios, both the number of heads and the batch size may be small, so their product may not even fill the 128 SMs of an A100.
    - The obviously parallelizable part is the outer loop, which can be freely scheduled onto different thread blocks, with no communication between them at all.
    - When parallelizing the outer loop in backpropagation, simple communication/synchronization is only needed for the dQ update; this is an addition whose order does not matter, so an atomic add is sufficient.

With thread blocks in place, the next question is how to partition the work among different warps within each thread block.
![work_part](https://jipeng4974.github.io/img/work_part.png)

As shown in the figure above, the partitioning scheme of `FlashAttention` requires every warp in the inner loop to write its results to shared memory and perform a synchronized addition, incurring some communication overhead, whereas the partitioning scheme of `FlashAttention2` guarantees that warps have no communication needs at all.


[^1]: T. Dao, et al. FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness. [[pdf]](https://arxiv.org/pdf/2205.14135)
[^2]: Tri Dao. FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning. [[pdf]](https://arxiv.org/pdf/2307.08691)
[^3]: Here, d is the head dimension and N is the sequence length. $N \gg d$. In GPT2, $N=1024, d=64$.
[^4]: Here, M is the SRAM size.
[^5]: M. Milakov, N. Gimelshein. Online Normalizer Calculation for Softmax. [[pdf]](https://arxiv.org/pdf/1805.02867)
[^6]: Markus N. Rabe, Charles Staats. Self-attention Does not Need $O(n^2)$ Memory. [[pdf]](https://arxiv.org/pdf/2112.05682)
[^7]: W. Kwon, et al. Efficient Memory Management for Large Language Model Serving with PagedAttention. [[pdf]](https://arxiv.org/pdf/2309.06180)
[^8]：Multiple thread blocks, and multiple warps within a single thread block, can time-share the same SM.
