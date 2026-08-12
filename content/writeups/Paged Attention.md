
+++
title = "Paged Attention"
date = "2024-07-30"
tags = ["Sys", "AI", "LLM"]
description = "Paged Attention: improving memory utilization and throughput when serving many concurrent requests."
showFullContent = false
+++

In LLM serving, accumulating enough requests through proper batching can improve LLM throughput. However, the KV Cache for each request is huge — in the naive implementation, the KV Cache has to reserve memory for `max_tokens`, but the actual number of tokens carried by each request is usually far smaller than `max_tokens`, causing massive memory waste and repeated dynamic memory allocation.

The goal of `PagedAttention`[^1] is to eliminate this memory waste and flexibly share KV cache within and across requests. With `PagedAttention`, vLLM achieves 2–4x the throughput of the previous SOTA systems, `FasterTransformer` and `Orca`.

Previously, the naive KV Cache implementation looked like the figure below: a short prompt plus the not-so-long current iteration occupy only 7+4 slots, while the remaining 2038 slots have to be reserved in memory to honor the promise of a maximum sequence length of 2048 (the actual number of tokens is only known once sampling finishes). Only after that can the next request's slots begin — the reserved part in between is completely wasted.

![naive_kv_cache](https://wujipeng.com/img/naive_kv_cache.png)

`PagedAttention` stores logically contiguous KV in non-contiguous memory space, borrowing a page-table-like mechanism (introducing a block_table) to sidestep memory fragmentation. Specifically, `PagedAttention` partitions the KV cache into a number of K blocks and V blocks, where each K/V block holds the K/V vectors corresponding to a fixed number of tokens, so the attention computation is also transformed into blockwise computation. This is somewhat similar to `FlashAttention`, but applied at a different scale: the former overcomes fragmentation and enables on-demand allocation for large-scale serving, while the latter makes a single self-attention computation fit entirely in SRAM.

![block_table](https://wujipeng.com/img/block_table.png)

These physical KV blocks can obviously be reused across multiple requests. As shown in the figure below, maintaining a small block table for each request is all it takes.

![vllm_two_requests](https://wujipeng.com/img/vllm_two_requests.png)


[^1]: W. Kwon, et al. Efficient Memory Management for Large Language Model Serving with PagedAttention. [[pdf]](https://arxiv.org/pdf/2309.06180)
