
+++
title = "LLM Serving"
date = "2024-12-09"
tags = ["Systems", "AI"]
description = "A summary of the computational characteristics of LLM serving and its optimization opportunities."
showFullContent = false
+++

LLM inference differs from small-model inference: it is far more memory-bound, and its inputs and outputs exhibit unique patterns that can be exploited, naturally giving rise to system-level optimization opportunities. I roughly divide these system-level optimization opportunities into three categories: batching optimizations, sampling optimizations, and model compression.

Batching optimizations feed the compute units of accelerators like GPUs more fully on the input side and make better use of SRAM locality[^5]; sampling optimizations[^6] deliver LLM results faster on the output side; and model compression keeps large models from being so large.

## The Computational Shape of LLM Serving
Compared with the previously mainstream serving of deep models, LLM serving has some unique characteristics and constraints:
- Much of the time in a single LLM call is spent loading model parameters (HBM->SRAM).
    - Batching lets one parameter load serve the inference of multiple seqs, significantly reducing the overall parameter-loading overhead.
- Seq2Seq: variable-length input, variable-length output.
    - The batching mechanism must therefore cope with both batch size and seq len being variable.
- Autoregressive (a request takes multiple iterations to process), with a cross-iteration KV maintained for each conversation.
    - A purely stateless approach would recompute all KVs every time, an unacceptably large overhead, so incremental decoding based on KV caching is a must.
    - RNNs are also autoregressive, but unlike RNNs, the transformer's KV cache grows with every iteration.
    - Non-autoregressive models are typically batched at the request granularity, which does not apply to the autoregressive case; the latter is clearly better suited to iteration-granularity batching.
- Unlike the original transformer, GPT is decoder-only.
    - The variable-length batching padding strategies of earlier encoder-only and encoder-decoder architectures do not apply to LLMs.
- GPT has an additional, upfront, compute-intensive prefill stage (also called infill) for processing the prompt.
    - The prefill stage and the incremental iteration stage are hard to batch together (batching works well only when the computation is exactly the same).
    - This stage digests the prompt. Given that many LLM applications have very large prompts with largely identical content, much of the prefill computation is redundant.
- After generating the token probability distribution, GPT has a trailing sampling stage: selecting a token from the vocab based on probability density.
    - Once token selection is complete, the token chosen in the current iteration must again be fed as the input to the next iteration's forward pass.
    - If seq[A~A+5] has already been sampled in previous iterations, the new iteration must recompute seq[A~A+5] in addition to sampling seq[A+6].
    - The decoding results of already-processed tokens can be cached.
    - Many LLM applications require structured output with a fairly fixed format.
- The shape of the input tensor to the transformer's attention operator depends on the length of already-processed tokens.
    - Different sequences have different lengths, so the input shapes of the attn computation are not uniform, making batching difficult.
    - Fortunately, whether the attn computation is batched or not does not matter much, because attn involves no model weights, and thus there is no speedup from reusing one parameter load across multiple sequences.
- A key bottleneck of running LLMs on GPUs is the cost of loading data (requests + parameters) into HBM. LLM serving throughput is clearly bounded by batch size — that is, how much data can be fit in at once without running out of memory.
    - In a simple static batching implementation, seq_len * nr_seqs + model size determines memory usage. If seq_len is set high but not fully used, nr_seqs shrinks considerably as a result.
    - With memory this tight, model compression techniques such as quantization and pruning naturally become especially important for LLM serving — e.g., awq, gptq, gguf, smooth quant.

## Continuous Batching
As mentioned above, batching LLMs brings many benefits but even more difficulties; there is no particularly simple, obvious batching mechanism tailor-made for GPT.

Orca[^1] was the first to propose a complete batching solution for GPT models: it schedules at the iteration level and, through a necessary selective batching mechanism, excludes the operations that cannot be batched (without this exclusion, the probability that two requests can be batched as a whole is negligible). Although these operations cannot be batched, their impact on overall performance improvement is small.

![orca](https://wujipeng.com/img/orca.png)

## Paged Attention
The Orca approach does not account for the HBM footprint of the KV cache and preallocates max_seq_len by default.

But a monolithic KV cache fragments HBM, and preallocating a huge amount of memory for each seq — which in turn limits concurrency — is where the real bottleneck lies. vLLM optimizes for this by proposing Paged Attention[^2], a paged, page-table-like KV cache technique:
- In the prefill stage, the KV cache is allowed to be organized in pages in non-contiguous memory, so there is no need to preallocate memory for max_seq_len; it can be allocated at runtime.
- Most seqs obviously never reach max_seq_len, so paged attention saves a great deal of memory, which in turn allows the batch size to grow substantially.
- vLLM's implementation does not adopt Orca's selective batching, mainly because its paged attention operator is custom CUDA that can be batched together with non-attn operators. vLLM batches prefill and decoding separately, so overall it does not need a mechanism as complicated as selective batching.
    - But this also prevents fusing prefill and decoding steps. If some prompt is too long and its prefill cost too high, it can indeed block all subsequent decoding batches.

![block_table](https://wujipeng.com/img/block_table.png)

See [Paged Attention](https://jipeng4974.github.io/writeups/paged-attention) for details.

## Dynamic SplitFuse
DeepSpeed-FastGen[^3] proposes SplitFuse, another evolution of Continuous Batching. The idea is to split long-prompt requests into several small steps. These small steps are cheap and can fill scheduling gaps, while keeping the cost of prefill (prompt generation) and decode (token generation) steps consistent, ensuring there are no workloads of different sizes. This yields some throughput improvement, but its main advantage is stabilizing tail latency — a higher floor for online serving scenarios.

![split_fuse](https://wujipeng.com/img/split_fuse.png)

## Quantization and Pruning
Because Nvidia's architecture inherently favors graphics workloads, memory is naturally underprovisioned, so beyond reducing compute, various quantization and pruning techniques also play a critical role in reducing memory footprint for LLM serving.

See [Quantization and Pruning](https://jipeng4974.github.io/writeups/quantization-and-pruning) for details.

## Radix Attention
SGLang adopts Radix attention[^4], which retains the KV of common prefixes in a radix tree, so the lifetime of the KV cache is no longer confined to a single request but truly forms an LRU cache spanning multiple requests — fitting the real-world scenario where prompts are huge and mostly share the same prefix.

## Flash Attention
Continuous batching improves the SRAM locality of non-attn operations; for KV computation, Flash Attention[^8] makes the inner loop of the attn computation fit in SRAM.

![fast_att](https://wujipeng.com/img/fast_attention.png)

See [Flash Attention](https://jipeng4974.github.io/writeups/flash-attention) for details.

## Speculative Decoding
The idea of speculative decoding[^7] is to use two models that share the same tokenizer but differ in size. Assume the large model's latency is roughly N times that of the small model. In the time it takes the small model to output N tokens, the large model takes those N tokens, appends them to the seq to form its input, and outputs 1 token — N+1 tokens generated in total. These N+1 tokens are then sampled with greedy decoding; if the sampling result matches the small model's output, it is used directly; if not, sampling stops, and at the stopping point the original small-model token is replaced with the token from the large model's sampling result. Throughout the process, the large model actually needs only one forward pass: with luck it outputs N+1 tokens in one shot; without luck it outputs 1 token.

![speculative_decoding](https://wujipeng.com/img/speculative_decoding.png)


## Structured Decoding
SGLang implements structured decoding[^4] based on a compressed finite-state machine to accelerate specific structured outputs (such as regex-enabled JSON templates), decoding multiple tokens in one go. Suppose a key in this structured output's JSON template is always "top5 candidate"; then the multi-token phrase "top5 candidate" can be treated as a single token and processed in one iteration.

![structured_decoding](https://wujipeng.com/img/structured_decoding.png)


[^1]: Gyeong-In Yu and Joo Seong Jeong. Orca: A Distributed Serving System for Transformer-Based Generative Models. OSDI 22. [[pdf]](https://www.usenix.org/system/files/osdi22-yu.pdf)
[^2]: W. Kwon, et al. Efficient Memory Management for Large Language Model Serving with PagedAttention. [[pdf]](https://arxiv.org/pdf/2309.06180)
[^3]: C. Holmes, et al. DeepSpeed-FastGen: High-throughput Text Generation for LLMs via MII and DeepSpeed-Inference. [[pdf]](https://arxiv.org/pdf/2401.08671)
[^4]: L. Zheng, et al. SGLang: Efficient Execution of Structured Language Model Programs. [[pdf]](https://arxiv.org/pdf/2312.07104)
[^5]: Accelerators such as GPUs/NPUs/TPUs must load model parameters from off-chip memory into on-chip SRAM before the underlying hardware operators can compute; for larger models, this loading cost is often the real bottleneck. Batching therefore not only improves the utilization of accelerator compute units, but also makes better use of SRAM locality by reusing one copy of the model parameters across multiple requests.
[^6]: Sampling refers to the token-selection process based on density; decoding refers to the entire inference process of the decoder-only transformer.
[^7]: Y. Leviathan, et al. Fast Inference from Transformers via Speculative Decoding. [[pdf]](https://arxiv.org/pdf/2211.17192)
[^8]: T. Dao, et al. FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness. [[pdf]](https://arxiv.org/pdf/2205.14135)
