# Quantization and Pruning

> A summary of two model compression techniques that matter a lot for inference optimization: quantization and pruning.

---

LLMS index: [llms.txt](/llms.txt)

---

Among model compression techniques, quantization is the most practical one, and pruning is also worth a try. Quantization lowers precision; pruning trims away parameters.

# Quantization
The k-bit quantization problem can be viewed as mapping a float value $x$ in the range $(x_{min},x_{max})$ through a quantization function $g(x)$ to an integer value $Q$ in the range $(0,2^{k-1})$, while keeping the overall model accuracy loss as small as possible (well, if no end-to-end model accuracy/perplexity metric is available, the objective can instead be set to minimizing the output MSE/RMSE). $Q=round(g(x))$,

## Uniform vs Non-uniform
Depending on whether the distribution of Q is uniform, quantizers can be divided into uniform quantizers and non-uniform quantizers[^8].

Non-uniform quantization typically uses a few discrete levels to approximate other distributions (the true distribution of $x$ might be lognormal or normal), aiming to improve precision in the denser parts of the value range; the downside is somewhat higher computational cost.

## Affine vs Scale
Within uniform quantization, there are two further choices for the transform function: affine and scale. The former uses an affine function ($g(x)=kx+b$), while the latter uses only ($g(x)=kx$): 0 remains 0 after mapping, and $Q$ and $x$ are symmetric around 0, so it is also called symmetric quantization. It is actually a special case of affine quantization; with the offset removed, the computation is simpler and easier to vectorize.

## PTQ vs QAT
Depending on whether backprop is involved, quantization falls into two categories: PTQ (Post-Training Quantization) and QAT (Quantization-Aware Training). Because of its high training cost, QAT is hard to scale to large models, so PTQ is used more often for large-model quantization.

## Dynamic vs Static 
Static-precision quantization converts weights, activations, and gradients uniformly into low-precision representations, e.g., W8A8 quantization. At inference time, W8A8 is entirely int8 arithmetic, with no need to execute any quantization or dequantization functions.

The parameters of static quantization (scale factor $k$, zero point $b$) are fixed. So how do we pick suitable $k$ or $b$? Static quantization usually collects activation distributions over a calibration set and searches for the optimum that minimizes MSE. But this calibration process can also introduce overfitting to the calibration set. Static quantization works when the input data distribution is well understood and can be properly captured by the calibration set.

Dynamic-precision quantization, also called weight-only quantization, only quantizes weights to low precision while activations stay in high precision (so the model becomes a mixed-precision model). In dynamic quantization, the quantization parameters are computed on the fly at inference time, so no dedicated calibration phase is needed. During inference, the activation precision is adjusted dynamically (the upper bound is the activation precision stored in the model, the lower bound is the weight precision; this requires applying quantization functions to the activations or dequantization functions to the weights), so some floating-point computation is retained.

In general, static quantization suits CNNs, while dynamic quantization suits RNNs and transformers.

The computational cost of the quantizer affects inference performance in mixed-precision models (e.g., GPTQ with int4 weight quantization plus fp16 activations, torchao QAT's 8da4w, or llama.cpp's q4_k). Take llama.cpp's q4_k as an example: some intermediate tensors (such as the sum of two 4-bit values, which has a chance of overflow if stored in 4 bits again) need to be quantized to 8 bits rather than 4 bits. The model tensors can therefore be quantized with a more compute-intensive but lower-error quantizer (e.g., a non-uniform quantizer — expensive to compute, but it does not affect runtime overhead), while intermediate tensors use a cheap uniform quantizer[^10].

## LLM Quantization
- GPTQ[^17]: a one-shot quantization method for large models. It batches all weights into matrices and quantizes them layer by layer, minimizing the output MSE each time. GPTQ uses int4/fp16 mixed-precision quantization: 4-bit for weight quantization, fp16 for activations. GPTQ leverages second-order information for error compensation, but it may overfit the calibration set during reconstruction, causing the model to lose generality.

- AWQ[^18]: a low-bit weight-only quantization method that quantizes only the weights, keeping activations and gradients in full precision. AWQ holds that only 0.1–1% of the weights are salient and that these salient weights should be skipped. The post-activation distribution is more salient than the weights themselves, so AWQ uses the activation distribution to find the weights to skip.

- GGUF: in llama.cpp's k-quant family, qN_0 denotes N-bit scale quantization, qN_1 denotes N-bit affine quantization, and qN_k stands for a special block-wise quantization: the original model weights are split into blocks, each with its own scaling factor simply derived from the maximum value (this is obviously not optimal; an improved version came later, see [^10]). Salient weights are quantized to higher precision and the rest to lower precision — a mixed-precision scheme. Take the q2_k quant as an example: salient weights are quantized to 4-bit, while the other weights use 2-bit. q4_0, by contrast, quantizes all weights uniformly to 4-bit.

- SmoothQuant[^23]: unlike per-channel activation quantization, SmoothQuant applies a smoothing operation to the magnitudes to avoid overly drastic inter-channel variation. SmoothQuant makes the originally very uniform weights slightly uneven, but they remain easy to compute with.

![smooth](https://wujipeng.com/img/smooth_quant.png)

## k-bit Inference Scaling Laws
According to the 35,000 k-bit inference experiments in [^5], with the total model size held constant, 4-bit precision is almost always the optimal choice.

# Pruning
Analogous to PTQ, there is also PTP (Post-Training Pruning). This article mainly discusses PTP — pruning that does not require expensive retraining (recovery can be done via LoRA).

## Structured Sparsity
Structured sparsity prunes convolutions and matrix multiplications along specific dimensions (channel, conv kernel), changing their shapes and producing a smaller model.

LLM-Pruner[^20] and Torch-Pruning[^26] are structured-sparsity methods for LLMs. Isomorphic Pruning[^27] is a recent SOTA method for ViTs and modern CNNs.

## Unstructured Sparsity
Unstructured sparsity sparsifies at the granularity of individual parameters; it does not change the shape of the parameter matrices, it merely zeroes out some of their values. It requires the underlying inference implementation to effectively exploit matrix sparsity for acceleration.

SparseGPT[^24] can be applied to 175B-scale models (one-shot, no retraining) without significantly sacrificing perplexity, reaching 60% unstructured sparsity.

SparseGPT reduces the pruning problem to extremely large-scale sparse regression instances and solves them efficiently with a new approximate sparse-regression solver, capable of sparsifying a 100B-scale model on a single GPU within a few hours.

## Semi-structured Sparsity
N:M Pruning[^25] is a semi-structured sparsity method. Unstructured-sparsity techniques like SparseGPT can be modified and adapted into 2:4 sparsity to gain acceleration on the A100.


[^1]: [Quantization-Aware Training for Large Language Models with PyTorch](https://pytorch.org/blog/quantization-aware-training/)
[^2]: Y Lin, et al. FQ-ViT: Post-Training Quantization for Fully Quantized Vision Transformer. (PTQ, addressing extreme non-uniform distribution in attention maps & serious inter-channel variation in LayerNorm inputs) [[pdf]](https://arxiv.org/pdf/2111.13824)
[^3]: M. Sun, et al. A Simple and Effective Pruning Approach for LLMs(Wanda). [[pdf]](https://arxiv.org/pdf/2306.11695)
[^4]: [Exploiting NVIDIA Ampere Structured Sparsity with cuSPARSELt](https://developer.nvidia.com/blog/exploiting-ampere-structured-sparsity-with-cusparselt/)
[^5]: T. Dettmers, L. Zettlemoyer. The case for 4-bit precision: k-bit Inference Scaling Laws. [[pdf]](https://arxiv.org/pdf/2212.09720)
[^6]: [torchao](https://github.com/pytorch/ao/)
[^7]: [Accelerating Neural Network Training with Semi-Structured (2:4) Sparsity](https://pytorch.org/blog/accelerating-neural-network-training/)
[^8]: Raghuraman Krishnamoorthi. Quantizing Deep Convolutional Networks for Efficient Inference: A whitepaper. [[pdf]](https://arxiv.org/pdf/1806.08342)
[^9]: [Lloyd-Max Quantization ](https://www.khoury.northeastern.edu/home/gsharp/csg142-fall-2006/Lloyd-Max-Quant.pdf)
[^10]: [llama.cpp issue: Investigate alternative approach for Q4 quantization](https://github.com/ggerganov/llama.cpp/issues/397)
[^11]: A. Gholami, et al. A Survey of Quantization Methods for Efficient Neural Network Inference. [[pdf]](https://arxiv.org/pdf/2103.13630)
[^12]: Y. Choukroun, et al. Low-bit Quantization of Neural Networks for Efficient Inference.(low-bit/4bit) [[pdf]](https://arxiv.org/pdf/1902.06822)
[^13]: B. Jacob, et al. Quantization and Training of Neural Networks for Efficient Integer-Arithmetic-Only Inference.(int8) [[pdf]](https://arxiv.org/pdf/1712.05877)
[^14]: T. Dettmers, et al. 8-BIT Optimizer via Block-wise Quantization. [[pdf]](https://arxiv.org/pdf/2110.02861)
[^15]: [llama.cpp issue: Need help to understand q4_0, q4_1, q4_2, q4_3 quantization](https://github.com/ggerganov/llama.cpp/discussions/1121)
[^16]: [A Guide to Quantization in LLMs](https://symbl.ai/developers/blog/a-guide-to-quantization-in-llms/)
[^17]: E. Frantar, et al. GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers. [[pdf]](https://arxiv.org/pdf/2210.17323)
[^18]: Ji Lin, et al. AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration. [[pdf]](https://arxiv.org/pdf/2306.00978)
[^19]: [Low-Rank Pruning of Llama2](https://mobiusml.github.io/low-rank-llama2/)
[^20]: [LLM-Pruner: On the Structural Pruning of Large Language Models](https://arxiv.org/pdf/2305.11627)
[^21]: [llama.cpp quantize the intermediate results to 8-bits instead of 4-bits to gain accuracy](https://github.com/ggerganov/llama.cpp/pull/951)
[^22]: [llama.cpp k-quants](https://github.com/ggerganov/llama.cpp/pull/1684)
[^23]: G. Xiao, et al. SmoothQuant: Accurate and Efficient Post-Training Quantization for Large Language Models. [[pdf]](https://arxiv.org/pdf/2211.10438)
[^24]: E. Frantar, D. Alistarh. SparseGPT: Massive Language Models Can Be Accurately Pruned in One-Shot. [[pdf]](https://arxiv.org/pdf/2301.00774)
[^25]: A. Zhou, et al. Learning N:M Fine-grained Structured Sparse Neural Networks From Scratch. [[pdf]](https://arxiv.org/pdf/2102.04010)
[^26]: G. Fang, et al. DepGraph: Towards Any Structural Pruning. [[pdf]](https://arxiv.org/pdf/2301.12900)
[^27]: G. Fang, et al. Isomorphic Pruning for Vision Models. [[pdf]](https://arxiv.org/pdf/2407.04616)
