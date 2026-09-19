# 量化和剪枝

> 總結推理優化問題中相當重要的模型壓縮技術——量化和剪枝。

---

LLMS index: [llms.txt](/llms.txt)

---

模型壓縮技術中，最實用的是量化，其次還可以嘗試剪枝。量化降低精度，剪枝裁剪參數。

# Quantization
可以將k-bit量化問題視作將取值範圍$(x_{min},x_{max})$的float數值$x$經過量化函數$g(x)$映射到取值範圍$(0,2^{k-1})$的int型數值$Q$，並儘可能減少整體模型精度損失（well，如果沒有端到端的模型accuracy/perplexity指標，也可以把目標設置爲最小化output MSE/RMSE）的問題。$Q=round(g(x))$，

## Uniform vs Non-uniform
根據Q的分佈是否爲均勻分佈，可講量化器分爲uniform quantizers和non-uniform quantizers[^8]。

Non-uniform quantization往往用幾個離散的等級模擬其他分佈（$x$真實的分佈可能是lognorm分佈、norm分佈），以期在更稠密的取值範圍內提升精度，缺點是計算量更大一些。

## Affine vs Scale
在uniform quantization中，變換函數又有兩個選擇：affine和scale。前者用一個仿射函數（$g(x)=kx+b$），後者只用（$g(x)=kx$），0在映射後仍爲0，$Q$和$x$在0兩側對稱，因此又叫對稱量化，實際上是仿射量化的一個特例，由於去掉了offset，計算更簡化了，也容易向量化。

## PTQ vs QAT
根據是否涉及backprop，量化可以分爲PTQ(post-training Quantization)和QAT(Quantization Aware Training)這兩類。QAT因訓練成本高昂，難以擴展到大模型。因此大模型量化更多地使用PTQ。

## Dynamic vs Static 
靜態精度量化把權重、激活函數、梯度統一轉成低精度表示，比如W8A8量化。推理階段W8A8完全是int8算術計算，不需要執行任何量化、反量化函數。

靜態量化的參數（scale factor $k$、zero point $b$）是固定的。那麼如何決定合適的$k$或$b$呢？靜態量化往往需要通過在校準集上收集激活分佈，尋找最小化MSE的最優解。但這種校準過程也會引入過擬合校準集的問題。在輸入數據的分佈非常明確，可以被校準集正確刻畫的場景下，可以使用靜態量化。

動態精度量化，又稱weight-only量化，只把權重量化到低精度，激活仍保留高精度（因此模型變成了混合精度模型）。動態量化中，量化參數是推理時即時演算的，因此無需專門的校準階段。推理時激活精度會動態調整精度（上限是模型中存儲的激活精度，下限是權重精度，需施加量化函數到激活，或反量化函數到權重），因此保留了部分浮點數計算。

通常靜態量化適合CNN，動態量化適合RNN、transformers。

量化器的計算量在混合精度模型（比如int4權重量化+fp16激活函數量化的gptq，或torchao QAT的8da4w，或llama.cpp的q4_k）中會影響推理性能。以llama.cpp的q4_k爲例，中間張量（比如兩個4-bits的sum，再用4-bits有幾率產生overflow）有些需要用8-bits而非4-bits量化，可以把模型張量用計算量更高誤差更少的量化器（比如non-uniform量化器，計算量雖大但不影響運行時開銷），中間張量用計算量小的uniform量化器[^10]。

## LLM Quantization
- GPTQ[^17]：一種適用於大模型的oneshot量化方法，把所有權重都批量放入矩陣中，逐層量化，每次都最小化輸出MSE。GPTQ採用int4/fp16混合精度量化，4-bit用於權重量化，激活函數仍用fp16。GPTQ利用二階信息做誤差補償，但可能在重建過程中過擬合校準集，導致模型損失泛用性。

- AWQ[^18]：一種低比特weight-only的量化方法，只對權重進行量化，將激活函數和梯度保留爲全精度。AWQ認爲只有0.1~1%的權重是salient的，應跳過這些salient權重。激活後的分佈比權重本身更加salient，因此AWQ根據激活分佈尋找需要被跳過的權重。

- GGUF：在llama.cpp的k-quant體系中，qN_0表示N-bits scale量化，qN_1表示N-bits affine量化，qN_k代表特殊的block-wise量化，把原模型權重分塊，每個塊有自己的根據最大值簡單算出的scaling factor（這樣顯然並不最優，後來又做了改進版本，見[^10]），salient權重被量化到高精度，其他的則量化到低精度，是混合精度的。以q2_k quant爲例，salient權重被量化到4-bit，而其他權重則是2bit。q4_0則分別把所有權重都量化到4-bit。

- SmoothQuant[^23]：與per-channel激活量化不同，SmoothQuant把magnitudes做了一個smooth操作，避免過於劇烈的inter-channel variation。SmoothQuant把原本非常uniform的權重分別變得稍有起伏，但實際上仍然易於計算。

![smooth](https://wujipeng.com/img/smooth_quant.png)

## k-bit Inference Scaling Laws
根據[^5]中的三萬五千次k-bit推理實驗，模型總體積不變的情況下，4-bit精度幾乎永遠是最優解。

# Pruning
對應PTQ，也存在PTP(Post-Training Pruning)，本文主要討論PTP，即無需高昂重訓練成本（可以通過LoRA恢復）的剪枝。

## 結構化稀疏
結構化稀疏在特定維度（chanel、conv kernel）上對卷積、矩陣乘做剪枝操作，改變其shape，生成更小的模型。

LLM-Pruner[^20]、Torch-Pruning[^26]是針對LLM的結構化稀疏方法。Isomorphic Pruning[^27]則是近期針對ViT和現代CNN的SOTA方法。

## 非結構化稀疏
非結構化稀疏以每一個參數爲單元進行稀疏，不改變參數矩陣shape，只是令其中部分值爲零。要求底層推理實現能有效地利用矩陣稀疏性進行加速。

SparseGPT[^24]在不顯著犧牲perplexity的前提下，對175B級別的模型使用（one-shot，無需retraining），達到60%非結構化稀疏。

SparseGPT將剪枝問題規約到超大規模稀疏迴歸實例上，用一種新的近似稀疏迴歸求解器高效求解，能在單GPU上幾個小時內跑完100B級模型的稀疏。

## 半結構化稀疏
N:M Pruning[^25]是一種半結構化稀疏方法。SparseGPT這樣的非結構化稀疏技術可以通過修改、適配成2:4 sparsity在A100上得到加速。


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
