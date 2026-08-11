+++
title = "Kimi K3：沿序列、深度与宽度扩展 LLM"
date = "2026-07-28"
tags = ["AI", "LLM", "AI-Assisted", "En"]
description = "LLM架构演化，从dense transformer到Kimi K3。"
showFullContent = false
+++

## 1. LLM 的核心架构

现代 LLM 可以被理解为**多层（multi-layer）、多头（multi-head）、且日益多专家（multi-expert）**的系统。

从高层次来看：

- **多层（Multi-layer）**：`L` 个 Transformer 风格的 decoder 层垂直堆叠，形成深度。
- **多头（Multi-head）**：每个 attention 层使用 `H` 个头，通常满足 `d_model = H × d_head`。
- **多专家（Multi-expert）**：许多前沿模型用 **Mixture-of-Experts（MoE）** 层替换 dense FFN/MLP 层，在增大总参数容量的同时，每个 token 只激活一部分专家。

一个标准的 pre-norm decoder block 如下所示：

```text
Input x
  ↓
LayerNorm
  ↓
QKV projection
  ↓
Multi-Head Self-Attention
  ↓
Output projection
  ↓
Residual Add
  ↓
LayerNorm
  ↓
FFN / MLP or MoE
  ↓
Residual Add
  ↓
Output
```

hidden state 的形状通常为：

```text
(B, T, d_model)
```

其中：

- `B` = batch 大小
- `T` = 序列长度
- `d_model` = 隐藏维度

历史上，`Q`、`K`、`V` 通常由三个独立的线性投影产生。在优化过的实现中，它们常常被融合成单个投影，以获得更好的 kernel 效率。

## 2. Attention、路由与知识存储

`W_q`、`W_k`、`W_v` 投影不应被理解为直接"发明"新的事实。它们的主要功能更接近**路由与检索（routing and retrieval）**：

- `Q` 表示当前 token 在寻找什么。
- `K` 表示其他位置暴露出来供匹配的内容。
- `V` 承载着将被检索和混合的内容。

在这个视角下，attention 本质上是一种**按内容寻址的路由机制（content-addressed routing mechanism）**。它决定信息应从哪里来、以及以多大的强度被混合。

习得的事实性知识与变换能力，更多与以下组件相关：

- attention 的输出投影，
- FFN / MLP 层，
- MoE 专家层，
- 以及在更新的设计中，latent expert 投影与 gated 输出路径。

MLP 或 MoE 部分尤其重要，因为它提供非线性变换，并充当习得行为的主要存储基底。

## 3. 面向性能的架构演化

LLM 架构面临的主要压力是沿三个维度扩展（scaling）：

1. **序列长度**，
2. **网络深度**，
3. **模型宽度 / 参数量**。

其架构演化路径可以概括为：

```text
Vanilla MHA
→ FlashAttention-optimized MHA
→ Linear Attention / MLA
→ DeltaNet
→ Gated DeltaNet
→ KDA / Kimi Linear
→ Hybrid KDA + Gated MLA + MoE + AttnRes
```

### 3.1 Vanilla MHA

原始的 multi-head attention 具有随序列长度平方增长的复杂度：

```text
O(N²)
```

它表达能力强，提供直接的全局 token 间交互，但在长上下文下代价高昂。

### 3.2 FlashAttention

FlashAttention 保持标准的 softmax attention 形式不变，但改进了内存访问与 kernel 执行。它主要改变的是**attention 的计算方式**，而非 attention 计算的内容。

### 3.3 Linear Attention 与 MLA

Linear attention 试图降低序列复杂度：

```text
O(N²) → O(N)
```

这改善了长上下文的可扩展性，但可能损失 full attention 那种精确的全局检索行为。

**MLA（Multi-head Latent Attention）** 解决的是另一个瓶颈：KV cache 的大小。MLA 不再缓存每个 head 各自的完整 key 和 value，而是缓存一个压缩的 latent 表示，并在 attention 计算时重构出 key 和 value，从而在保留全局 attention 行为的同时降低 KV cache 的内存占用。Kimi K3 将 Gated MLA 用作其周期性的全局 attention 层[[1]](https://github.com/MoonshotAI/Kimi-K3)。

### 3.4 DeltaNet 与 Gated DeltaNet

DeltaNet 引入了基于 Delta Rule 的高效 recurrent 式记忆更新，实现了可并行的 `O(N)` 序列处理。

Gated DeltaNet 通过一个标量门 `α` 加入遗忘机制，使模型能够控制保留或覆写多少记忆。

### 3.5 KDA / Kimi Delta Attention

KDA 在 DeltaNet 式递归的基础上引入了更细粒度的遗忘机制。KDA 不再使用单一的粗粒度标量门，而是采用 **channel-wise retention（逐通道保留）**，允许不同 channel 以不同速率衰减。

在 Kimi K3 中，KDA 进一步做了以下改进：

- **逐 channel 的遗忘门（channel-wise forget gates）**，
- 用于数值稳定性的**有下界衰减（lower-bounded decay）**，
- **chunkwise 并行计算**，
- **full-rank 输出门控**，
- 以及 FlashKDA 等专用 kernel，用于高效执行。

这使 KDA 成为一种长上下文序列混合（sequence-mixing）机制，避免了 full attention 中不断增长的 KV cache。

## 4. 残差路径设计：AttnRes

标准的残差连接将此前所有层的信息压缩进单一信息流。这简单且高效，但让后面的层对早期表示的选择性访问能力十分有限。

**Attention Residuals（AttnRes）** 在深度方向上应用类似 attention 的机制。层不再均匀累加此前各层的输出，而是可以有选择地从早期表示中检索。

Kimi K3 使用的是 **Block Attention Residuals**，而非完整的逐层残差 attention。官方报告称，这将内存与通信开销从 `O(Ld)` 降至 `O(Nd)`，其中 `L` 为层数，`N` 为残差块数量[[1]](https://github.com/MoonshotAI/Kimi-K3)。

具体到 Kimi K3：

- 各层被划分为 **12 层的 AttnRes block**，
- 形成 **8 个完整的 12 层 block，外加一个不完整的末尾 block**，
- 且 embedding 也被作为一个残差来源。

这比只说"每 12 层一次 AttnRes"更准确：实际机制是**在 embedding 与此前各 block 输出之上做 block 级的深度检索**。

## 5. Kimi K3 架构

描述 Kimi K3 最好的方式是三条架构轴线：

1. **序列混合（Sequence mixing）**：Hybrid KDA + Gated MLA。
2. **深度混合（Depth mixing）**：Block Attention Residuals。
3. **宽度 / channel 混合（Width / channel mixing）**：Stable LatentMoE。

这比把"MLA query LoRA""输出门控（output gating）""SiTU 激活"等条目并列罗列是更好的抽象。那些是重要的实现细节，但主要的架构结构是三轴扩展设计。

### 5.1 序列混合：Hybrid KDA + Gated MLA

Kimi K3 使用一个重复的 **4 层 attention 单元**：

```text
3 × KDA layers + 1 × Gated MLA layer
```

这使高效 linear 式序列混合与完整全局 attention 之间形成 3:1 的比例。

Kimi K3 共有 **93 层**，即 23 × 4 个单元 + 1 个最后的 gated MLA 层。

```text
23 repeating attention units:
  each = 3 KDA + 1 Gated MLA
  total = 69 KDA + 23 Gated MLA

Final layer:
  + 1 Gated MLA

Overall:
  69 KDA + 24 Gated MLA = 93 attention layers
```

### 5.2 深度混合：Block Attention Residuals

Kimi K3 使用 AttnRes 来改善沿深度方向的信息流动。

AttnRes 不再强迫每一层只依赖最新的残差流，而是让各层可以有选择地从以下来源检索：

- token embedding，
- 此前各 block 的输出，
- 以及当前 block 已部分累积的输出。

这改善了深度方向的信息访问，同时又无需付出"每一层都对所有此前层做 attention"的全部代价。

### 5.3 宽度混合：Stable LatentMoE

Kimi K3 使用 **Stable LatentMoE** 进行稀疏 channel 混合。模型共有 **896 个 routed expert**，每个 token 激活 **16 个 routed expert**，并使用 **2 个 shared expert**[[1]](https://github.com/MoonshotAI/Kimi-K3)。

关键思想是 routed expert 路径在一个更低维的 latent 空间中运作。Kimi K3 的取值为：

```text
d_model = 7168
latent MoE dimension = 3584
```

因此，expert 路由路径在模型宽度的一半上运行，在保留大规模专家池的同时降低了通信量与 expert 权重传输量。

Stable LatentMoE 加入了若干稳定化机制：

- 在 up-projection 之前加 RMSNorm，
- 用 **SiTU-GLU** 约束激活值增长，
- 用 **Quantile Balancing（QB）** 做 expert 负载均衡。

这一点很重要，因为在这种规模下，极度稀疏的 MoE 否则容易出现激活不稳定与路由不均衡。

## 6. Kimi K3 关键规格

Kimi K3 是一个 **2.78T 参数的 MoE 模型**，每个 token 激活 **104.2B 参数**，hidden dimension 为 **7168**，有 **96 个 attention head**，训练上下文长度为 **1M token**[[1]](https://github.com/MoonshotAI/Kimi-K3)。

| 设计方面 | Kimi K3 的选择 | 目的 |
|---|---:|---|
| 总层数 | 93 | 更深的骨干网络 |
| attention 组成 | 69 KDA + 24 Gated MLA | 高效长上下文混合加周期性全局 attention |
| 重复 attention 单元 | 3 KDA + 1 Gated MLA | 3:1 混合 attention 比例 |
| 额外的最后一层 | 1 个 final Gated MLA | 确保最后一层具有全局 attention |
| hidden dimension | 7168 | 主模型宽度 |
| attention head 数 | 96 | 更高的 attention 并行度 |
| routed expert 数 | 896 | 更大的稀疏专家池 |
| 激活的 routed expert | 每 token 16 个 | 更高的激活容量 |
| shared expert | 2 | 稳定的公共变换路径 |
| latent MoE 维度 | 3584 | 降低路由路径开销 |
| 上下文长度 | 1M token | 长上下文能力 |

## 7. NoPE 与长上下文

Kimi K3 使用 **NoPE**，即不对 MLA 的 query 或 key 应用任何显式位置编码。位置与临近性（recency）信息改由 KDA 的 recurrent 门控与衰减机制隐式处理。

这一点很重要，因为它避免了 RoPE base 重调或插值之类的上下文扩展 hack。报告称 Kimi K3 无需修改位置编码即可外推到 1M token 的上下文[[1]](https://github.com/MoonshotAI/Kimi-K3)。

## 8. 主要架构变化

对 Kimi K3 主要架构变化更一致的总结如下：

1. **Hybrid 序列混合**
   - 3 个 KDA 层后接 1 个 Gated MLA 层。
   - 重复 23 次。
   - 外加最后一个 Gated MLA 层。
   - 最终组成：`69 KDA + 24 Gated MLA = 93 层`。

2. **Block 级深度混合**
   - AttnRes 让各层可以从 embedding 和此前 block 的输出中检索。
   - 以 block 为单位实现，以控制训练与推理开销。
   - 使用 12 层的 block，末尾 block 不完整。

3. **稀疏宽度 / channel 混合**
   - Stable LatentMoE 取代了大部分 dense FFN 容量。
   - 896 个 routed expert，每 token 激活 16 个，2 个 shared expert。
   - latent expert 路径降低路由开销。

4. **稳定性与效率机制**
   - SiTU-GLU 限制激活爆炸。
   - RMSNorm 稳定 latent expert 的 up-projection。
   - Quantile Balancing 改善 expert 负载均衡。
   - KDA 与 MLA 中都使用 full-rank 输出门。

5. **原生多模态输入路径**
   - MoonViT-V2 编码图像与视频。
   - 一个轻量的 projector 将视觉特征映射到共享 embedding 空间，再进入骨干网络处理。

## 9. 设计权衡

Kimi K3 体现了若干重要的权衡：

1. **Linear 效率 vs. 全局检索**
   - KDA 提供高效的长上下文序列混合。
   - 周期性的 Gated MLA 保留完整的全局 token 交互。

2. **深度访问 vs. 开销**
   - AttnRes 改善对早期表示的选择性访问。
   - Block 级 AttnRes 避免了逐层残差 attention 的全部内存开销。

3. **参数规模 vs. serving 成本**
   - MoE 提供非常大的总容量。
   - 稀疏激活使每 token 计算量低于 dense 激活。

4. **专家多样性 vs. 训练稳定性**
   - 896 个 routed expert 增强专业化。
   - 需要 SiTU-GLU、RMSNorm 与 Quantile Balancing 来保持系统稳定。

5. **长上下文 vs. 位置编码复杂度**
   - NoPE 加 KDA 避免了扩展上下文时的 RoPE 重调。
   - Gated MLA 仍提供周期性的全局 attention。

## 参考资料
1. [GitHub - MoonshotAI/Kimi-K3: Open Frontier Intelligence · GitHub](https://github.com/MoonshotAI/Kimi-K3)
