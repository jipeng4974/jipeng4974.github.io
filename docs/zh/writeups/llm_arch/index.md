# Kimi K3: Scaling LLMs Across Sequence, Depth, and Width

> LLM架构演化，从dense transformer到Kimi K3。

---

LLMS index: [llms.txt](/llms.txt)

---

## 1. Core Architecture of LLMs

Modern LLMs can be understood as **multi-layer, multi-head, and increasingly multi-expert** systems.

At a high level:

- **Multi-layer**: `L` Transformer-style decoder layers are stacked vertically, creating depth.
- **Multi-head**: each attention layer uses `H` heads, typically with `d_model = H × d_head`.
- **Multi-expert**: many frontier models replace dense FFN/MLP layers with **Mixture-of-Experts MoE** layers, increasing total parameter capacity while activating only a subset of experts per token.

A standard pre-norm decoder block looks like:

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

The hidden state usually has shape:

```text
(B, T, d_model)
```

where:

- `B` = batch size
- `T` = sequence length
- `d_model` = hidden dimension

Historically, `Q`, `K`, and `V` were often produced by three separate linear projections. In optimized implementations, they are frequently fused into a single projection for better kernel efficiency.

## 2. Attention, Routing, and Knowledge Storage

The `W_q`, `W_k`, and `W_v` projections should not be interpreted as directly inventing new facts. Their main function is closer to **routing and retrieval**:

- `Q` represents what the current token is looking for.
- `K` represents what other positions expose for matching.
- `V` carries the content to be retrieved and mixed.

In this view, attention is primarily a **content-addressed routing mechanism**. It decides where information should flow from and how strongly it should be mixed.

Learned factual and transformation capacity is more heavily associated with:

- attention output projections,
- FFN / MLP layers,
- MoE expert layers,
- and, in newer designs, latent expert projections and gated output paths.

The MLP or MoE part is especially important because it provides nonlinear transformation and acts as a major storage substrate for learned behavior.

## 3. Performance-Oriented Architecture Evolution

The main pressure on LLM architecture is scaling along three axes:

1. **sequence length**,
2. **network depth**,
3. **model width / parameter count**.

The architecture path can be summarized as:

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

Vanilla multi-head attention has quadratic sequence complexity:

```text
O(N²)
```

It is expressive and provides direct global token-to-token interaction, but it becomes expensive for long contexts.

### 3.2 FlashAttention

FlashAttention keeps the standard softmax attention formulation but improves memory access and kernel execution. It mainly changes **how attention is computed**, not what attention computes.

### 3.3 Linear Attention and MLA

Linear attention tries to reduce sequence complexity:

```text
O(N²) → O(N)
```

This improves long-context scalability but can lose some of the exact global retrieval behavior of full attention.

**MLA Multi-head Latent Attention** addresses a different bottleneck: KV-cache size. Instead of caching full head-specific keys and values, MLA caches a compressed latent representation and reconstructs keys and values during attention, reducing KV-cache memory while preserving global attention behavior. Kimi K3 uses Gated MLA as its periodic global-attention layer[[1]](https://github.com/MoonshotAI/Kimi-K3).

### 3.4 DeltaNet and Gated DeltaNet

DeltaNet introduces an efficient recurrent-style memory update based on the Delta Rule, enabling parallelizable `O(N)` sequence processing.

Gated DeltaNet adds a forgetting mechanism through a scalar gate `α`, allowing the model to control how much memory is retained or overwritten.

### 3.5 KDA / Kimi Delta Attention

KDA extends the DeltaNet-style recurrence with a more fine-grained forgetting mechanism. Instead of a single coarse scalar gate, KDA uses **channel-wise retention**, allowing different channels to decay at different rates.

In Kimi K3, KDA is further refined with:

- **channel-wise forget gates**,
- **lower-bounded decay** for numerical stability,
- **chunkwise parallel computation**,
- **full-rank output gating**,
- and specialized kernels such as FlashKDA for efficient execution.

This makes KDA a long-context sequence-mixing mechanism that avoids the continuously growing KV cache of full attention.

## 4. Residual Path Design: AttnRes

Standard residual connections compress all previous layer information into a single stream. This is simple and efficient, but it gives later layers limited selective access to earlier representations.

**Attention Residuals AttnRes** apply an attention-like mechanism across depth. Instead of uniformly accumulating previous layer outputs, a layer can selectively retrieve from earlier representations.

Kimi K3 uses **Block Attention Residuals** rather than full per-layer residual attention. The official report says this reduces memory and communication overhead from `O(Ld)` to `O(Nd)`, where `L` is the number of layers and `N` is the number of residual blocks[[1]](https://github.com/MoonshotAI/Kimi-K3).

For Kimi K3 specifically:

- layers are partitioned into **12-layer AttnRes blocks**,
- this creates **8 full 12-layer blocks plus a partial final block**,
- and the embedding is also included as a residual source.

This is cleaner than saying “AttnRes every 12 layers” only: the actual mechanism is **blockwise depth retrieval over embedding and preceding block outputs**.

## 5. Kimi K3 Architecture

Kimi K3 is best described using three architectural axes:

1. **Sequence mixing**: Hybrid KDA + Gated MLA.
2. **Depth mixing**: Block Attention Residuals.
3. **Width / channel mixing**: Stable LatentMoE.

This is a better abstraction than listing items such as “MLA query LoRA,” “output gating,” and “SiTU activations” at the same level. Those are important implementation details, but the main architectural structure is the three-axis scaling design.

### 5.1 Sequence Mixing: Hybrid KDA + Gated MLA

Kimi K3 uses a repeating **4-layer attention unit**:

```text
3 × KDA layers + 1 × Gated MLA layer
```

This gives a 3:1 ratio between efficient linear-style sequence mixing and full global attention.

Kimi K3 has **93 layers**, 23 × 4 units + 1 final gated MLA layer.

```text
23 repeating attention units:
  each = 3 KDA + 1 Gated MLA
  total = 69 KDA + 23 Gated MLA

Final layer:
  + 1 Gated MLA

Overall:
  69 KDA + 24 Gated MLA = 93 attention layers
```

### 5.2 Depth Mixing: Block Attention Residuals

Kimi K3 uses AttnRes to improve information flow across depth.

Instead of forcing every layer to rely only on the latest residual stream, AttnRes lets layers selectively retrieve from:

- the token embedding,
- preceding block outputs,
- and the current block’s partial accumulated output.

This improves depth-wise information access without paying the full cost of attending over every previous layer at every layer.

### 5.3 Width Mixing: Stable LatentMoE

Kimi K3 uses **Stable LatentMoE** for sparse channel mixing. The model has **896 routed experts**, activates **16 routed experts per token**, and also uses **2 shared experts**[[1]](https://github.com/MoonshotAI/Kimi-K3).

The key idea is that the routed expert path operates in a lower-dimensional latent space. Kimi K3 uses:

```text
d_model = 7168
latent MoE dimension = 3584
```

So the expert routing path works at half the model width, reducing communication and expert-weight traffic while still allowing a large expert pool.

Stable LatentMoE adds several stabilizers:

- RMSNorm before the up-projection,
- **SiTU-GLU** to bound activation growth,
- **Quantile Balancing QB** for expert load balancing.

This matters because extremely sparse MoE at this scale can otherwise suffer from activation instability and routing imbalance.

## 6. Kimi K3 Key Specs

Kimi K3 is a **2.78T-parameter MoE model** with **104.2B activated parameters per token**, a **7168 hidden dimension**, **96 attention heads**, and a **1M-token training context length**[[1]](https://github.com/MoonshotAI/Kimi-K3).

| Design Area | Kimi K3 Choice | Purpose |
|---|---:|---|
| Total layers | 93 | Deeper backbone |
| Attention composition | 69 KDA + 24 Gated MLA | Efficient long-context mixing plus periodic global attention |
| Repeating attention unit | 3 KDA + 1 Gated MLA | 3:1 hybrid attention ratio |
| Extra final layer | 1 final Gated MLA | Ensures final layer has global attention |
| Hidden dimension | 7168 | Main model width |
| Attention heads | 96 | Higher attention parallelism |
| Routed experts | 896 | Larger sparse expert pool |
| Active routed experts | 16 per token | Higher activated capacity |
| Shared experts | 2 | Stable common transformation path |
| Latent MoE dimension | 3584 | Reduces routed-path cost |
| Context length | 1M tokens | Long-context capability |

## 7. NoPE and Long Context

Kimi K3 uses **NoPE**, meaning no explicit positional encoding is applied to MLA queries or keys. Positional and recency information is instead handled implicitly through KDA’s recurrent gating and decay mechanism.

This is important because it avoids context-extension hacks such as RoPE base retuning or interpolation. The report states that Kimi K3 extrapolates to 1M-token contexts without positional-encoding modification[[1]](https://github.com/MoonshotAI/Kimi-K3).

## 8. Major Architectural Changes

A more consistent summary of Kimi K3’s major architecture changes is:

1. **Hybrid sequence mixing**
   - 3 KDA layers followed by 1 Gated MLA layer.
   - Repeated 23 times.
   - Plus one final Gated MLA layer.
   - Final composition: `69 KDA + 24 Gated MLA = 93 layers`.

2. **Blockwise depth mixing**
   - AttnRes lets layers retrieve from embedding and prior block outputs.
   - Implemented blockwise to control training and inference overhead.
   - Uses 12-layer blocks, with a partial final block.

3. **Sparse width/channel mixing**
   - Stable LatentMoE replaces most dense FFN capacity.
   - 896 routed experts, 16 active per token, 2 shared experts.
   - Latent expert path reduces routing cost.

4. **Stability and efficiency mechanisms**
   - SiTU-GLU limits activation explosion.
   - RMSNorm stabilizes latent expert up-projection.
   - Quantile Balancing improves expert load balance.
   - Full-rank output gates are used in both KDA and MLA.

5. **Native multimodal input path**
   - MoonViT-V2 encodes images and videos.
   - A lightweight projector maps visual features into the shared embedding space before backbone processing.

## 9. Design Tradeoffs

Kimi K3 reflects several important tradeoffs:

1. **Linear efficiency vs. global retrieval**
   - KDA provides efficient long-context sequence mixing.
   - Periodic Gated MLA preserves full global token interaction.

2. **Depth access vs. overhead**
   - AttnRes improves selective access to earlier representations.
   - Blockwise AttnRes avoids the full memory cost of per-layer residual attention.

3. **Parameter scale vs. serving cost**
   - MoE gives very large total capacity.
   - Sparse activation keeps per-token compute lower than dense activation.

4. **Expert diversity vs. training stability**
   - 896 routed experts increase specialization.
   - SiTU-GLU, RMSNorm, and Quantile Balancing are needed to keep the system stable.

5. **Long context vs. positional encoding complexity**
   - NoPE plus KDA avoids RoPE retuning when extending context.
   - Gated MLA still provides periodic global attention.

## References
1. [GitHub - MoonshotAI/Kimi-K3: Open Frontier Intelligence · GitHub](https://github.com/MoonshotAI/Kimi-K3)
