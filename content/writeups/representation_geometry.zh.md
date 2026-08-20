+++
title = "表征几何"
date = "2026-08-14"
tags = ["AI", "Math"]
description = "表征学习的反复迭代中，应尝试估计嵌入子流形的内禀维数，在输入和输出环节上设计合理的 ambient space。"
showFullContent = false
+++

考虑以 8s clip的logmel 为输入的音频表征模型，其 embedding pipeline 是如下降维过程：

$$
\begin{gathered}
w\in\mathbb{R}^{176400}
\;\xrightarrow{\;\Phi\;(\text{logmel})\;}\;
x\in\mathbb{R}^{11008}
\;\xrightarrow{\;h_\theta\;(\text{ViT-L})\;}\;
\mathrm{CLS}\in\mathbb{R}^{1024}
\\[6pt]
\;\xrightarrow{\;g_\theta\;(\text{proj head})\;}\;
z\in\mathbb{R}^{256}
\;\xrightarrow{\;\ell_2\;}\;
\hat z\in S^{255}
\end{gathered}
$$

1. 从点视角看，信号空间的某个单点 $w$，被重参数化为语义度量空间的点。

2. 从 ambient 视角看，维数从信号空间的 176400 维，收缩到像素空间的 11008 维，经过 ViT-L，映射到 1024 维，再经过 proj head，映射到 256 维，最后经过 $\ell_2$ 归一化，映射到 255 维的球面。其中可学部分是 $f_\theta=\ell_2\circ g_\theta\circ h_\theta$，即 $\mathbb{R}^{11008}\to S^{255}$ 这一段；$\Phi$ 是前置的固定压缩。
- 所谓 ambient 即流形的 support，最原始的 ambient 就是 waveform 信号空间，即 $\mathbb{R}^{176400}$。经过 logmel 变换，映射到 11008 维的 logmel像素空间。最终的$S^{255}$通常来说也是输出子流形的 ambient，因为实际选择的输出维数往往比内禀维数略高。
- Ambient space——如waveform 信号空间、logmel 像素空间、最终向量的球面空间——不仅是直接可见的，还是人为规定的。需铭记，它们的维度是非自然的，非内禀的。
- 是否有可能让$S^{D-1}$这个 ambient刚好和其嵌入子流形近似重合？困难，但值得尝试，一方面要根据对嵌入子流形内禀维度的估算调整超参D，另一方面引入 SIGReg/VISReg正则项。


> **定义（流形假设）。** 设 $\mu_{\text{in}}=\Psi_\#(\mathrm{law}(c,\alpha))$。其**支撑集**
> $$\mathcal{M}_{\text{in}}\;=\;\operatorname{supp}(\mu_{\text{in}})\;\subset\;\mathbb{R}^{P}$$
> 在流形假设下是一个（近似）光滑的 $d_{\text{in}}$-维嵌入子流形，$d_{\text{in}}\ll P$。
> 局部地，任一点 $x$ 附近存在微分同胚 $\varphi:\ U\subset\mathbb{R}^{d_{\text{in}}}\to\mathcal{M}_{\text{in}}$。

3. 从流形视角看，

$$
\mathcal{M}_{\text{proj}} \;=\; f_\theta\big(\mathcal{M}_{\text{in}}\big)
\;=\;\operatorname{supp}\big((f_\theta)_\#\,\mu_{\text{in}}\big)\;\subset\;S^{255}
$$
- 本文讨论所谓流形都是各抽象层次上 ambient 的嵌入子流形，其内禀维数即切空间维数。
- Music JEPA 的训练数据从 N 个音频基础上进行变速变调裁切加噪失真变换合成。因此logmel数据实际是原始内容乘以增广手段所得，输入流形的内禀维度也包含内容维度$d_{\mathcal C}$和增广维度$d_{\mathcal A}$：

$$
\underbrace{c\in\mathcal C}_{\text{内容}}
\;\times\;
\underbrace{\alpha\in A}_{\text{增广}}
\;\xrightarrow{\;\Psi\;}\;
x \in \mathbb{R}^{11008}
$$

- 嵌入子流形内禀维度的降低，更接近表征学习降维的本质——习得 anchor 和 degraded version之间的invariance。从输入流形的$d_{\text{in}}\approx d_{\mathcal C}+d_A$ 降低到输出流形的$d_{\text{proj}}\approx d_{\mathcal C}$。让模型在检索任务上只尊重内容，而忽略增广（在此场景，增广即退化）。

- 怎么估计嵌入子流形的内禀维数？已有成熟的 intrinsic dimension estimation方法，如TwoNN、Levina–Bickel MLE。半径 $r$ 的邻域内点数按 $\sim r^{d}$ 增长；TwoNN / MLE 就是从最近邻距离比反解这个 $d$。