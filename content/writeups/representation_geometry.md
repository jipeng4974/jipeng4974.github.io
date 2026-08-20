+++
title = "Geometry of Representations"
date = "2026-08-14"
tags = ["AI", "Math"]
description = "In the iterative loop of representation learning, one should try to estimate the intrinsic dimension of the embedded submanifold and design a reasonable ambient space at both the input and output ends."
showFullContent = false
+++

Consider an audio representation model that takes the logmel of an 8s clip as input. Its embedding pipeline is the following dimensionality-reduction process:

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

1. From the point-wise perspective, a single point $w$ in the signal space is reparameterized into a point in a semantic metric space.

2. From the ambient perspective, the dimension shrinks from 176400 in the signal space to 11008 in the pixel space, is mapped by ViT-L to 1024, then by the proj head to 256, and finally, after $\ell_2$ normalization, onto the 255-dimensional sphere. The learnable part is $f_\theta=\ell_2\circ g_\theta\circ h_\theta$, i.e. the segment $\mathbb{R}^{11008}\to S^{255}$; $\Phi$ is a fixed, upstream compression.
- By ambient we mean the support of the manifold; the most primitive ambient is the waveform signal space, i.e. $\mathbb{R}^{176400}$. After the logmel transform, it is mapped into the 11008-dimensional logmel pixel space. The final $S^{255}$ is likewise usually the ambient of the output submanifold, since the chosen output dimension is typically slightly higher than the intrinsic dimension.
- Ambient spaces — the waveform signal space, the logmel pixel space, the spherical space of the final vectors — are not only directly visible but also human-prescribed. Keep in mind that their dimensions are non-natural and non-intrinsic.
- Is it possible to make the ambient $S^{D-1}$ approximately coincide with the submanifold embedded in it? Difficult, but worth trying: on one hand, tune the hyperparameter $D$ according to the estimated intrinsic dimension of the embedded submanifold; on the other hand, introduce SIGReg/VISReg regularization terms.


> **Definition (manifold hypothesis).** Let $\mu_{\text{in}}=\Psi_\#(\mathrm{law}(c,\alpha))$. Its **support**
> $$\mathcal{M}_{\text{in}}\;=\;\operatorname{supp}(\mu_{\text{in}})\;\subset\;\mathbb{R}^{P}$$
> is, under the manifold hypothesis, an (approximately) smooth $d_{\text{in}}$-dimensional embedded submanifold, with $d_{\text{in}}\ll P$.
> Locally, around any point $x$ there exists a diffeomorphism $\varphi:\ U\subset\mathbb{R}^{d_{\text{in}}}\to\mathcal{M}_{\text{in}}$.

3. From the manifold perspective,

$$
\mathcal{M}_{\text{proj}} \;=\; f_\theta\big(\mathcal{M}_{\text{in}}\big)
\;=\;\operatorname{supp}\big((f_\theta)_\#\,\mu_{\text{in}}\big)\;\subset\;S^{255}
$$
- Every "manifold" discussed in this article is an embedded submanifold of the ambient at its respective level of abstraction, and its intrinsic dimension is the dimension of the tangent space.
- Music JEPA's training data is synthesized from $N$ base audio clips via time-stretching, pitch-shifting, cropping, noise addition, and distortion. The logmel data is therefore the original content multiplied by the augmentation machinery, and the intrinsic dimension of the input manifold consists of a content dimension $d_{\mathcal C}$ and an augmentation dimension $d_{\mathcal A}$:

$$
\underbrace{c\in\mathcal C}_{\text{content}}
\;\times\;
\underbrace{\alpha\in A}_{\text{augmentation}}
\;\xrightarrow{\;\Psi\;}\;
x \in \mathbb{R}^{11008}
$$

- Reducing the intrinsic dimension of the embedded submanifold gets closer to the essence of dimensionality reduction in representation learning — learning invariance between the anchor and its degraded versions: from the input manifold's $d_{\text{in}}\approx d_{\mathcal C}+d_A$ down to the output manifold's $d_{\text{proj}}\approx d_{\mathcal C}$. The model should respect only the content in retrieval tasks while ignoring the augmentations (in this scenario, augmentation is degradation).

- How does one estimate the intrinsic dimension of the embedded submanifold? There are well-established intrinsic dimension estimation methods, such as TwoNN and the Levina–Bickel MLE. The number of points within a neighborhood of radius $r$ grows as $\sim r^{d}$; TwoNN / MLE simply invert the nearest-neighbor distance ratios to recover this $d$.
