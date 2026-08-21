+++
title = "Music JEPA Regularizers"
date = "2026-08-04"
tags = ["AI"]
aiAssisted = true
description = "Revisiting anti-collapse regularizers for Music LeJEPA"
showFullContent = false
+++

This post follows up on the [Music LeJEPA](https://jipeng4974.github.io/writeups/music_lejepa/) work. It walks through the math and source implementations of three regularizers — VicReg, SigReg, VisReg — plus a low-rank local-optimum trap I ran into in practice, and, based on the intuition that "preventing dimensional collapse is far easier than reviving dead dimensions," proposes a λ scheduling strategy for anti-collapse regularization.

## Three Regularizers

All three are anti-collapse regularizers in JEPA-style training, sharing the same framework:

$$L = \lambda \cdot L_{reg} + (1-\lambda) \cdot L_{inv}$$

The proj tensor has shape `(V, B, D)` — V views, batch size B, embedding dimension D. The inv term pulls the projections of different views of the same sample toward each other (L2 loss); the reg term keeps all samples from collapsing to a single point. The three methods differ only in the reg term: what shape they constrain the embedding distribution to, and how.

| Method | Constraint target | Constraint mechanism |
|---|---|---|
| VICReg | Per-dimension variance ≥ γ, decorrelation between dimensions | Per-dimension statistics (second moments) |
| SIGReg | Embedding distribution = isotropic standard Gaussian N(0,I) | Random 1D projections + characteristic-function matching (Epps–Pulley test) |
| VISReg | Also N(0,I), but with center/scale/shape separated | Random 1D projections + sorted quantile matching (sliced-Wasserstein style) |

## VICReg

VICReg[^1] flattens the `(V, B, D)` projections into `N = V·B` samples and imposes two constraints on each dimension.

The variance term uses a hinge loss to push each dimension's standard deviation above a threshold γ:

$$v(Z) = \frac{1}{D}\sum_{d=1}^{D} \max(0,\ \gamma - \mathrm{Std}(Z_{:,d}))^2$$

The covariance term penalizes the off-diagonal entries of the covariance matrix, decorrelating different dimensions:

$$c(Z) = \frac{1}{D}\sum_{i \neq j} \mathrm{Cov}(Z)_{i,j}^2$$

Source code (the full regularizer, excluding the invariance term):

```python
import torch.nn as nn
import torch.nn.functional as F


class VICReg(nn.Module):
    def __init__(self, var_weight=25.0, cov_weight=1.0, gamma=1.0):
        super().__init__()
        self.var_weight = var_weight
        self.cov_weight = cov_weight
        self.gamma = gamma

    def forward(self, z):
        V, B, D = z.shape
        z_flat = z.reshape(V * B, D)
        std = z_flat.std(dim=0) + 1e-6
        var_loss = F.relu(self.gamma - std).pow(2).mean()          # hinge(γ - std)
        z_centered = z_flat - z_flat.mean(dim=0, keepdim=True)
        cov = (z_centered.T @ z_centered) / (V * B - 1)            # D×D covariance matrix
        off_diag = cov.pow(2).sum() - cov.diagonal().pow(2).sum()  # sum of squared off-diagonal entries
        cov_loss = off_diag / D
        return self.var_weight * var_loss + self.cov_weight * cov_loss
```

Its strengths are simplicity, no randomness, and every dimension receiving a deterministic, full-strength gradient. The weaknesses are just as plain — second-moment constraints are loose: a distribution with sufficient variance and diagonal covariance is not necessarily Gaussian (a uniform distribution on a sphere can slip through); and the covariance matrix costs $O(B \cdot D^2)$, which gets expensive in high dimensions.

## SIGReg

SIGReg comes from LeJEPA[^2] and rests on two statistical results.

First, the Cramér–Wold theorem: a D-dimensional distribution is uniquely determined by all of its 1D projections. Randomly sample K unit directions $a_1..a_K$ and make every projection $z \cdot a_k$ follow N(0,1), and you approximate "the entire distribution is an isotropic Gaussian."

Second, the Epps–Pulley test[^3]: a classic 1983 goodness-of-fit test for normality. Its basis is that a distribution is uniquely determined by its characteristic function (Bochner's theorem). The characteristic function of the standard normal is $\varphi(t) = e^{-t^2/2}$ (purely real), so the test compares the empirical characteristic function $\hat\varphi(t) = \frac{1}{N}\sum_j e^{itx_j}$ against it via a weighted L2 distance:

$$T = N \int_{-\infty}^{\infty} \left|\hat\varphi(t) - e^{-t^2/2}\right|^2 e^{-t^2/2}\, dt$$

Expanding the squared modulus into cos/sin real and imaginary parts and evaluating a trapezoidal numerical integral at knots points over $t \in [0, t_{max}]$ (by symmetry, the t<0 part is folded in with doubled weight) yields a differentiable loss. The key point is that this statistic is fully differentiable with respect to the samples — a hypothesis test turns into a regularizer. This is LeJEPA's core observation.

Source code:

```python
import torch
import torch.nn as nn


class SIGReg(nn.Module):
    def __init__(self, *, knots: int = 17, t_max: float = 3.0, num_projections: int = 256):
        super().__init__()
        self.num_projections = int(num_projections)
        t = torch.linspace(0, float(t_max), int(knots), dtype=torch.float32)
        dt = float(t_max) / (int(knots) - 1)
        weights = torch.full((int(knots),), 2 * dt, dtype=torch.float32)
        weights[[0, -1]] = dt                       # trapezoidal rule: half weight at endpoints
        window = torch.exp(-t.square() / 2.0)       # weight function = target characteristic function
        self.register_buffer("t", t)
        self.register_buffer("phi", window)
        self.register_buffer("weights", weights * window)

    def forward(self, proj: torch.Tensor) -> torch.Tensor:
        _, B, D = proj.shape

        A = torch.randn(D, self.num_projections, device=proj.device, dtype=proj.dtype)
        A = A.div_(A.norm(p=2, dim=0, keepdim=True) + 1e-12)   # random unit projection directions

        t = self.t.to(device=proj.device, dtype=proj.dtype)
        phi = self.phi.to(device=proj.device, dtype=proj.dtype)
        weights = self.weights.to(device=proj.device, dtype=proj.dtype)

        x_t = (proj @ A).unsqueeze(-1) * t                        # (V,B,K,knots)
        err = (x_t.cos().mean(-3) - phi).square() + x_t.sin().mean(-3).square()
        statistic = (err @ weights) * B                           # EP statistic per projection
        loss = statistic.mean()
        return loss
```

Line by line against the math: `proj @ A` is the Cramér–Wold slicing, projecting the embeddings onto K=256 random directions; `x_t.cos().mean(-3)` and `x_t.sin().mean(-3)` are the real and imaginary parts of the empirical characteristic function (averaged over the batch); the imaginary part in the error term is squared directly because the standard normal's characteristic function is purely real; `err @ weights` is the trapezoidal integral weighted by $e^{-t^2/2}$, and ×B is the N-fold scaling of the EP statistic.

SIGReg constrains the full distribution rather than just second moments, so in theory it strictly prevents collapse, at a cost of $O(B \cdot D \cdot K)$ — usually much less than VICReg's $O(B \cdot D^2)$. The downsides: the random projections introduce Monte Carlo noise, and the penalties for mean shift, overall scale, and distribution shape are all mixed into a single statistic, so the gradient signal-to-noise ratio is mediocre.

## VisReg

VisReg targets the same N(0,I), but deterministically splits the constraint into three parts — center/scale/shape — replacing SIGReg's "everything rides on random projections + characteristic function" approach.

Center — zero the mean:

$$L_{center} = \|\bar{z}\|^2$$

Scale — the RMS radius from the mean equals 1:

$$L_{scale} = \left(\frac{\|z - \bar z\|_2}{\sqrt{B}} - 1\right)^2$$

Shape — after centering and normalizing, project onto K random directions and align the sorted sample values with the theoretical quantiles of the standard normal. The B uniform quantile points of the standard normal are given by the inverse error function:

$$q_i = \sqrt{2}\,\mathrm{erf}^{-1}\!\left(\frac{2i}{B+1} - 1\right),\quad i = 1..B$$

$$L_{shape} = \frac{1}{K}\sum_{k}\frac{1}{B}\sum_{i}\left(\mathrm{sort}(z_{norm} \cdot a_k)_i - q_i\right)^2$$

This is essentially a squared sliced 1D Wasserstein-2 distance — in 1D, the optimal transport for W₂ is exactly sorted alignment, and randomizing the directions corresponds to sketching an isotropic Gaussian.

```python
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class VISReg(nn.Module):
    def __init__(self, num_projections: int = 256, scale_weight: float = 1.0,
                 shape_weight: float = 1.0, center_weight: float = 1.0):
        super().__init__()
        self.K = num_projections
        self._cached_B = -1
        self._cached_target = None
        self.scale_weight = scale_weight
        self.shape_weight = shape_weight
        self.center_weight = center_weight

    def _get_target(self, B: int, device) -> torch.Tensor:
        if self._cached_B != B:                      # target quantiles depend only on B; cache and reuse
            q = torch.linspace(1, B, B, device=device, dtype=torch.float32) / (B + 1)
            self._cached_target = torch.erfinv(2 * q - 1).mul_(math.sqrt(2))
            self._cached_B = B
        return self._cached_target.to(device=device)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        _, B, D = z.shape

        mu = z.mean(dim=1, keepdim=True)
        center_loss = mu.pow(2).mean()                               # center

        z_centered = z - mu
        std = z_centered.norm(dim=1).div(math.sqrt(B)).clamp_min(1e-6)
        scale_loss = (std - 1.0).pow(2).mean()                       # scale

        z_norm = z_centered / std.detach().unsqueeze(1)              # detach: decouple shape from scale
        W = F.normalize(torch.randn(D, self.K, device=z.device, dtype=z.dtype), dim=0)
        p_sorted = (z_norm @ W).sort(dim=1).values                   # K random projections, each sorted
        target = self._get_target(B, z.device).view(1, B, 1)         # N(0,1) theoretical quantiles
        shape_loss = (p_sorted - target).pow(2).mean()               # sorted alignment = 1D W2

        return self.scale_weight * scale_loss + self.shape_weight * shape_loss \
             + self.center_weight * center_loss
```

Note the `std.detach()` line — scale learning is left entirely to scale_loss, while shape_loss only asks "does the shape look normal," and their gradients never interfere. This is why VisReg's gradients are cleaner than SIGReg's, and it is also the setup for the trap described later. Center and scale use deterministic closed-form constraints (no Monte Carlo noise); only shape uses random projections; hyperparameters like the t grid and knots are also done away with.

## Practical Observation: erank Crashes, Recovers, Then Stalls

My setup: an anchor music clip and a degraded music clip are aligned via the invariance objective, with a conventional λ-weighted regularizer applied, and effective rank (the entropic effective dimensionality of the embedding covariance spectrum) monitored for collapse. At random initialization, feeding music spectra into the model yields embeds with an erank of roughly 20–30. The training curve shows three phases:

- It crashes rapidly to ~6 early on — call it dimensional collapse;
- It slowly climbs back to ~100;
- Growth falls to nearly zero and it stalls.

This is what SigReg does; VisReg does slightly better[^4]; VICReg was not tried.

If λ is not set fairly large early in training (larger than is common in natural-image training), the inv loss converges quickly and collapse simply overwhelms the regularizer.

Another anomalous experiment: with VisReg + λ=0.99 held for the first 3000 steps, erank does not rise early on but instead declines slowly — while the VisReg loss decreases in lockstep. On the surface the embedding is getting closer to an isotropic Gaussian, yet the number of dead dims grows. In practice I've never seen anyone actually use λ=0.99; I tried it and the result was unexpected, but it is mathematically explicable.

## The Low-Rank Local-Optimum Trap

Conclusion first: VisReg's global optimum really is N(0,I); the problem is not the target but the gradient structure along the optimization path. The loss measures the shape of marginal distributions; erank measures the covariance spectrum, and the two can move in opposite directions along the optimization path. There are four specific mechanisms.

**(a) The recovery gradient is diluted and stochastic.** The leading term of what the shape term measures on a Gaussian cloud N(0,Σ) along a random direction a is equivalent to W₂², i.e. $(a^\top\Sigma a - 1)^2$. Taking the expectation over random unit directions:

$$\mathbb{E}_a\big[(a^\top\Sigma a - 1)^2\big] \propto \frac{\|\Sigma - I\|_F^2}{D}$$

Σ=I is indeed the global optimum, but a single dying eigenvalue contributes only O(1/D) to the expected loss, and each step estimates it with Monte Carlo over just K=256 directions, giving per-step noise of ~1/√K. The invariance collapse pressure, by contrast, is deterministic and full-strength every step — any dimension sensitive to the degradation is a source of inv loss, and the gradient cuts it off directly. Deterministic collapse versus stochastically diluted recovery: it is obvious who wins early on.

**(b) Self-suppression: recovery force ∝ defect size.** The smaller the eigenvalue λ_j, the smaller the residual it leaves in the sketch, and the weaker the recovery gradient. Moreover, in deep collapse the B samples nearly coincide on the projection, so the relative ordering in sorted matching is almost random and the gradient is dominated by permutation noise. The stall after erank climbs back to ~100 is not "running out of momentum"; it is a fixed point where the recovery force has decayed to balance the inv collapse pressure.

**(c) The factor-dropping escape path.** The whitening from `std.detach()` makes the shape term immune to per-dimension variance, leaving only scale_loss's $(std_d - 1)^2$ to detect dead dims — and it only asks "is the variance 1," never "where does the variance come from." Imagine the embedding dimensions are a linear mixture of k latent factors: after the network drops one factor, it can re-mix the remaining k−1 factors so every coordinate dimension still has std=1, mean 0, and a bell-shaped marginal. At that point center, scale, and shape are all green, the VisReg loss keeps decreasing, and erank drops one notch. The "loss down, dead dims up" phenomenon in the λ=0.99 experiment is exactly this mechanism.

**(d) λ=0.99 walks into the trap on its own.** Useful rank expansion can only be driven by the invariance task — reg only wants the cloud to look Gaussian and does not care how samples are arranged within it (sorted matching is permutation-invariant within each projection). Setting 0.99 cuts the inv anchor to 0.01: the relative geometry of samples loses its maintenance, and discriminative factors are slowly eroded by weight decay and noisy gradients. When reg dominates, the nearest Gaussian-like configuration from the current cloud is "compress and rearrange existing structure" rather than "expand into new directions" — expansion requires encoding new input information, and there is no inv signal to organize it. A regularizer that is too pure means optimizing an objective orthogonal to information content, and the low-rank local solution becomes the direction of steepest descent.

## λ Scheduling: Prevention over Rescue

The downhill slope of collapse is deterministic, full-strength, and direct per-dimension; the uphill climb of revival is diluted by O(1/D), noisy at 1/√K, and has recovery force ∝ defect size — the deeper a dimension dies, the harder it is to save, and as λ_j→0 the gradient SNR tends to zero. The asymmetry is extreme. So the primary goal of λ scheduling is not "pull erank back after collapse" but "never let collapse happen at all." Once erank crashes to single digits, most of the remaining training is spent paying off early debt — and it can only be paid back up to the fixed point.

Concrete strategies:

- Don't use extreme λ. 0.99 cuts the inv anchor and walks into the trap; too small and inv converges instantly and collapse follows. Take the middle-to-larger range (0.5–0.8, fine-tuned against erank monitoring).
- Set λ near its target value from step 0 (or reach it after only a short warmup of a few hundred steps). Waiting until you see erank drop to raise λ is too late — dead-dim recovery is slow and incomplete. Never let inv run unsupervised.
- Use erank, not loss, as the scheduling signal. A decreasing VisReg loss does not prove a healthy distribution (the factor-dropping path); erank is the direct measure of the spectrum. Set a lower alarm bound (e.g. 70% of the initial value) and raise λ or lower the lr when it is breached.
- Confirm which space erank is computed in. VisReg acts on proj; if you only monitor emb, there is the projector-hiding problem — the projector can whiten a collapsing emb into a Gaussian proj. Monitor erank of both emb and proj.
- Use the per-dimension std histogram as an auxiliary diagnostic: if all stds ≈ 1 while erank drops, that confirms factor dropping (correlation collapse); if some dims have std→0, the scale term is being overpowered by inv, meaning λ is insufficient.

## Monitoring Metrics: What Else Is Needed Beyond erank

erank only sees the entropy of the spectrum; the VisReg loss only sees marginals over K random sketches — each has blind spots. Approaching an isotropic Gaussian has three orthogonal failure modes: anisotropy (Σ≠I, second order), non-Gaussianity (shape, higher order), and inter-dimensional dependence (joint structure). No single scalar covers them all, so use a tiered suite.

First, per-dim std should be added to diagnose dead dims. std=0 is a fully dead dimension — constant across the whole batch, carrying no information that distinguishes samples. std=1 merely hits the scale term's target value — it says "not dead," not "useful": the variance could come from real information, from noise, or from the re-mixing of other factors.

The second and third steps are introducing the correlation matrix and kurtosis, to judge whether the variance is "independent and Gaussian."

Finally, in pursuit of perfection, one can also compute the multivariate closed-form version of the Epps–Pulley test, yielding a monitoring suite tiered across time scales:

- Every step, lightweight metrics (O(D³) eigendecomposition):
    - Per-dimension std histogram + erank.
    - log det Σ[^5].
    - −log det R[^6].
- Every 3k steps, medium metrics (O(B·D²)):
    - Mean/max of per-dimension excess kurtosis and skew — when the spectrum is perfectly flat but every dimension is leptokurtic and heavy-tailed, spectral metrics are completely blind and kurtosis exposes it immediately. All per-dimension marginals being normal does not imply joint normality (the copula counterexample), so this tier only checks necessary conditions.
    - Mahalanobis χ² test: if z~N(μ,Σ), the squared Mahalanobis distance d²=(z−μ)ᵀΣ⁻¹(z−μ) follows χ²_D. Compute the empirical distribution of d² over the batch and make a QQ plot against χ²_D quantiles, or compress it to a single scalar with the KS distance — this tests both the radial profile (the shell structure of a Gaussian) and ellipticity at once, and is sensitive to second-moment escapees such as "spectrum close to I but shaped like a uniform distribution on a sphere."
- Every 10k steps, heavyweight metrics (O(B²·D)):
    - The BHEP statistic — the multivariate closed-form version of the Epps–Pulley test. With a Gaussian kernel as the weight, the integral has an analytic solution, requiring neither numerical integration nor random directions:

$$T_\beta = \frac{1}{B^2}\sum_{j,k} e^{-\frac{\beta^2}{2}\|Y_j - Y_k\|^2} - \frac{2}{B}(1+\beta^2)^{-D/2}\sum_j e^{-\frac{\beta^2\|Y_j\|^2}{2(1+\beta^2)}} + (1+2\beta^2)^{-D/2}$$
    where Y_j=Σ^{-1/2}(z_j−z̄) are the studentized samples and β≈1 suffices. Because the characteristic function uniquely determines the distribution, it is nonzero for any departure from multivariate normality — it cannot be fooled by factor re-mixing and has no sketch noise.



## Improvements Targeting the Low-Rank Blind Spot and Dead-Dim Revival

1. In early training, consider using VICReg's deterministic constraints to quickly prop erank up, then switch to VisReg mid-to-late training to let the embedding distribution converge to an isotropic Gaussian.
2. Add VIC's covariance term to VIS.
3. Improve VIS's scale term to avoid excessive dilution: use a stronger scale term to guarantee a floor and force erank up.
4. Consider curriculum learning: don't set the invariance strength too high at the start. Grade the difficulty of the degradation synthesis, starting with the easiest (e.g. slight tempo/pitch shifts, light noise) and increasing difficulty gradually. Overly hard alignment can also push the model to find loopholes in the regularizer and hack them — though such hacking is not entirely useless, as it at least points toward better regularizer designs.

```
# Current: 1/D averaging; a dead dim's distress signal is diluted by the 512 denominator
l_scale = (1.0 - std).square().mean()

# Fix: constant thrust focused on the worst k dims, giving dead dims O(1)-level revival force
k = max(1, std.size(-1) // 8)
worst = std.topk(k, largest=False).values          # bottom-k std
l_revive = (1.0 - worst).square().mean()
l_scale = 0.5 * (1.0 - std).square().mean() + 0.5 * l_revive
```

[^1]: Bardes, Ponce, LeCun. VICReg: Variance-Invariance-Covariance Regularization for Self-Supervised Learning. ICLR 2022. [[arxiv]](https://arxiv.org/abs/2105.04906)
[^2]: Balestriero, LeCun. LeJEPA: Provable and Scalable Self-Supervised Learning Without the Heuristics. 2025. [[arxiv]](https://arxiv.org/abs/2511.08544)
[^3]: Epps, Pulley. A test for normality based on the empirical characteristic function. Biometrika 70(3), 1983.
[^4]: An unrigorous guess: "slightly better" may simply be because sorted matching has stronger recovery force along low-variance directions than characteristic-function matching blunted by the $e^{-t^2/2}$ window — the two have different gradient geometries, and longer training curves are needed to verify.
[^5]: log det Σ = Σᵢ log λᵢ, where λᵢ are the covariance eigenvalues. Geometrically it is the log volume of a high-dimensional ellipsoid — each dimension contributes one log λ; one dead dimension zeroes the volume and log det→−∞, so it is far more sensitive to dying dimensions than erank. Once the scale term locks the trace at D, it attains its maximum at Σ=I; it is also proportional to the Gaussian differential entropy h = ½·log det(2πe·Σ).
[^6]: R is the correlation matrix (Σ normalized to an all-ones diagonal). −log det R measures the overall correlation between dimensions: it is 0 when R=I, and the stronger the correlation, the smaller the determinant. For Gaussian distributions, −½·log det R exactly equals the total correlation, i.e. the sum of mutual information across dimensions — when factors are dropped/re-mixed it rises deterministically, without going through any sketch.
