# Diffusion Probabilistic Models

> This post is my notes on Professor Jun Zhu's talk — "Diffusion Models for Generating High-Dimensional Data". Notably, DPM practice makes clever use of analytical solutions: both the closed form $q(x_N|x_0)$ in the forward process and the analytic-form variance estimate in the reverse process greatly improve training performance — a testament to the elegance of mathematics.

---

LLMS index: [llms.txt](/llms.txt)

---

# Diffusion Models for Generating High-Dimensional Data
## The Generative Modeling Paradigm
Unlike discriminative methods, the generative modeling paradigm is: given a set of IID[^1] samples $x_i \sim p_D(x)$ drawn from an unknown data distribution, learn a model distribution $p_\theta(x)$ with parameter space $\theta \in \Theta$ that approximates the data distribution: $p_\theta(x) \approx p_D(x)$.

Classic generative models in machine learning include:
- Mixture of Gaussian for clustering
- Naive Bayes for classification
- Mixture of Experts(MoE) for unsupervised/supervised learning
- Probability graphical models, e.g. bayesian networks
- Nonparametric Bayesian methods
- Deep generative models

Generative models naturally have the potential to serve as foundation models, because their essence is modeling the joint distribution of multivariate variables: as long as we can effectively estimate $p(x,y)$, we naturally gain the ability to make conditional predictions on $p(x)$ — which amounts to building a classifier. Research has shown that classifiers built this way achieve higher data efficiency in semi-supervised settings where training data is scarce, and some work has also found such classifiers to be more robust against adversarial perturbations.

The rise of deep generative models stems from:
- Compared with discriminative models, their expressive power has grown, enabling them to describe the complex distributions of high-dimensional data.
- Algorithmically, mature variational and Markov chain Monte Carlo (MCMC) methods are available.
- On the data side, self-supervised or unsupervised methods make it easier to exploit large-scale data.
- On the hardware side, new GPU hardware supports the demand for much greater compute.

Differentiable neural-network deep generative models use differentiable DNNs to learn the complex relationships among random variables, with the goal of turning standard Gaussian white noise into real distributions of natural scenes (natural images, audio, video). They achieve very good results even in a fully unsupervised setting. Based on how the probability density function is defined, these models fall into explicit and implicit categories:
- Explicit models such as VAE, Energy-based models, Auto-Regressive, Flow-based Models, and DPM(Diffusion probabilistic models) directly describe the probability distribution of the data to be generated.
- Implicit models such as GAN and Moment-matching DGM instead describe a transformation process, and still require some criterion to guide the model toward producing data that better matches the desired distribution.

From the perspective of training objectives, these models can further be divided into three categories: maximum likelihood estimation (MLE), Score-matching, and adversarial training.
- Score-matching: Moment-matching DGM, Diffusion Models
- Adversarial training: GAN
- MLE: Everything else

## Diffusion Models
Diffusion in physical processes destroys structure over time, going from order to disorder.

In diffusion models, the diffusion process likewise gradually adds Gaussian noise to the data, driving down its signal-to-noise ratio.

Song et al., ICLR 2021[^2] describe a diffusion transition as: $q(x_i|x_{i-1}) = \mathcal{N}(x_i;\sqrt{1- \beta_i}x_{i-1},\beta_iI)$; letting $\alpha_i = \prod_{k=1}^{i} 1 - \beta_i$, we have $q_{\alpha_N}(x_N|x_0) = \prod_{i=1}^{N} q(x_i|x_{i-1}) = \mathcal{N}(x_N;\sqrt{\alpha_N} x_0, (1-\alpha_N)I)$ — the lengthy recursive expression eventually reduces to a concise closed form[^5], so the loss of the N-step forward process can be defined very conveniently. Here $\beta_i$ is a series of noise multipliers, which can be hyperparameters or the result learned via reparameterization. For each training sample $x_0 \sim q_D(x)$, a discrete Markov chain $\{x_0,x_1,...,x_N\}$ can be constructed; after $N$ rounds of noise injection, it eventually approaches Gaussian white noise: $q(x_N|x_0) \sim \mathcal{N}(O,I), N \rightarrow \infty$.

![SDE](https://jipeng4974.github.io/img/sde.png)

The reverse denoising process of the above procedure, $p(x_{i-1}|x_i)$, is unknown and must be learned or estimated — it can be solved via variational approximation, e.g., approximating $p(x_{i-1}|x_i)$ with a Gaussian distribution $\mathcal{N}(\mu_n(x_n), \theta_n^2I)$ whose mean is a function of $x_i$, and minimizing the KL divergence to bring it close.

In principle, Diffusion models are relatively simple:
- There is only noise addition and denoising — no need to learn an encoder and decoder; you only learn to denoise based on how the noise was added.
- The loss function is also fairly simple.
- Convergence is rigorously guaranteed mathematically.

## Large-Scale Training and Efficient Data Generation
In earlier variational-approximation approaches, the variance parameter of the noise was generally fixed and left unoptimized. Analytical-DPMs[^6], proposed by TSAIL at Tsinghua University, found that the mean function and variance at each timestep of the reverse process can be given an analytical form — one that also coincides with some forms hand-designed by other researchers — ultimately yielding a variance estimator that requires no additional training. For a trained DPM, you only need to insert one line of code to take advantage of this analytic-form variance estimate. The estimate makes the variance at each step more accurate, reducing the total number of steps required and translating into a 20–80x performance improvement. This method was later used in Dall-E 2.

Another work from the TSAIL team, DPM-Solver[^7], builds a dedicated solver that brings the number of steps down from several hundred to around a dozen.

Because it involves noise addition and denoising, the underlying architecture of diffusion models naturally borrows from U-Net (CNN). The TSAIL team's third work attempted to combine diffusion models with the transformer, designing U-ViT[^8]; they set up a 500M-parameter large model (the largest at the time), demonstrating that it genuinely helps model scalability. A contemporaneous work, DiT, is very similar. Stable Diffusion 3.0 uses the DiT architecture.

Recall what was said earlier — "generative models naturally have the potential to serve as foundation models, because their essence is modeling the joint distribution of multivariate variables: as long as we can effectively estimate $p(x,y)$, we naturally gain the ability to make conditional predictions on $p(x)$" — based on this heuristic, another research effort from the TSAIL team is UniDiffuser[^9], which aims to use a single model to solve the multiple tasks that previously required several models: marginal diffuser, conditional diffuser, and joint diffuser. At the time, DALL-E 2 and Stable Diffusion could only do text-to-image, while UniDiffuser could do image-to-text as well as text-to-image.

After images, they went on to do Vidu[^10], a text-to-video work that scales up along the time axis, achieving 16s generation. In addition, they worked on 3D content generation — CRM[^11] for image-to-3D and ProlificDreamer[^12] for text-to-3D — scaling up along the spatial dimension. In their latest work, Vidu4D[^13], they perform 4D (i.e., sequential 3D) reconstruction.

## From Generation to Discriminative Classifiers
Generative AI estimates a joint distribution $P(x,y)$; by Bayes' theorem, $p(y|x) = \frac{p(x,y)}{p(x)} = \frac{p(y)p(x|y)}{p(x)}$, and $y^* = \arg \underset{y\in \mathcal{Y}}{\max} p(y|x)$.

If the joint distribution is accurate, this classifier is optimal — the so-called Bayes classifier.

Moreover, the work of Chen et al 2024[^14] shows that a pretrained generative foundation model can be converted into a noise-robust classifier.

[^1]: IID stands for Independent and Identically Distributed
[^2]: Song et al. Score-based generative modeling through stohastic differential equations. ICLR 2021. [[arxiv]](https://arxiv.org/abs/2011.13456)
[^3]: Ho et al. Denoising diffusion probabilistic models(DDPM). NeurlPS 2020. [[arxiv]](https://arxiv.org/abs/2006.11239)
[^4]: In $\mathcal{N}(O,I)$, $I$ denotes the identity matrix, $O$ denotes the zero matrix.
[^5]: Some supplementary good ol' fashioned mathematical rigour: https://math.stackexchange.com/a/4568122
[^6]: Bao et al. Analytic-DPM: an Analytic Estimate of the Optimal Reverse Variance in Diffusion Probabilistic Models. ICLR 2022. [[arxiv]](https://arxiv.org/abs/2201.06503) 
[^7]: Lu et al. DPM-Solver: A Fast ODE Solver for Diffusion Probabilistic Model Sampling in Around 10 Steps. [[arxiv]](https://arxiv.org/abs/2206.00927)
[^8]: Bao et al. All are Worth Words: A ViT Backbone for Diffusion Models. CVPR 2023. [[arxiv]](https://arxiv.org/abs/2209.12152)
[^9]: Bao et al. One Transformer Fits All Distributions in Multi-Modal Diffusion at Scale. [[arxiv]](https://arxiv.org/abs/2303.06555)
[^10]: Bao et al. Vidu: a Highly Consistent, Dynamic and Skilled Text-to-Video Generator with Diffusion Models. [[arxiv]](https://arxiv.org/abs/2405.04233) 
[^11]: Wang et al. CRM: Single Image to 3D Textured Mesh with Convolutional Reconstruction Model. NeurlPS 2023. [[arxiv]](https://arxiv.org/abs/2403.05034)
[^12]: Wang et al. ProlificDreamer: High-Fidelity and Diverse Text-to-3D Generation with Variational Score Distillation. [[arxiv]](https://arxiv.org/abs/2305.16213)
[^13]: Wang et al. Vidu4D: Single Generated Video to High-Fidelity 4D Reconstruction with Dynamic Gaussian Surfels. [[arxiv]](https://arxiv.org/abs/2405.16822)
[^14]: Chen et al. Robust Classification via Single Diffusion Model. ICML 2024. [[arxiv]](https://arxiv.org/abs/2305.15241)
[^15]: Chen et al. Offline Reinforcement Learning via High-Fidelity Generative Behavior Modeling. [[arxiv]](https://arxiv.org/abs/2209.14548)
[^16]: Chen et al. Contrastive Energy Prediction for Exact Energy-Guided Diffusion Sampling in Offline Reinforcement Learning. ICML 2023. [[arxiv]](https://arxiv.org/abs/2304.12824)
[^17]: Chen et al. Efficient Black-box Adversarial Attacks via Bayesian Optimization Guided by a Function Prior. [[arxiv]](https://arxiv.org/abs/2405.19098)
[^18]: Hao et al. DPOT: Auto-Regressive Denoising Operator Transformer for Large-Scale PDE Pre-Training. [[arxiv]](https://arxiv.org/abs/2403.03542)
[^19]: Hu et al. Accelerating Transformer Pre-training with 2:4 Sparsity. [[arxiv]](https://arxiv.org/abs/2404.01847)
