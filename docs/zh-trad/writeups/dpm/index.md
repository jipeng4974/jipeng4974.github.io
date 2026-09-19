# Diffusion Probabilistic Models

> 本文是對朱軍教授的分享——“用於生成高維數據的擴散模型”的筆記。值得注意的是DPM實踐中巧妙使用瞭解析解，無論是前向過程的closed form $q(x_N|x_0)$，還是逆向過程中解析形式的方差估計，都大大提升了訓練性能，體現了數學的精妙。

---

LLMS index: [llms.txt](/llms.txt)

---

# 擴散模型生成高維數據
## 生成式模型範式
不同於判別式方法，生成式建模範式是：給定未知數據分佈的一組IID[^1]數據$x_i \sim p_D(x)$，去學一個參數空間爲$\theta \in \Theta$的模型分佈$p_\theta(x)$，令其逼近數據分佈：$p_\theta(x) \approx p_D(x)$。

機器學習中，生成式模型的經典方法包括：
- Mixture of Gaussian for clustering
- Naive Bayes for classification
- Mixture of Experts(MoE) for unsupervised/supervised learning
- Probability graphical models, e.g. bayesian networks
- Nonparametric Bayesian methods
- Deep generative models

生成式模型天然具備構建基礎模型的潛質，因爲它的本質是對多元變量聯合分佈建模，只要能有效估計$p(x,y)$，自然就具備了對$p(x)$進行條件預測的能力——這也就構建出分類器，而且研究表明這樣構建出來的分類器在半監督這種訓練數據比較少的情況下表現出更高的數據利用率，此外也有些工作發現這種分類器在對抗干擾時表現得更魯棒。

深度生成式模型的興起，源自：
- 相比判別式模型，模型的表達能力變強了，能描述高維數據的複雜分佈。
- 算法上有成熟的變分和馬爾可夫鏈蒙特卡洛方法（MCMC）。
- 數據上更容易通過自監督或無監督方法利用大規模數據。
- 硬件上得到新GPU硬件對更大算力需求的支撐。

可微分神經網絡深度生成模型用可微分DNN去學習隨機變量之間的複雜關係，目標是將標準高斯白噪聲變成一個自然場景下的真實分佈（自然圖片、聲音、視頻）。完全無監督下就能取得非常好的效果。這些模型根據概率密度函數定義可分爲顯式和隱式：
- 顯式模型如VAE、Energy-based models、Auto-Regressive、Flow-based Models、DPM(Diffusion probabilistic models)直接描述預期產出數據的概率分佈。
- 隱式模型如GAN、Moment-matching DGM則描述了一個變換過程，還需要通過一些準則去引導模型產生更符合預期數據分佈的數據。

從訓練目標來看，這些模型又可以分爲最大似然估計（MLE）、Score-matching、對抗訓練（Adversarial training）三類。
- Score-matching：Moment-matching DGM, Diffusion Models
- Adversarial training：GAN
- MLE：Everything else

## 擴散模型
物理過程中的擴散是隨着時間推移，破壞結構，從有序到無序。

擴散模型中擴散過程也是逐漸給數據加高斯噪聲，使其信噪比下降。

Song et al., ICLR 2021[^2]將一個擴散變換描述爲：$q(x_i|x_{i-1}) = \mathcal{N}(x_i;\sqrt{1- \beta_i}x_{i-1},\beta_iI)$，令$\alpha_i = \prod_{k=1}^{i} 1 - \beta_i$，則有$q_{\alpha_N}(x_N|x_0) = \prod_{i=1}^{N} q(x_i|x_{i-1}) = \mathcal{N}(x_N;\sqrt{\alpha_N} x_0, (1-\alpha_N)I)$，冗長的遞歸表達式最終可以劃歸爲一個簡潔的closed form[^5]，因此可以很方便地定義N步前向過程的loss。其中$\beta_i$是一系列噪聲乘數，可以是超參，也可以說reparameterization學習到的結果。對每個訓練數據$x_0 \sim q_D(x)$，都可以構造一個離散馬爾可夫鏈$\{x_0,x_1,...,x_N\}$，經過$N$次加噪，最終使之趨近高斯白噪。$q(x_N|x_0) \sim \mathcal{N}(O,I), N \rightarrow \infty$。

![SDE](https://wujipeng.com/img/sde.png)

上述過程的逆向去噪過程$p(x_{i-1}|x_i)$則是未知、需要學習或估算的——可以用變分近似的方法求解，比如用一個均值爲$x_i$函數的高斯分佈$\mathcal{N}(\mu_n(x_n), \theta_n^2I)$去近似$p(x_{i-1}|x_i)$，用KL散度最小化的方法使之逼近。

原理上來說Diffusion模型相對簡單：
- 只有加噪去噪，不需要去學encoder和decoder，只需要根據加噪學去噪。
- 損失函數也比較簡單。
- 數學上是嚴格保證收斂的。

## 大規模訓練和高效數據生成
此前變分近似的做法中，噪聲的方差參數一般是固定下來，不做優化的。清華大學TSAIL提出的Analytical-DPMs[^6]發現可以給出逆向過程中每個時間點的均值函數和方差的一個解析形式——這個形式和一些學者手工設計的一些形式也比較耦合——最終得到一個不需要任何額外訓練的方差估計器。訓練好的DPM，只需要插入一行代碼，就能用上這個解析形式的方差估計。用這個估計值使每一步的方差估計變得更準，使所需的整體步數變少，折算下來有20~80倍的性能提升。後來這個方法也在Dall-E 2中使用了。

TSAIL團隊的另一個工作DPM-Solver[^7]做了專門的求解器，使步數從幾百步降低到十餘步。

由於涉及加噪去噪，擴散模型的底層架構自然而然借鑑U-Net（CNN）。TSAIL團隊的第三個工作，嘗試把擴散模型和transformer結合，設計了U-ViT[^8]，在當時設置了5億參數的（當時算最大的）大模型，證明對模型的可擴展性確實有幫助。同期有個工作DiT非常類似。Stable Diffusion 3.0就用的是DiT架構。

回顧前文所述的“生成式模型天然具備構建基礎模型的潛質，因爲它的本質是對多元變量聯合分佈建模，只要能有效估計$p(x,y)$，自然就具備了對$p(x)$進行條件預測的能力”，基於這種heuristics，TSAIL團隊的另一個研究是UniDiffuser[^9]，目標是用一個模型解決原本marginal diffuser、conditional diffuser、joint diffuser這多個模型才能解決的多個任務。當時DALL-E 2和Stable Diffusion只能文到圖，而UniDiffuser能圖到文或文到圖。

做完圖像之後，又做了Vidu[^10]文生視頻的工作，在時間軸上做了升維，實現了16s的生成。此外，還做了3D內容生成，CRM[^11]圖生3D，ProlificDreamer[^12]文生3D，在空間上做了升維。在最新的工作Vidu4D[^13]中，做了4D（即sequential 3D）重建。

## 從生成到判別式分類器
生成式AI估計一個聯合分佈$P(x,y)$，基於貝葉斯定理可得$p(y|x) = \frac{p(x,y)}{p(x)} = \frac{p(y)p(x|y)}{p(x)}$，$y^* = \arg \underset{y\in \mathcal{Y}}{\max} p(y|x)$。

如果聯合分佈是準確的，那麼這個分類器就是最優的，即所謂貝葉斯分類器。

此外，Chen et al 2024[^14]的工作表明可以將一個預訓練好的生成式基座模型轉化成一個對噪聲魯棒的分類器。

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
