# Music LeJEPA

> 初试Music LeJEPA（未完待续）

---

LLMS index: [llms.txt](/llms.txt)

---

# 表征学习的发展脉络
## 我们是表征派在音乐领域的践行者
ByteCover3的成功建立在谢赛宁和何凯明的ResNeST的成功之上，也用到了Yann Lecun的对比学习。

Music LeJEPA则是试图将LeCun在视觉领域的新方法复刻到音乐表征中。虽然没有世界模型那么玄乎（预测/对齐的不是世界表征，而仅仅是旋律表征），但确实是一个根本上的范式转变。

我们的音乐识别模型是正统的表征派路线。Yann Lecun、谢赛宁是表征派的代表人物，在JEPA框架下取得了一些进展，目前正在尝试从表征学习入手建立Real World AI。

## 表征派的元信念
表征派的核心哲学观点认为：AI的核心不是 multimodal，也不是 generation，而是 representation。生成是理解的副产物。

不能因为生成式AI在商业上的巨大成功，就误以为“生成即理解”，或“生成式模型可以顺带解决理解问题”。

事实上在VAE之前，autoencoders are not good representation learners。而VAE，又和LeJEPA在先验上异曲同工。
- Vanilla autoencoders，reconstruction loss = 输入输出的L2 loss。
- Sparse autoencoders，在reconstruction loss之外额外引入sparsity loss——限制只有少部分neuron被激活，这样输出的embeddings会less entangled，线性可分性更强。目前主要用于可解释性研究上。
- Denoising autoencoders，以原始信号的degraded views为输入，尝试重建原始输入，是后世生成式中score-based diffusion的来源。
- Masked autoencoders，即著名的MAE，kaiming尝试在ViT上复刻BERT的masked autoencoding，当时被认为很成功，若干年后回顾，实则是失败（投入巨量资源，以像素重建为目标，训出来的表征不具备线性可分性，夹杂太多无关特征，至少在检索领域是不可用的）。
- Variational autoencoders，真正的革命性范式转变，在reconstruction loss之外额外引入了让latent更接近标准高斯的KL项。这和LeJEPA引入SIGReg正则项有异曲同工之妙！

生成式表征的问题在于训练目标和“语义层面的理解”背离，或多或少揉入了更表层的理解，比如像素级重建必然会使模型学到局部纹理。即便是VAE也无法和专注于表征的模型在理解任务指标上媲美。

VAE和LeJEPA的成功，都源于这样的inductive bias或者说设计哲学：不能只优化任务目标，必须同时对隐空间的"几何结构"施加统计约束。

## LeJEPA之前SSL的发展脉络

```
AE演化出的一条线是生成模型。
                Autoencoder (AE)
                      │
      ┌───────────────┼────────────────┐
      │               │                │
      ▼               ▼                ▼
Denoising AE     Sparse AE         Contractive AE
      │               │
      │               ▼
      │        LLM Interpretability
      │
      ▼
Variational AE (VAE)
      │
      ▼
Latent Diffusion
      │
      ▼
Stable Diffusion

另一条线则是自监督表征学习。

Autoencoder
      │
      ▼
Masked Autoencoder (MAE)
      │
      ▼
Self-Supervised Vision Models
      │
      ▼
JEPA / LeJEPA
```

SSL不止Masked生成式自监督这一条线，最自然而然的一个路线是contrastive SSL，如SimCLR、MoCo，依赖构造正负样本对和hard mining，是有监督对比学习在自监督领域的直接迁移，自然免不了supervised contrastive learning的一系列痛点——contrastive loss训练不稳定，很难构造可靠的hard mining链路，需要巨大的batch size。

继contrastive SSL、生成式SSL之后，又有一个SSL路线，曰逆向自蒸馏，由Dino系列发扬光大，音乐领域的MERT也用了类似的技术。目前回头看自蒸馏路线只是一个错误尝试的工程补救而已，没有可解释性，也没有任何借鉴价值——但不代表Dinov3整体没有借鉴价值，Dinov3的loss有四部分组成，自蒸馏只是其中一部分，其Kelo loss也是一种几何正则，和LeJEPA的SigReg正则异曲同工。

## LeJEPA：让SSL从炼丹变成科学
和前世代SSL相比，LeJEPA是极简的：

$$L_{LeJEPA} = (1-λ) L_{invariance} + λ L_{sigreg}$$

LeJEPA把JEPA目标简化成invariance loss(所谓的latent prediction) + 几何分布正则 SIGReg。

所谓invariance loss，放在音频检索领域，其实就是global views和local views对应的projected embeddings的L2 loss。之前也有过类似的做法，比如SimCLR contrastive SSL中的正样本对之间的距离最小化，但此前的方法未能彻底解决表征坍塌问题，也没能找到数学上最优的几何分布约束。

# 一些工程实践
## 初始权重
实践表明主要基于自然图像训练的SOTA预训练视觉模型Dinov3并不适合作为初始权重进行SSL，相比从初始化权重开始训练，invariance loss收敛稍快一点，但sigreg loss收敛缓慢，初始effective ranks极低，在长时间训练后effective ranks能提升，但有上界，不能持续提升。
- 自然图像上预训练太久的模型，容易把所有灰度频谱图都视为极为接近的。Dinov3 512 output dims上甚至吝啬到只给4个有效维度给cqt/logmel频谱图像。40000 steps后effective ranks增长到40，增长曲线放缓。
- 从随机化初始化权重开始预训练。30000 steps后effective ranks增长至100。

## 难度渐进的Curriculum Learning
训练Degradation-robust Melody Matching模型，需构造各种不破坏旋律的transform，如裁剪、变速变调、施加各种振幅包络变化、基于足够丰富的noise bank加噪等。其中大部分invariance学起来比较简单，但也有一些非常困难需模型有一定基础后再尝试学习，如比较极端的加噪，极端的变速变调。

此外，还需实现一个sigreg λ scheduler，随着训练步数提升而增加各向同性正则权重。

## Contrastive后训练
尽管non-contrastive SIGReg正则可以在任何具体下游任务上都有好的表现，effective ranks的爬升是缓慢的，预训练投入是相当大的。若要迅速见效，可针对具体下游任务，借助人工标注数据，进行contrastive后训练。

传统的Triplet loss在某些场景非常有效，但有更多缺陷：
- 绝大多数triplets没有梯度。
- 有梯度的那些triplets往往又多出false negative。
- 投入大量人力做人工标注在很多场景又几乎不可能。
- 真正hard的negatives梯度太大，又会导致训练震荡。以至于不得不退而求其次找semi hard negatives。
- Offline mining会导致随着训练steps导致的embedding分布变化而越发过期。
- Online mining对超大batch size有需求。
- Mining规则越堆越复杂，heuristics越来越多，以至于越发违反the bitter lesson。

相比之下，大多数contrastive SSL选择整个batch参与计算的InfoNCE，从训练链路中剔除了复杂、脆弱的hard mining。

Lecun团队去年提出的X-sample contrastive loss重定义了对比学习的对象——基完整的sample similarity graph，而非pairs进行学习。InfoNCE也可以算作一种粗糙的similarity graph——相似度矩阵中只有positives之间是1，其他都是0——把所有negatives都粗暴归零，是一种信息浪费。

苏剑林6月博文[强制间隔投影](https://kexue.fm/archives/11784)中提出一种巧妙的margin loss实现，很适合拉大anchor和negatives之间的margin，比较适配检索任务，也值得尝试。


## 可规模化的退化视图合成
离线做昂贵的基于waveform的退化合成，如变速、变调、混响、NoiseBank加噪。每一项退化设置若干变化幅度，在线生成view时从中任取。
- 维护一个有足够多样性的noise bank（人声歌唱、环境噪声、白噪声、各种ugc音频、音乐、tv）。在waveform进行加噪更合理，更仿真一些。

在线做廉价的基于cqt/logmel tensor的退化合成，如随机振幅包络（时间方向缓慢变化），局部能量扰动，随机噪声底（在谱图上添加低幅度噪声），时频masking，动态范围压缩，频域EQ（随机频率响应曲线，对每个频带乘一个缓慢变化的增益），spectral tilt（整体变亮/变暗，高低频能量倾斜）。

Waveform -> CQT这一步颇昂贵（换LogMel后CPU开销也仍然可观），对所有waveform合成都离线化处理，可保证训练链路中，不会在数据pipeline上引入CPU瓶颈。
- 另一种可行的方案是部署一个大规模CQT提取集群（但我不愿意在通用训练框架中引入对网络、外部系统的依赖）。
- 还有一种思路是直接用廉价的log mel频谱代替正统的cqt表示，这很可能是可行的，后续可以做个ablation study。
