+++
title = "Music JEPA 和三体宇宙"
date = "2026-08-05"
tags = ["ai"]
description = "生存维数，维度坍塌，死维复活，宇宙规律，田园宇宙，幸存子空间，威慑项λ。"
showFullContent = false
+++

前文[Music JEPA Regularizers](https://jipeng4974.github.io/posts/music_jepa_reg/)兼具数学之严格、工程之呆板和AI之平庸，写完总觉得缺了什么没表达出来。本文尝试用三体宇宙进行尽可能有效的类比，尝试将Music JEPA思想实验过程中，我的那些穿越星云的想象，以及从惊诧到欢欣的心路历程分享出来。

## 多重概念映射表

| Music JEPA SSL | 三体宇宙 |
|---|---|
| embedding space | 宇宙 |
| 音乐表征模型 | 整体意义上的文明 |
| reward hacking | 在残酷宇宙中求生并被迫作弊的文明生存策略 |
| effective ranks | 文明实际展开生存的维数 |
| invariance loss（对齐规则） | 宇宙规律 |
| 维度坍缩（erank下降） | 二向箔/文明向低维迁移 |
| 虚假高斯繁荣的幸存子空间/低秩盲区 | 黑域/低维小宇宙 |
| collapsed ranks（$\sigma\to 0$ ）|  死维度/热寂维度 |  
| $N(0,I_{512})$ isotropic Gaussian理想分布 | 早期高维宇宙的田园时代 |
| 复活死维度的正则项 | 放弃小宇宙，归还质量的归零运动 |
| 对比学习项拉大异类距离 | 个体文明之间的黑暗森林法则 |
| $\lambda_{\text{reg}}$ | 归零主义的影响力 |
| $\lambda_{\text{infonce}}$  | 执剑人的威慑度 |

## Music JEPA的训练目标和实验现象
Music JEPA训练目标包括：
- 目标1: 让模型学会anchor music和degraded music之间的invariance，即服从宇宙规律。
- 目标2: 让模型embedding space的分布趋近isotropic Gaussian。
- 此外，实践中还往往可引入目标3：额外设置一个对比学习项，让模型学会将一些明显的negatives和anchor拉开距离——虽然这与JEPA无关，但对比学习简单有力，能在目标1和目标2之外，额外提供一些不会错得离谱的训练动力。

前文[Music JEPA Regularizers](https://jipeng4974.github.io/posts/music_jepa_reg/)中提到了两个有趣的实验现象：
- SIGReg、VISReg能防坍塌，但均难以复活死维，VISReg稍强，但常规λ调度下，提升erank方面的训练动力仍明显不足。
- 如果给VISReg一个超超高的λ——全力以赴奔向isotropic Gaussian，会发现它不仅不复活死维，反而会导致维度坍塌，VISReg loss下降的同时，erank缓慢阴跌。也就是说训练陷入了局部低秩最优，或者说“低秩盲区”。

补充一些新观察到的现象：
- 但常规λ调度下，VISReg和SIGReg的eranks会在特定数值K收敛，彻底失去增长动力，这个K实际上接近白化后表征流形的内禀维度上限。
    - VISReg和SIGReg都确切具有死维复活能力，虽然速率和稳定性略有不同。
    - 除了invariance对齐规则外，参与训练的内容丰富度也影响表征流形的内禀维度上限。
- 超强λ约束和激进invariance目标约束下，模型有可能在训练的部分时间（前中段）偶然爆发出invariance loss降低和erank提升的理想叠加态，但这种爆发不稳定，训练步数增加后会屡次发生坍塌。

## 宇宙规律是最残酷的武器，律法不可过苛
星际文明间的黑暗森林法则有些黑暗但并不残酷，残酷的是不可阻挡的宇宙自由度流失（热寂、降维、光速降低）。

类似地，在Music JEPA自监督表征学习中，真正残酷的是目标1 invariance learning。过分激进的invariance loss会导致模型进行痛苦的低维逃逸，通过对表征流形进行信息删除，削足适履地满足invariance对齐，这意味着更严重的内禀维度坍塌，远比contrastive loss难以避免的false negative样本对geometry的局部扭曲更严重。

所以**不要设定太夸张的invariance目标**，此前我在一次实验中强制要求15s anchor裁切出来的1.5s～15s子集在强变速变调加噪失真下仍然能和anchor对齐，于是出现了壮观的低维逃逸，表征流形内禀维度的损失靠超高$λ_{reg}$也救不回来——eranks上升曲线放缓并不是正则项设计缺陷，后续的白化实验证明了实则已经触及这个invariance规则下对应的维度上限了。

秦律，失期法皆斩，过于残酷的律法不但没有带来更坚固的秩序，反而会招至沸反盈天。

因此JEPA语境下Curriculum Learning的真正意义，不在于让learner从简单到难，逐步攀登，学会更神奇的对齐任务[^1]。而在于当发现learner在双重约束下，不得不痛苦自残、向低维逃逸时，适时地放弃后续的课程，从而规避灾难性的坍塌。

## 追逐Isotropic Gaussian和“田园时代”，而避免陷入低秩盲区
早期高维三体宇宙拥有充沛而均匀的自由度，没有黑域，没有维度战争，没有过分拥挤后的黑暗森林战争，没有偏安一隅的小宇宙，文明有充裕的生存空间，因此被称作“田园时代”。三体宇宙中的归零主义的目的是归还维度，修复大宇宙，回归“田园时代”。

Music JEPA的目标2即回归isotropic Gaussian的“田园时代”。为何isotropic Gaussian是理想分布？因为数学上它已被证明是是最适合所有下游任务的理想分布。Embedding space接近isotropic Gaussian方可自称是通用foundation model。以检索任务为例，哪怕不做任何negative mining，不做任何对比学习，isotropic Gaussian也能天然保证negative cosine p99999足够低——表征自然而然就胜任大规模向量库的检索。

前文中介绍的3种正则项中，SIGReg和VISReg都直接以isotropic Gaussian为目标，VICReg有更强的复活死维，撑erank下限的能力，只是不保证形状。因此将VICReg和VISReg结合起来使用，或在VISReg中引入VICReg的协方差项（完全不影响isotropic Gaussian目标），有助于避免JEPA学习过程中陷入“低秩盲区”[^2]。

所谓“低秩局部最优/低秩盲区”是一种幸存子空间的虚假高斯繁荣的幻像。在激进invariance 目标和强力$λ_{reg}$的双重约束下，SIGReg和VISReg均有几率陷入这种局部最优——看起来正则项loss的确在降，但erank增长乏力，甚至缓慢阴跌。看着监控里embedding space上的维度一个一个死掉，活着的维度却更拥挤更热闹，昂然有序地朝着伊甸园形态逼近——这一幕让人不经想起三体宇宙中主动将自身降维，向低维度逃逸的文明：低维度小宇宙中秩序在新生，文明熙熙攘攘，低维生物视角下俨然是毫无破绽的伊甸园，但高维大宇宙遗民视角下，一切旧世代文明痕迹都在灰飞烟灭。


## 引入对比学习项——三重张力的微妙平衡
Music JEPA 实践中的目标3不够JEPA，但行之有效，负面效果可控，因此是可选项。

类比到三体宇宙，引入对比学习项相当于在已经面临必然沦亡宿命，渴望回归“田园时代”的诸文明之间，引入额外的黑暗森林法则。这第三种张力或许在宇宙消亡的大图景下无足轻重，但在特定时间节点（无法投入更多GPU hours）对特定文明个体（特定下游任务）来说却也能发挥决定性作用。

代价是引入新的超参$λ_{infonce}$，需谨慎控制，达到某种微妙平衡。
- $λ_{infonce}$太低或为0，则在isotropic Gaussian 的田园时代真正降临之前，永远会有异常高的negative cos。恰如程心作为执剑人。
- $λ_{infonce}$足够高，相当于威慑等级足够高，从而保住异类距离的下限。恰如罗辑作为执剑人。
- $λ_{infonce}$太高则喧宾夺主，削弱JEPA的主目标。在微小扰动下选择同归于尽，也会断绝文明健康演进的生路。

三重张力作用下，什么才是理想结局？
- 从 Music JEPA 的角度说，理想结局是模型学会了难度适中的 degraded crop invariance 对齐，同时让 embedding 分布大致接近 isotropic Gaussian，证得“基础模型”。
- 从三体宇宙文明的角度说，理想结局是文明适应光速不变，熵增热寂，黑暗森林等宇宙规则和推论，同时齐心协力执行归零行动，复活大宇宙死维度，回归“田园时代”。

[^1]: 在苛刻的invariance目标下，ViT的确能学到无中生有的超短时长和长程旋律的对齐，甚至同一首歌的无关片段的对齐。
[^2]: 当然可能存在更数学、更严谨、更完备的正则方法克服这种低秩盲区，只是我当下缺乏足够的数学功底和时间去探索。