+++
title = "Semantic Depth & Music Re-ID"
date = "2026-03-14"
tags = ["sys", "ai"]
description = "本文提出语义深度指标，从归纳偏置、数据驱动、性能工程、级联排序等角度讨论Music Re-ID。"
showFullContent = false
+++

从基于纯先验的手工特征检索，到纯数据驱动的深度度量学习，再到给模型提供一些先验领域知识辅助学习，是螺旋上升的过程。

即便是代表极致数据驱动的预训练MLLM，也普遍将log mel频谱而非原始波形送入模型的audio encoder，这本身就是一种基于人耳听觉和心理声学的归纳偏置。

音乐是天然高度结构化的，音乐检索领域的标注数据天然是稀缺且不公开的，因此基于先验知识引入归纳偏置是合乎逻辑的，可以让模型不必从声学、信号处理、音乐理论学起。


# 归纳偏置

## 选取合适的语义深度
语义深度(semantic depth)是我生造的指标。

从现实世界采样信号到人的主观认知之间存在若干个抽象层次，自信号层面每下潜一个抽象层次，则语义深度+1。

音乐表征领域中，不妨将语义深度定义如下：

- 语义深度为0的通用音频信号特征，无音乐语义，如peaks、spectra flux、shazam/quad fingerprints、mfcc。

- 语义深度为1的音乐微观结构特征，携带音乐语义，如harmonic peaks/ridges、attack onsets、beat positions、spectral envelope、pitch contour。

- 语义深度为2的音乐中观结构特征，往往覆盖5~30s有记忆点的motif，如melody、rhythm、groove。

- 语义深度为3的音乐宏观结构特征，往往覆盖全曲，能挖掘更深层的叙事、情感、结构、风格、关联信息、特定语境下的信息。


## 选取恰当的输入形态
如果确定应用场景中query和doc几乎都是音乐，输入层面自然可以选用合乎乐理的cqt频谱，并根据该场景所需的语义深度选取对应的音频切片粒度。

## 捕捉不变量的归纳偏置
其次可以将某些手工特征作为辅助信息提供给模型。这些手工特征大多属于频谱的某种压缩表示，关注特定语义深度为1的音乐不变量，例如pitch contour，harmonic peaks的拓扑关系。

# 数据驱动
## 硬样本挖掘
确定一个fpr足够低的得分阈值，基于KNN检索取top k，得分超阈值则为positive(用于扩充sim group)，其他为negative。

从top k中negative cands中剔除掉hard negative（得分大于sim group平均相似度而小于阈值，这部分只能依赖人工标注），留下得分低于sim group平均相似度的semi-hard negatives。

## 生成数据
音乐或视频生成模型续写、改编样本。

## 数据增强
变速变调、扰动加噪、混音。

# 性能工程

## Encoder轻量化
当query足够短，比如普遍是仅10s的音频clip时，可fit in cnn[^1]感受野，且该cnn架构进行过恰当的attention hybrid改造[^2]，transformer encoder的长程相关性建模优势就不存在了。

## MRL
MRL（套娃表征学习）允许表征向量的前若干维构成的子向量是直接可用于检索的，这显然优于独立的PCA降维或训练时加个线性层降维，更加灵活，值得尝试。

MRL主要用于超大规模检索中降低向量索引的存储和检索开销。

MRL在推理阶段也降低了分类头实际激活参数量，在class数量特别多的场景可稍稍降低检索成本。

## QAT
和MRL相比，QAT固然也减小向量尺寸，但主要是为了发挥硬件低精度推理的性能优势，大幅降低推理成本。QAT和MRL的收益和损失均正交。

# 级联排序
多数场景下，metrics learning + 大规模ANN向量检索足以解决音频检索问题。

少数困难场景，需要引入级联检索，用embedding模型做粗召，用某种reranker做精排。
- reranker可以是一个基于原有embedding模型的cross-encoder reranker。
- 或基于预训练MLLM在rerank任务上做后训练。

[^1]: 这里的CNN只是大类，实际会使用Resnet中的ResNeSt变种。ResNeSt比Resnet多了Split-Attention Conv。

[^2]: 在ResNeSt-50基础上，还可以在layer2, layer3插入NonLocal自注意力模块，进一步提升上下文理解能力。此外，低层还会用IBN替代BN，即一半通道走InstanceNorm，另一半SyncBN。引入InstanceNorm可去风格，抑制模型对能量包络的学习。