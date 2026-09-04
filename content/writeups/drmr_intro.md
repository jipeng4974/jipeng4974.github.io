
+++
title = "Degradation-Robust Music Representation Learning with LeJEPA"
date = "2026-09-04"
tags = ["AI"]
description = "Learning invariance from synthetic degraded music"
showFullContent = false
+++

## 1. Introduction

Music identification systems have traditionally been evaluated under relatively controlled forms of signal degradation. In real-world user-generated content (UGC), however, increasingly challenging queries may preserve the perceptual identity of a musical work while differing substantially from its original recording. Since 2026, we have observed a growing number of such queries on a large-scale short-form video platform. Two particularly prominent sources are AI-generated covers and copyright-infringing musical transformations, including heavily modified, remixed, or otherwise transformed versions of copyrighted music. These examples substantially alter the acoustic realization of a song while often preserving enough musical information for human listeners to recognize its identity.

This emerging regime exposes a limitation of existing music identification systems and benchmarks. Supervised retrieval models such as ByteCover3 can achieve strong performance on established benchmarks, yet their robustness can degrade substantially when confronted with these previously unseen transformations. A fundamental challenge is the scarcity of suitable training data. Identifying genuinely difficult but identity-preserving queries from large-scale UGC requires human inspection and annotation, making it impractical to construct a sufficiently large training corpus. In our setting, only several hundred such examples can be manually identified and labeled each month. This volume is insufficient for large-scale supervised training, but sufficient to establish a small, carefully curated benchmark for evaluating robustness to modern music degradations.

These observations motivate a different approach to representation learning. Rather than relying on manually labeled degraded examples, we ask whether a music representation can learn to preserve musical identity across a broad family of synthetic degradations. Given an original recording \(x\) and a degradation transformation \(D\), we seek a representation \(f\) satisfying

$$
f(x) \approx f(D(x)),
$$

while remaining discriminative across different musical works. The key idea is to exploit the large amount of clean music available for self-supervised training and generate diverse degraded views synthetically, thereby decoupling the scale of training data from the availability of human annotations.

We propose a degradation-robust music representation learning framework based on a LeJEPA-style joint-embedding objective. Starting from 120K source music tracks, we generate 40 degraded variants per track, yielding approximately 4.8 million degraded examples and roughly 290K hours of offline training data. A ViT-L backbone is trained to learn representations that are invariant to these degradations, using an invariance objective together with a regularization term that encourages an approximately isotropic Gaussian embedding distribution. The resulting representation is evaluated directly in large-scale music retrieval.

Can synthetic degradation diversity learned from clean recordings generalize to real-world degradations that are difficult to collect and may never have been observed during training?

On a newly constructed challenging benchmark containing severe but human-identifiable degradations, our method substantially outperforms existing self-supervised music representation models and supervised retrieval systems. These results suggest that degradation-specific self-supervision can provide a scalable alternative to manually labeling difficult real-world queries, and that current music identification benchmarks may underestimate the robustness required by modern UGC applications.

> T.B.C.