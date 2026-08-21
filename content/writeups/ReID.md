+++
title = "Semantic Depth & Music Re-ID"
date = "2026-03-14"
tags = ["Systems", "AI"]
description = "This post proposes a semantic depth metric and discusses Music Re-ID from the angles of inductive bias, data-driven methods, performance engineering, and cascaded ranking."
showFullContent = false
+++

From purely prior-based handcrafted feature retrieval, to purely data-driven deep metric learning, and then to providing the model with prior domain knowledge to assist learning — the process spirals upward.

Even pretrained MLLMs, which represent the extreme of data-driven approaches, universally feed log mel spectrograms rather than raw waveforms into their audio encoders — this in itself is an inductive bias grounded in human hearing and psychoacoustics.

Music is naturally highly structured, and annotated data in the music retrieval domain is naturally scarce and non-public. Therefore, introducing inductive bias based on prior knowledge is logical — it spares the model from having to learn acoustics, signal processing, and music theory from scratch.


# Inductive Bias

## Choosing the Right Semantic Depth
Semantic depth is a metric I coined myself.

There are several levels of abstraction between a signal sampled from the real world and human subjective perception. Each time you dive one level of abstraction down from the signal level, semantic depth increases by 1.

In the field of music representation, we might define semantic depth as follows:

- Semantic depth 0: general audio signal features with no musical semantics, e.g., peaks, spectral flux, shazam/quad fingerprints, MFCC.

- Semantic depth 1: micro-structural music features that carry musical semantics, e.g., harmonic peaks/ridges, attack onsets, beat positions, spectral envelope, pitch contour.

- Semantic depth 2: meso-structural music features, typically covering memorable motifs of 5–30s, e.g., melody, rhythm, groove.

- Semantic depth 3: macro-structural music features, typically covering the entire piece, capable of mining deeper narrative, emotional, structural, and stylistic information, as well as relational and context-specific information.


## Choosing the Right Input Form
If the queries and docs in the target scenario are almost all music, the input layer can naturally adopt the musically motivated CQT spectrogram, and the audio slice granularity can be chosen according to the semantic depth required by the scenario.

## Inductive Bias for Capturing Invariants
Certain handcrafted features can also be provided to the model as auxiliary information. Most of these handcrafted features are compressed representations of the spectrum, focusing on musical invariants at a specific semantic depth of 1, such as pitch contour and the topological relationships among harmonic peaks.

# Data-Driven
## Hard Sample Mining
Set a score threshold with a sufficiently low FPR, take the top k from KNN retrieval: those scoring above the threshold are positives (used to expand the sim group), and the rest are negatives.

From the negative candidates in the top k, remove the hard negatives (scores higher than the sim group's average similarity but below the threshold — this part can only rely on manual annotation), and keep the semi-hard negatives whose scores are below the sim group's average similarity.

## Generated Data
Use music or video generation models to continue or adapt samples.

## Data Augmentation
Speed and pitch shifting, perturbation and noise injection, remixing.

# Performance Engineering

## Encoder Lightweighting
When the query is short enough — for example, audio clips of only 10s — it can fit within the receptive field of a CNN[^1], and if that CNN architecture has undergone proper attention-hybrid modifications[^2], the transformer encoder's advantage in long-range dependency modeling disappears.

## MRL
MRL (Matryoshka Representation Learning) allows the sub-vectors formed by the first few dimensions of a representation vector to be used directly for retrieval. This is clearly superior to standalone PCA dimensionality reduction or adding a linear layer during training for dimensionality reduction — more flexible and worth trying.

MRL is mainly used in ultra-large-scale retrieval to reduce the storage and retrieval cost of the vector index.

MRL also reduces the number of actually activated parameters in the classification head during inference, which can slightly lower retrieval cost in scenarios with a particularly large number of classes.

## QAT
Compared with MRL, QAT certainly also reduces vector size, but its main purpose is to exploit the performance advantage of hardware low-precision inference, substantially cutting inference cost. The gains and losses of QAT and MRL are orthogonal to each other.

# Cascaded Ranking
In most scenarios, metric learning + large-scale ANN vector retrieval is sufficient to solve audio retrieval problems.

In a few difficult scenarios, cascaded retrieval is needed: use an embedding model for coarse recall, and some reranker for fine ranking.
- The reranker can be a cross-encoder reranker built on the original embedding model.
- Or a pretrained MLLM post-trained on the rerank task.

[^1]: CNN here is just a broad category; in practice we would use the ResNeSt variant of ResNet. ResNeSt adds Split-Attention Conv on top of ResNet.

[^2]: On top of ResNeSt-50, we can further insert NonLocal self-attention modules into layer2 and layer3 to improve contextual understanding. In addition, the lower layers replace BN with IBN — half the channels go through InstanceNorm, the other half through SyncBN. Introducing InstanceNorm removes style and suppresses the model's learning of the energy envelope.
