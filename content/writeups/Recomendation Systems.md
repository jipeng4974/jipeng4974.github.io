+++
title = "Recommender Systems"
date = "2025-01-22"
tags = ["AI", "Systems"]
aiAssisted = true
description = "Reading papers with the help of DeepSeek R1 to sort out recommender systems."
showFullContent = false
+++

## Overview of Search & Recommendation Architecture
Traditional search-and-recommendation systems use a three-stage funnel (matching/recall, pre-ranking/coarse ranking, ranking/fine ranking).

The recall stage balances relevance with some ranking capability, and usually employs multi-channel recall.
- In the traditional three-stage structure, the separation of recall and ranking yields low commercial returns. So during recall, one can use a relevance model for training-sample augmentation[^1], or model relevance bad cases as negative feedback[^2].
- Sample space design must account for the SSB problem[^7]:
    - Use in-batch negative sampling[^8] and logQ popularity debiasing[^9].
    - Construct medium-hard negative samples[^10].
    - See also the MUDA (Modified Unsupervised Domain Adaption) method from Pinterest ads retrieval[^11].
- Model architectures: naive two-tower, MVKE[^12], QARM[^13].

Given guaranteed relevance, the pre-ranking stage mainly focuses on ranking capability to filter out high-quality results.
- Pre-ranking likewise has to deal with SSB.[^15]
- Besides traditional pre-ranking models, methods aligned with fine ranking are also worth considering: COPR[^16].

The goal of the ranking stage is to select the best content from the candidate set through fine-grained modeling and multi-dimensional optimization, accurately predict user behavior, and maximize some business objective.
- Ranking: fine-grained scoring of hundred-scale candidates (DeepFM, DIN, MMOE), focusing on accurate prediction (CTR/CVR) and feature interaction.
- Re-ranking: tuning the top results (PRM, DPP, MMR), focusing on list-level optimization (diversity, context).

## Mobius
The Mobius project aims to improve performance by integrating business metrics such as CPM[^3], CTR[^4], CPA[^5], and ROI[^6] into the matching layer, while respecting low-latency and compute-resource constraints.

Mobius has two key techniques: an actively learned CTR model and fast ad retrieval. The former uses a teacher-student framework, leveraging the existing relevance model to generate synthetic data and improving the CTR model's generalization to long-tail queries and ads. The latter is the retrieval technology those of us working on content retrieval know very well: ANNS/MIPS and OPQ vector compression.

In Mobius, the CTR model is integrated directly into the recall stage (in traditional search-and-recommendation, CTR usually sits in the ranking stage, disconnected from recall) as a two-tower DNN (query tower + ad tower), so relevance and commercial value are considered together at recall time, enabling efficient filtering over billions of ad candidates.

ANNS needs no elaboration. OPQ (Optimized Product Quantization) is a novel low-loss compression mechanism for high-dimensional vectors: a slight upgrade over PQ that applies an orthogonal matrix R to the original vectors so that the transformed vectors are distributed more uniformly across subspaces, reducing the correlation between sub-vectors and thereby reducing PQ quantization error (PQ splits a vector into multiple sub-vectors, which may be correlated with each other).

## "Not-to-Recommend" Loss
In [^2], Google proposes a "Not-to-Recommend" loss trained on user negative feedback, which explicitly uses negative-feedback signals to optimize the log-likelihood of not recommending the item. This loss is combined with the standard cross-entropy loss over positive-feedback interactions, forming a joint learning mechanism that accounts for both positive and negative feedback.

Most prior work leveraging negative feedback relied on ad-hoc feature engineering.

## SimANS 
SimANS is a negative-sampling method proposed for dense text retrieval, aiming to solve the problems of uninformative negatives (too easy) and false negatives (too hard) in existing methods. Its core idea is to select ambiguous negatives — negatives whose relevance scores are close to those of the positives — balancing sample difficulty and informativeness to improve training.

Its innovation lies in being the first to explicitly target medium-hard negatives (ambiguous negatives) as the core sampling objective, providing a theoretical basis for their gradient properties (large mean, small variance), and experimentally validating their importance for model training.

In addition, SimANS introduces an exponential-function-based probability distribution that dynamically adjusts sampling weights according to score differences, avoiding the drawbacks of fixed strategies (e.g., random or Top-k) while remaining computationally efficient.

## MUDA
Traditional recommender systems suffer from selection bias in their training data: the training distribution often differs from the actual inference-time distribution. In ads retrieval, training data mainly comes from later stages (auction winners) — the result of multiple rounds of filtering — which differs substantially from the distribution of the real, full candidate ad set.

Traditional user-behavior-based modeling ignores the information carried by the vast number of unselected ads. MUDA[^11] converts ranking results (i.e., pseudo-labels) into binary labels (positive/negative examples) and replaces the regression loss with binary cross-entropy, avoiding overfitting to low-confidence pseudo-labels, thereby indirectly leveraging — during training — unlabeled data whose distribution is closer to the real inference-time distribution.

## MVKE
Previous user-profile modeling generally used multiple independent two-tower models to predict CTR and CVR[^14], but this suffers from data sparsity (single-action models — e.g., click-only or conversion-only — cannot learn complementarily, so modeling of data-sparse actions performs poorly) and insufficient feature interaction (the two towers are separated, so user-tag interaction is inadequate, making it hard to capture multi-topic preferences).

The MVKE model sets up Virtual-Kernel Experts (VKEs) and virtual kernel gates (the VKG gating structure) to jointly learn user preferences across different actions (click, payment) and topics (sports, automobiles), improving the diversity and accuracy of user profiles. In MVKE, a user tower contains multiple VKEs, each responsible for one specific domain of user preference and guiding feature interaction via so-called virtual kernels (a kind of learnable parameter). The VKG dynamically fuses the outputs of different VKEs based on tag embeddings and the virtual kernels.

In the "click -> conversion" sequence, clicking is a shallow task that can provide foundational information for the deeper conversion task.

## QARM
QARM is a recommendation architecture based on MLLMs (multimodal large language models). The paper points out two problems with MLLMs in industrial recommender systems: representation mismatch and unlearnable representations. The former means the objectives of pretrained multimodal models are inconsistent with downstream recommendation tasks (e.g., image representations are originally trained for image-text matching); the latter means multimodal representations cannot be optimized via gradient updates inside the recommendation model (MLLM-generated representations are often used only as extra inputs to the recommendation model).

The QARM framework proposes Item Alignment (an item alignment mechanism) and Quantitative Code (a quantitative coding mechanism).
- Alignment: fine-tune the pretrained multimodal model with business-specific user-item interaction data (e.g., high-quality Item2Item pairs) so the representations align with the downstream task. For example, adjust for causal product relationships in e-commerce scenarios rather than general semantic alignment.
- Quantization: convert the aligned multimodal representations into learnable discrete code IDs (e.g., VQ vector quantization and RQ residual quantization), supporting end-to-end training. The code IDs serve as input features to the recommendation model (e.g., user interest sequences, item attributes), replacing static representations.

## ASH & ASMOL
Traditional pre-ranking is often treated as a mini version of ranking: lightweight models evaluated with ordered-list metrics like AUC. But this approach's offline evaluation is inconsistent with online A/B test results. Moreover, the goal of pre-ranking is not really to produce an ordered list, but a high-quality unordered candidate set.

The paper[^15] argues that blindly imitating ranking models limits pre-ranking's potential; one should focus on set quality rather than mere consistency.

The paper introduces the ASH (All-Scenario Hitrate) metric, which pools purchase positives from different scenarios such as recommendations and shopping carts, mitigating single-scenario sample bias and measuring pre-ranking candidate-set quality more comprehensively. Experiments show ASH is strongly correlated with online GMV, outperforming traditional metrics like AUC (Area Under the ROC Curve)[^17] and ISPH (In-Scenario Purchase Hitrate).

The paper also proposes the multi-objective learning framework ASMOL:
- Full-space training samples: integrate impression samples, ranking candidate samples, and pre-ranking candidate samples, covering a more comprehensive data distribution.
- Multi-objective loss: jointly optimize impression, click, and purchase tasks, learning the priority order (purchase > click > impression).
- Knowledge distillation: distill knowledge from the ranking model, but only on impression samples (to avoid noise interference).
- Single-model architecture: one unified model handles multiple tasks, outperforming traditional multi-model strategies.

## COPR
COPR[^16] observes that relatively lightweight pre-ranking models are less capable than complex ranking models, so it models alignment with ranking results as the objective. Through chunked sampling and rank alignment, it relaxes the goal from score alignment to rank alignment, effectively mitigating the model-capacity gap and the amplification of bid errors. Its plug-and-play design adapts to various pre-ranking models and was successfully deployed in Taobao's advertising system, significantly improving ad performance and platform revenue. This work offers new ideas for collaborative model optimization in cascaded architectures.

COPR lacks an exploration strategy. Chunked sampling may retain ad groups of different priorities, and ΔNDCG weighting focuses more on the ranking correctness of top ads — this mainly affects ranking accuracy rather than diversity. If the top ads themselves lack diversity (e.g., repeatedly recommending similar high-CTR ads), it may trigger an information cocoon.



[^1]: MOBIUS: Towards the Next Generation of Query-Ad Matching in
Baidu’s Sponsored Search. [[pdf]](https://arxiv.org/pdf/2409.03449)
[^2]: Learning from Negative User Feedback and Measuring
Responsiveness for Sequential Recommenders. [[pdf]](https://arxiv.org/pdf/2308.12256) 
[^3]: CPM (Cost per Mille) is the cost per thousand ad impressions, suited for brand exposure.
[^4]: CTR (Click-through Rate) is the user click rate, suited for search ads.
[^5]: CPA (Cost per Action) is the cost per user action (e.g., download, purchase), suited for deep-goal conversion.
[^6]: ROI (Return on Investment) is the ratio of revenue to cost.
[^7]: SSB (Sample Selection Bias) refers to the problem where the sample-selection process in training data is not random or does not match the real data distribution, hurting model performance. In recommender systems, popular items often account for most user-interaction data (a highly skewed data distribution is the norm), while data for long-tail (unpopular) items is too sparse; if a model relies too heavily on popular-item data, recommendation quality for long-tail items suffers. If negatives (items the user did not interact with) are sampled randomly, high-share popular items are easily over-sampled as negatives, which is unfair to popular items. In addition, if training data comes from a specific time period while test data comes from another, the model may overfit the historical period and fail to adapt to current trends.
[^8]: In-batch negative sampling is a simplified negative-sampling method: for each positive sample (user-item pair) in the current training batch, all other items in the batch serve as negatives. This obtains negatives directly from the current batch — no extra sampling step and no need to maintain a global negative pool — reducing computational overhead (large-scale recommender systems have too many items; randomly sampling negatives from the entire dataset on every training step is very time-consuming and also easily over-samples popular items as negatives). It has drawbacks too, such as insufficient negative diversity.
[^9]: Estimate item popularity (the probability of being sampled) and correct the model score with logQ to reduce the excessive penalty on popular items during negative sampling; see [Sampling-Bias-Corrected Neural Modeling for Large Corpus Item Recommendations](https://dl.acm.org/doi/abs/10.1145/3298689.3346996). The logQ correction here is inspired by the [sampled softmax model](https://www.iro.umontreal.ca/~lisa/pointeurs/importance_samplingIEEEtnn.pdf).
[^10]: Uninformative negatives generated by simple in-batch random negative sampling are often too easy; for a more balanced negative-sampling method, see: SimANS: Simple Ambiguous Negatives Sampling for Dense Text Retrieval. [[pdf]](https://arxiv.org/pdf/2210.11773)
[^11]: An Empirical Study of Selection Bias in Pinterest Ads Retrieval. [[pdf]](https://dl.acm.org/doi/pdf/10.1145/3580305.3599771)
[^12]: Mixture of Virtual-Kernel Experts for Multi-Objective User Profile Modeling. [[pdf]](https://arxiv.org/pdf/2106.07356)
[^13]: QARM: Quantitative Alignment Multi-Modal Recommendation at Kuaishou. [[pdf]](https://arxiv.org/pdf/2411.11739)
[^14]: CVR (Conversion Rate): the ratio of conversions to clicks.
[^15]: Rethinking the Role of Pre-ranking in Large-scale E-Commerce Searching System. [[pdf]](https://arxiv.org/pdf/2305.13647)
[^16]: COPR: Consistency-Oriented Pre-Ranking for Online Advertising. [[pdf]](https://arxiv.org/pdf/2306.03516)
[^17]: The ROC (Receiver Operating Characteristic) curve is a tool for evaluating binary-classifier performance: with the true positive rate (TPR) on the vertical axis and the false positive rate (FPR) on the horizontal axis, compute the (FPR, TPR) coordinate for each threshold and connect these points in order of decreasing threshold to form the ROC curve. AUC is the area under the ROC curve: AUC=1 is a perfect classifier, AUC=0.5 is random guessing.
[^18]: ISPH@k, the in-scenario purchase hit rate, measures whether the top-k candidates output by pre-ranking contain the item the user actually purchased. Its limitations: (1) when k equals the total number of pre-ranking output candidates, ISPH@k is always 1 and loses evaluative meaning. (2) The pre-ranking candidate set is only exposed after fine ranking, so ISPH@k reflects the selection of the entire ranking stage rather than the quality of the pre-ranking candidate set itself. (3) It only captures preferences within a single scenario.
