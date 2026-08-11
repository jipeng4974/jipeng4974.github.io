+++
title = "Music LeJEPA"
date = "2026-07-13"
tags = ["AI"]
description = "First attempt at Music LeJEPA (to be continued)"
showFullContent = false
+++

# The Evolution of Representation Learning
## We Are Practitioners of the Representation School in Music
ByteCover3's success was built on the success of Saining Xie and Kaiming He's ResNeST, and also made use of Yann LeCun's contrastive learning.

Music LeJEPA attempts to replicate LeCun's new method from the vision domain in music representation. Although it is not as mystical as a world model (what gets predicted/aligned is not world representations but merely melody representations), it is indeed a fundamental paradigm shift.

Our music recognition model follows the orthodox representation school. Yann LeCun and Saining Xie are leading figures of this school, having made some progress under the JEPA framework, and are currently attempting to build Real World AI starting from representation learning.

## The Meta-Belief of the Representation School
The core philosophical view of the representation school is that the essence of AI is neither multimodal nor generation, but representation. Generation is a byproduct of understanding.

We must not, because of generative AI's enormous commercial success, mistakenly believe that "generation is understanding," or that "generative models can solve understanding problems along the way."

In fact, before VAE, autoencoders are not good representation learners. And VAE, in turn, shares a similar prior with LeJEPA.
- Vanilla autoencoders: reconstruction loss = the L2 loss between input and output.
- Sparse autoencoders: introduce an additional sparsity loss beyond the reconstruction loss — restricting only a small fraction of neurons to be activated, so the output embeddings become less entangled and more linearly separable. Currently mainly used in interpretability research.
- Denoising autoencoders: take degraded views of the original signal as input and try to reconstruct the original input; they are the origin of score-based diffusion in later generative models.
- Masked autoencoders, the famous MAE: Kaiming attempted to replicate BERT's masked autoencoding on ViT. It was considered a success at the time, but looking back years later, it was actually a failure (enormous resources invested, pixel reconstruction as the target, and the trained representations lack linear separability, are mixed with too many irrelevant features, and are at least unusable in the retrieval domain).
- Variational autoencoders: the truly revolutionary paradigm shift, introducing beyond the reconstruction loss a KL term that pulls the latent closer to a standard Gaussian. This is strikingly similar in spirit to LeJEPA's introduction of the SIGReg regularization term!

The problem with generative representation is that the training objective diverges from "semantic-level understanding," mixing in more superficial understanding to some degree — pixel-level reconstruction, for instance, inevitably makes the model learn local textures. Even VAE cannot rival representation-focused models on understanding task metrics.

The success of both VAE and LeJEPA stems from this inductive bias, or design philosophy: you cannot only optimize the task objective; you must simultaneously impose statistical constraints on the "geometric structure" of the latent space.

## The Evolution of SSL Before LeJEPA

```
One lineage evolved from AE into generative models.
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

The other lineage is self-supervised representation learning.

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

SSL is not limited to the masked generative self-supervision line. The most natural route is contrastive SSL, such as SimCLR and MoCo, which rely on constructing positive/negative sample pairs and hard mining — a direct transfer of supervised contrastive learning into the self-supervised domain, and one that naturally inherits a whole series of pain points from supervised contrastive learning: contrastive loss training instability, the difficulty of building a reliable hard mining pipeline, and the need for huge batch sizes.

Following contrastive SSL and generative SSL, another SSL route emerged, called reverse self-distillation, popularized by the Dino series; MERT in the music domain also used similar techniques. Looking back now, the self-distillation route is merely an engineering remedy for a mistaken attempt — it has no interpretability and no reference value whatsoever — but that does not mean DINOv3 as a whole has no reference value. DINOv3's loss consists of four parts, and self-distillation is only one of them; its Kelo loss is also a kind of geometric regularization, similar in spirit to LeJEPA's SigReg regularization.

## LeJEPA: Turning SSL from Alchemy into Science
Compared with previous-generation SSL, LeJEPA is minimalist:

$$L_{LeJEPA} = (1-λ) L_{invariance} + λ L_{sigreg}$$

LeJEPA simplifies the JEPA objective into invariance loss (the so-called latent prediction) + the geometric distribution regularizer SIGReg.

The so-called invariance loss, in the audio retrieval domain, is essentially the L2 loss between the projected embeddings corresponding to global views and local views. Similar practices existed before, such as minimizing the distance between positive pairs in SimCLR contrastive SSL, but previous methods neither fully solved the representation collapse problem nor found the mathematically optimal geometric distribution constraint.

# Some Engineering Practices
## Initial Weights
Practice shows that DINOv3, a SOTA pretrained vision model trained mainly on natural images, is not suitable as initial weights for SSL. Compared with training from initialized weights, the invariance loss converges slightly faster, but the sigreg loss converges slowly and the initial effective ranks are extremely low; after long training the effective ranks can improve, but with an upper bound — they cannot keep increasing.
- A model pretrained too long on natural images tends to treat all grayscale spectrograms as extremely similar. DINOv3's 512 output dims are so stingy that only 4 effective dimensions are allocated to cqt/logmel spectrogram images. After 40,000 steps, the effective ranks grow to 40, with the growth curve flattening.
- Pretraining from randomly initialized weights: after 30,000 steps, the effective ranks grow to 100.

## Curriculum Learning with Progressive Difficulty
To train a Degradation-robust Melody Matching model, we need to construct various transforms that do not destroy the melody, such as cropping, speed/pitch shifting, applying various amplitude envelope changes, and noise addition based on a sufficiently rich noise bank. Most of these invariances are relatively easy to learn, but some are very difficult and should only be attempted after the model has built a certain foundation — such as fairly extreme noise addition and extreme speed/pitch shifting.

In addition, we need to implement a sigreg λ scheduler that increases the isotropic regularization weight as training steps progress.

## Contrastive Post-training
Although the non-contrastive SIGReg regularizer can deliver good performance on any specific downstream task, the climb of effective ranks is slow and the pretraining investment is considerable. To see results quickly, one can perform contrastive post-training for a specific downstream task with the help of human-annotated data.

The traditional Triplet loss is very effective in certain scenarios, but has even more drawbacks:
- The vast majority of triplets have no gradient.
- The triplets that do have gradients often introduce extra false negatives.
- Investing massive human effort in manual annotation is nearly impossible in many scenarios.
- Truly hard negatives produce gradients that are too large, causing training oscillation — forcing one to settle for semi-hard negatives instead.
- Offline mining becomes increasingly stale as the embedding distribution shifts with training steps.
- Online mining demands extremely large batch sizes.
- Mining rules pile up ever more complex, with more and more heuristics, increasingly violating the bitter lesson.

By contrast, most contrastive SSL opts for InfoNCE, in which the entire batch participates in the computation, removing the complex and fragile hard mining from the training pipeline.

The X-sample contrastive loss proposed by LeCun's team last year redefines the object of contrastive learning — learning based on the complete sample similarity graph rather than on pairs. InfoNCE can also be regarded as a crude similarity graph — in the similarity matrix, only positives are 1 and everything else is 0 — crudely zeroing out all negatives, which is a waste of information.

In his June blog post [Forced Margin Projection](https://kexue.fm/archives/11784), Jianlin Su proposed an ingenious margin loss implementation that is well suited for widening the margin between the anchor and negatives, fits retrieval tasks quite well, and is also worth trying.


## Scalable Degraded View Synthesis
Do expensive waveform-based degradation synthesis offline, such as speed shifting, pitch shifting, reverb, and NoiseBank noise addition. Set several magnitudes of variation for each degradation, and randomly pick from them when generating views online.
- Maintain a noise bank with sufficient diversity (vocal singing, ambient noise, white noise, various UGC audio, music, TV). Adding noise in the waveform is more reasonable and more faithful to reality.

Do cheap cqt/logmel tensor-based degradation synthesis online, such as random amplitude envelopes (slowly varying along the time axis), local energy perturbation, random noise floor (adding low-amplitude noise to the spectrogram), time-frequency masking, dynamic range compression, frequency-domain EQ (random frequency response curves, multiplying each frequency band by a slowly varying gain), and spectral tilt (overall brightening/darkening, tilting the energy between high and low frequencies).

The Waveform -> CQT step is quite expensive (CPU overhead remains considerable even after switching to LogMel). Moving all waveform synthesis offline ensures that the training pipeline does not introduce a CPU bottleneck in the data pipeline.
- Another feasible option is to deploy a large-scale CQT extraction cluster (but I am unwilling to introduce dependencies on the network and external systems into a general training framework).
- Yet another idea is to directly use cheap log mel spectrograms in place of the orthodox CQT representation; this is very likely feasible, and an ablation study can be done later.

