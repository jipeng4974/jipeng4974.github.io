+++
title = "The Isomorphism Between Lenses and Encoders"
date = "2026-07-01"
tags = ["Physics", "AI"]
description = "Camera lenses and encoders exhibit a striking isomorphism."
showFullContent = false
+++

## Lens $\cong$ Encoder
A camera lens and an image or audio encoder are isomorphic; both are fundamentally dimensionality-reduction functions that take high-dimensional physical reality as input: $$ f: \mathbb{R}^n \to \mathbb{R}^d, d \ll n$$

A lens compresses an infinite light field into a pixel matrix on the camera sensor:
```
 3D Scene
    ↓
   Lens
    ↓
2D Sensor Image
```

An encoder compresses some artificial sampling of infinite physical reality into a low-dimensional vector:
```
 Waveform
    ↓
  Encoder
    ↓
 Embedding
```

## Astigmatism & Coma $\cong$ Anisotropy

I recently bought an RF 28mm f/2.8 pancake lens and found that it uses aspherical resin elements with a special shape. Resin is naturally non-uniform, so the lens exhibits astigmatism (different resolution in the vertical and horizontal directions) and coma (off-axis error). This is structurally identical to the natural non-uniformity of a model's "material" in representation learning and the natural anisotropy of its outputs. It occurred to me that, in the eyes of a higher-dimensional being, all of humanity's various activities might be one and the same thing. A Canon optical designer designing a lens and me trying to build an audio encoder are fundamentally doing the same thing: projecting high-dimensional physical reality onto a low-dimensional representation through a nonlinear transformation.

The non-uniformity of resin elements makes anisotropy in the final output inevitable. The same holds for our representation models: the embeddings they produce are almost certainly strongly anisotropic.

## Overcoming Anisotropy
An optimization objective independent of any specific downstream task emerges naturally — isotropy. Optics engineers and ML engineers alike are fighting anisotropy, striving for isotropy.

- If an optics engineer switches to higher-grade glass elements and employs various compensation elements, the lens can approach isotropy. The final image plane will then show no comet-shaped light spots from coma and no difference in sharpness between the longitudinal and transverse directions — earning the lens the red-ring L-badge name.
- If an ML engineer uses regularization such as SIGReg to constrain a model to produce embeddings that are more uniform across dimensions, those embeddings generalize to a wider range of applications — not limited to the single supervised setting targeted by the loss+sampler design, but also usable in cross-modal alignment, clustering, and other scenarios — earning the model the foundation model name.

## Adapting to Anisotropy
Another approach accepts anisotropy as a fact and exploits it to cut costs. Camera correction algorithm/firmware engineers and HPC engineers working on large-scale vector retrieval systems are both adapting to anisotropy.

- For example, Canon's digital correction technology and lens design complement each other: using resin elements in the RF 28mm f/2.8 actually saves cost.
- Similarly, Google's engineers accepted the fact that most models' embeddings are not so "round" and proposed ScaNN's anisotropic quantization, achieving more aggressive ANNS performance optimization.

This approach is effective in engineering, but it lacks robustness and generality. Once the in-camera correction is lost (suppose we force-mount the lens onto some strange camera body), the RF 28mm f/2.8's image quality degrades noticeably. And once applied outside ANNS scenarios, anisotropic quantization can no longer deliver the minimal error it achieves in vector retrieval.
