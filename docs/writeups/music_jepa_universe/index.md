# Music JEPA and the Three-Body Universe

> Dimensions of survival, dimensional collapse, resurrecting dead dimensions, the laws of the universe, the Edenic universe, surviving subspaces, and the deterrence term λ.

---

LLMS index: [llms.txt](/llms.txt)

---

The previous post, [Music JEPA Regularizers](https://jipeng4974.github.io/writeups/music_jepa_reg/), combined mathematical rigor, engineering dullness, and AI mediocrity; after finishing it, I always felt something had gone unexpressed. This post attempts an analogy — as effective as I can make it — with the Three-Body universe, to share the imaginings that carried me across nebulae during the Music JEPA thought experiments, and my journey from astonishment to delight.

## The Multi-Concept Mapping Table

| Music JEPA SSL | The Three-Body universe |
|---|---|
| embedding space | the universe |
| music representation model | civilization in the collective sense |
| reward hacking | the survival strategy of civilizations forced to cheat in order to survive in a cruel universe |
| effective ranks | the number of dimensions across which a civilization actually unfolds its survival |
| invariance loss (the alignment rule) | the laws of the universe |
| dimensional collapse (erank decline) | the dual-vector foil / civilizations migrating into lower dimensions |
| the surviving subspace of false Gaussian prosperity / the low-rank blind spot | black domains / low-dimensional pocket universes |
| collapsed ranks ($\sigma\to 0$) | dead dimensions / heat-death dimensions |
| $N(0,I_{512})$, the ideal isotropic Gaussian distribution | the Edenic Age of the early high-dimensional universe |
| the regularizer that resurrects dead dimensions | the Returners movement that abandons pocket universes and returns their mass |
| the contrastive term pushing dissimilar examples apart | the dark forest law among individual civilizations |
| $\lambda_{\text{reg}}$ | the influence of Returnerism |
| $\lambda_{\text{infonce}}$ | the Swordholder's degree of deterrence |

## Music JEPA's Training Objectives and Experimental Observations
Music JEPA's training objectives include:
- Objective 1: teach the model the invariance between anchor music and degraded music — that is, to obey the laws of the universe.
- Objective 2: make the distribution of the model's embedding space approach an isotropic Gaussian.
- In addition, practice often suggests an Objective 3: add a contrastive learning term so the model learns to push some obvious negatives away from the anchor — although this has nothing to do with JEPA per se, contrastive learning is simple and effective, and beyond objectives 1 and 2 it supplies some extra training signal that can't be too far wrong.

The previous post, [Music JEPA Regularizers](https://jipeng4974.github.io/writeups/music_jepa_reg/), mentioned two interesting experimental observations:
- SIGReg and VISReg can prevent collapse, but both struggle to resurrect dead dimensions. VISReg is slightly stronger, but under a conventional λ schedule its training momentum for raising erank is still clearly insufficient.
- If you give VISReg an absurdly high λ — going all-in toward the isotropic Gaussian — you find that not only does it fail to resurrect dead dimensions, it actually causes dimensional collapse: while the VISReg loss decreases, erank slowly grinds downward. In other words, training falls into a local low-rank optimum, or a "low-rank blind spot."

Some newly observed phenomena to add:
- Under a conventional λ schedule, the eranks of VISReg and SIGReg converge to a specific value K and lose all growth momentum; this K is in fact close to the upper bound of the intrinsic dimensionality of the representation manifold after whitening.
    - Both VISReg and SIGReg do genuinely possess the ability to resurrect dead dimensions, though they differ slightly in rate and stability.
    - Beyond the invariance alignment rule, the richness of the content involved in training also affects the upper bound of the representation manifold's intrinsic dimensionality.
- Under an ultra-strong λ constraint together with an aggressive invariance objective, the model can, during part of training (the early-to-middle phase), occasionally burst into an ideal superposition of decreasing invariance loss and rising erank — but this burst is unstable, and collapse recurs as training steps accumulate.

## The Laws of the Universe Are the Cruelest Weapon; the Law Must Not Be Too Harsh
The dark forest law among interstellar civilizations is dark but not truly cruel; what is cruel is the unstoppable loss of the universe's degrees of freedom (heat death, dimensional reduction, the lowering of the speed of light).

Likewise, in Music JEPA self-supervised representation learning, what is truly cruel is Objective 1, invariance learning. An overly aggressive invariance loss drives the model into a painful escape to lower dimensions: by deleting information from the representation manifold, it satisfies the invariance alignment the way one cuts off toes to fit a shoe — which means a more severe collapse of intrinsic dimensionality, far worse than the local distortions of geometry caused by the false-negative pairs that a contrastive loss can hardly avoid.

So **do not set an exaggerated invariance objective**. In one earlier experiment, I forced the model to keep 1.5s–15s subsets cropped from a 15s anchor aligned with the anchor even under heavy time-stretching, pitch-shifting, noise, and distortion. The result was a spectacular escape to lower dimensions, and the loss of the representation manifold's intrinsic dimensionality could not be rescued even by an ultra-high $λ_{reg}$. The flattening of the erank growth curve was not a design flaw of the regularizer — subsequent whitening experiments proved that the dimensionality ceiling corresponding to that invariance rule had in fact been reached.

The law of Qin decreed that those who missed their deadline were all to be executed; laws that cruel did not bring a sturdier order — they invited a realm seething with rebellion.

Thus, in the JEPA context, the true meaning of curriculum learning is not to let the learner climb step by step from easy to hard and master ever more magical alignment tasks[^1]. It is to abandon the remaining curriculum at the right moment, when you find that the learner, caught under the double constraint, has no choice but to painfully mutilate itself and flee toward lower dimensions — thereby averting catastrophic collapse.

## Chasing the Isotropic Gaussian and the "Edenic Age" While Avoiding the Low-Rank Blind Spot
The early high-dimensional Three-Body universe had abundant and uniform degrees of freedom: no black domains, no dimensional warfare, no dark forest wars born of overcrowding, no pocket universes tucked away in corners; civilizations had ample room to live, so it was called the "Edenic Age." The aim of Returnerism in the Three-Body universe is to give back the dimensions, repair the great universe, and return to the "Edenic Age."

Objective 2 of Music JEPA is precisely the return to the "Edenic Age" of the isotropic Gaussian. Why is the isotropic Gaussian the ideal distribution? Because it has been proven mathematically to be the ideal distribution for all downstream tasks. Only when its embedding space approaches an isotropic Gaussian can a model claim to be a general-purpose foundation model. Take retrieval as an example: even with no negative mining at all and no contrastive learning, the isotropic Gaussian naturally guarantees that the negative cosine p99999 stays low enough — the representations are inherently fit for retrieval over large-scale vector stores.

Among the three regularizers introduced in the previous post, SIGReg and VISReg both target the isotropic Gaussian directly, while VICReg has a stronger ability to resurrect dead dimensions and prop up the floor of erank — it just doesn't guarantee the shape. So combining VICReg with VISReg, or introducing VICReg's covariance term into VISReg (which doesn't affect the isotropic Gaussian objective at all), helps avoid falling into the "low-rank blind spot" during JEPA learning[^2].

The so-called "low-rank local optimum / low-rank blind spot" is a mirage: the false Gaussian prosperity of a surviving subspace. Under the double constraint of an aggressive invariance objective and a strong $λ_{reg}$, both SIGReg and VISReg have some probability of falling into this local optimum — the regularizer loss is indeed decreasing, yet erank grows feebly, or even grinds slowly downward. On the monitor you watch the dimensions of the embedding space die off one by one, while the living dimensions grow ever more crowded and lively, marching in proud order toward the Edenic form — a scene that can't help but recall the civilizations in the Three-Body universe that voluntarily reduce their own dimensionality and flee into lower dimensions: in the low-dimensional pocket universes a new order is being born, civilizations bustle, and from the perspective of low-dimensional beings it is a flawless Eden; but from the perspective of the survivors left behind in the high-dimensional great universe, every trace of the old-generation civilizations is turning to ash and smoke.


## Introducing a Contrastive Term — The Delicate Balance of Three Tensions
Objective 3 in Music JEPA practice is not very JEPA, but it works, and its negative effects are manageable, so it is an option.

By analogy to the Three-Body universe, introducing a contrastive term is like introducing an extra dark forest law among civilizations that already face an inevitable fate of ruin and long to return to the "Edenic Age." This third tension may be insignificant in the grand picture of the universe's demise, but at a specific point in time (when no more GPU hours can be thrown in) and for a specific individual civilization (a specific downstream task), it can still play a decisive role.

The cost is a new hyperparameter, $λ_{infonce}$, which must be controlled carefully to reach some delicate balance.
- If $λ_{infonce}$ is too low or zero, then before the Edenic Age of the isotropic Gaussian truly descends, there will always be abnormally high negative cos. Just like Cheng Xin as the Swordholder.
- If $λ_{infonce}$ is high enough, the deterrence level is high enough to preserve the floor on the distance between dissimilar examples. Just like Luo Ji as the Swordholder.
- If $λ_{infonce}$ is too high, the guest usurps the host: it weakens JEPA's main objective. Choosing mutual destruction at the slightest disturbance also cuts off civilization's path of healthy evolution.

Under these three tensions, what is the ideal ending?
- From Music JEPA's point of view, the ideal ending is that the model learns moderately difficult degraded crop invariance alignment while bringing the embedding distribution roughly close to an isotropic Gaussian — attaining the rank of "foundation model."
- From the point of view of the Three-Body universe's civilizations, the ideal ending is that civilizations adapt to the laws and corollaries of the universe — the constancy of light speed, entropy increase and heat death, the dark forest — while working together to carry out the Return, resurrecting the dead dimensions of the great universe and returning to the "Edenic Age."

[^1]: Under a harsh invariance objective, a ViT can indeed learn out-of-thin-air alignments between ultra-short durations and long-range melodies — even alignments between unrelated segments of the same song.
[^2]: There may of course exist more mathematical, more rigorous, and more complete regularization methods to overcome this low-rank blind spot; I simply lack the mathematical foundation and the time to explore them right now.
