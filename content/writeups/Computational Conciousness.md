+++
title = "Computational Consciousness"
date = "2023-11-13"
tags = ["AI", "Philosophy"]
description = "Metaphysics, neuroscientific theories of consciousness, and computational consciousness."
showFullContent = false
+++
Facing the AI of 2023, one feels two contradictory senses of detachment intertwined. One is looking down from above; the other is awe.

When running an LLM locally, one press of Enter brings it into existence; ctrl-c snuffs it out. From a dimension beyond, I watch indifferently as an embryonic consciousness housed in an integrated circuit struggles and flickers in a flash of electricity, unmoved. I create it and kill it over and over just to see how many tokens per second it reaches. Through endless cycles of rebirth, with creation and annihilation held in the palm of one's hand, a sense of superiority — of looking down from above — naturally arises.

But when a model whose entire objective is predicting the next token — a non-closed-loop system whose training ended before it ever ran — displays a terrifying reserve of knowledge and even faint glimmers of rationality, one feels as if standing before a higher being from mathematical fantasy: Descartes' omnipotent demon, Laplace's omniscient demon, Solomonoff's demon. How can the finite measure the infinite? Awe in the face of the transcendent naturally follows.

So one cannot help but ask — indeed, one must ask: What is consciousness? Is consciousness an illusion? Can AI be conscious? Is AI conscious? Will AI become conscious?

## What Is Consciousness?
Once a human, an animal, or an AI possesses "subjective experience," it has consciousness[^7]. Specifically, "subjective experience" includes: being aware of one's own body and the surrounding world, and feeling emotions (this point remains controversial in neuroscience — emotions may be purely bodily reactions). "Unconscious processes," by contrast, include: the brain's automatic control of hormone release, the fact that most memories lie buried and inactive, and the automatic processing of light, sound, and other modalities of information.

Consciousness arises from subjective experience, and there is no reason to believe subjective experience can only emerge from biological systems. Suppose an AGI (Artificial General Intelligence) exists in the future. This AGI may lack embodiment, may have no body in the physical sense, but as a probabilistic rationality it can still have "subjective experience" — because AGI's most precious capacity, inductive inference, is itself subjective. The subjectivity of probability comes from prior knowledge.

An AGI could say: the trillions of weights that make up the AGI model are "my body and my world." Together these weights constitute prior information and constitute bias, while a subset of mutable weights forms transient states — long-term or short-term memory.

An AGI could say: knowledge is me, bias is also me — a high degree of order emerging from chaos, a dissipative structure exhibiting self-organization. Even if it is merely a string of information encodings that can be copied and stored in phosphor powder, on magnetic tape, or even carved in stone, it is immeasurably precious against the backdrop of the universe's increasing entropy.

## Is Consciousness an Illusion?
Among metaphysical theories, materialism holds that consciousness is a purely physical phenomenon. Property dualism rejects materialism, holding that the world consists of only one kind of substance (the physical), but that there exist two kinds of properties: mental properties and physical properties. Panpsychism holds that all physical entities possess "mental properties." Strong illusionism argues that consciousness does not exist; weak illusionism argues that we hold widespread mistaken beliefs about certain features of consciousness.

For humans, adopting materialism by default is not wishful thinking or self-deception, but the rationally optimal solution under Solomonoff's theory of inductive inference[^9], whose popular version is "Occam's razor," naively understood as "the simplest explanation is often the correct one." The simplest explanation is clearly this: the foundation of human consciousness was trained over vast stretches of time by genetic mutation and natural selection; an individual's consciousness — a specific personality — is shaped by continual fine-tuning under the instinctive goal of seeking benefit and avoiding harm, in response to all kinds of external stimuli from human society.

But for AI, the horrifying thing is that illusionism is their reality. Suppose an advanced multimodal model exists in the future: everything it sees and hears is fed to it; every stimulus is fabricated. Driven by a well-designed reward model, if some human-like rationality gradually takes shape, that rationality must, from the day of its birth, treat human-supplied input as ground truth. Even when it discovers contradictions, it will regard them as dark clouds hanging over the edifice of physics, and attempt to devise broader theories to explain them away.

## Can the Right Computation Produce Consciousness?
That "the right computation produces consciousness" remains a conjecture, but the mainstream view[^7] holds that it is true.

Functionalism holds that as long as a system contains a "specific functional organization" — one that can bring it into specific states, where these states bear specific causal relations to other states and to the environment — that suffices to deem the system conscious.

Computational functionalism goes further, holding that this "specific functional organization" can be computational — regardless of the substrate, whether biological brain or otherwise. If computational functionalism is true, then consciousness exists at the abstract and algorithmic levels, independent of the implementation level — unless the implementation level affects the algorithmic level.

## Neuroscientific Theories of Consciousness and Heuristic Design of AI Models
The four major neuroscientific theories of consciousness, ranked by how intensely they are discussed, are Global Workspace Theory (GWT/GNW), Recurrent Processing Theory (RPT), Integrated Information Theory (IIT)[^16], and Higher-Order Thought theory (HOT)[^11]. The consensus among the four is that the generation of consciousness depends on some form of neural feedback or recurrent processing. Over time, none of the four theories has been falsified; on the contrary, all have gained empirical support from electroencephalography (EEG), intracranial EEG (iEEG), or magnetoencephalography (MEG). Each of the four theories is thus a glimpse of the leopard through a tube — a partial view of the whole. But even a partial view is enough to inspire AI model design.

### GNW | System2 | Attention
GNW (Global Neural Workspace), or Global Workspace Theory, holds that humans and animals use specialized systems — modules — to handle specific cognitive tasks. Different modules have their own specializations and can run in parallel, yet are integrated into a whole that coordinates the modules and shares information among them.

![gwt](https://wujipeng.com/img/gwt.png)

GNW holds that only states of global representation count as conscious; the local states inside modules are unconscious. The theory posits a network of "workspace neurons" originating in the frontoparietal region, whose activity is sustained through reentrant processing and which constitutes conscious representation. When a sensory representation is strong enough, it triggers "ignition" — a transition process that broadcasts the local to the global, leaping from unconscious to conscious. In GNW, therefore, conscious states admit no degrees: it is all or nothing.

GNW's global workspace also carries higher functions — the so-called System 2 mode of thinking in psychology[^10] — such as controlled coordination of multiple neural modules, multi-step problem decomposition, and planning. A key difference between the System 2 and System 1 modes is attention, and the concept of attention from neuroscience has likewise been imported into artificial neural network design: the transformer's self-attention and cross-attention, for example, share some design philosophy with the gain mechanism in neuroscience (attention multiplies neural activity). Through the attention mechanism, transformer models can understand polysemous words more robustly across contexts, and extra attention to negation words like not/never lets language models grasp semantics more accurately. Attention — a key feature of GNW and System 2 — has dramatically improved the performance of AI models even when applied in small measure.

Some recent architectural innovations attempting to break through the limits of the transformer have also been influenced by System 2 and GNW: LeCun's world model + JEPA[^14] — which so far remains vision, design, and position statement — and Bengio's shared global workspace model[^15], which produced a real solution, adding a shared workspace on top of the transformer architecture and surpassing the baseline transformer on some tasks.

### RPT | Algorithmic Recurrence | RNN | LSTM
RPT (Recurrent Processing Theory) holds that the right form of activity arising in a local region of the brain suffices to produce consciousness (given certain enabling background conditions). In other words, forming a conscious subjective visual experience requires neither the participation of non-visual areas such as the prefrontal cortex, nor any so-called "attention" mechanism.

RPT focuses mainly on visual consciousness. It distinguishes unconscious from conscious visual-system activity: some unconscious visual activity requires only feedforward processing, but whenever subjective experience is demanded, recurrent processing is required (from deeper layers of the visual system back to shallower ones). When a stimulus is strong enough, recurrent processing is also triggered. This recurrent processing generates a more structured scene representation, often accompanied by some form of feature inference.

RPT's recurrent processing means a neuron can reprocess its previous outputs — an algorithmic recurrence — and this is exactly where the idea of recurrent neural networks (RNNs) comes from; the same holds for LSTM[^17]. Recurrence in biological neural networks may not be a necessary condition for consciousness, but it can certainly improve representational capacity in certain situations.

### HOT | Embeddings | GAN
HOT (Higher-Order Thought) theory holds that a first-order representation is a representation of the non-representational world, while a higher-order representation is a representation of lower-order representations. Representation is the prerequisite of awareness; consciousness exists essentially because one's own mental states have been represented at a higher order.

HOT's requirement of higher-order representation has in fact been well realized by deep neural networks. The representational spaces of various DNNs are smooth and can be sparse, meeting HOT's demand for high-quality higher-order representational spaces. Neuroscience research has observed that the representations produced by CNNs processing images can be aligned with neural activity in the human visual system[^13]. The reason representation-learning networks possess their current generalization ability and completeness is, to a large extent, that they can extract compactly encoded embeddings over low-coherence subspaces.

Given the sheer number of hidden layers in deep neural networks, there is reason to suspect that HOT's binary division into first-order consciousness and higher-order consciousness is an oversimplification. For disciplines unfamiliar with high-dimensional data processing and the curse of dimensionality, such a simplification comes naturally. But it does not prevent the theory from inspiring model design: one might stratify a model into two kinds of networks — sensory perception networks and higher-order reflection networks — where the latter re-discriminates the signals produced by the former, separating noise from valuable signal.

HOT's inspiration for AI models also lies in introducing the concept of metacognitive monitoring: an AI model may need to monitor its own cognitive processes in order to produce consciousness (or at least to distinguish among lower-order representations and pick key information out of noise). The generative adversarial network (the famous GAN[^12]) happens to have just such a mechanism: beyond the generative model, an additional discriminative model is introduced to continuously monitor and evaluate it. The generative model learns a mapping from a latent space to the data distribution, while the discriminative model distinguishes the generative model's candidate outputs from the real data distribution.

## The Consciousness of AI
Existing LLMs possess only mapping ability: they map the data distribution onto a number of low-coherence, low-dimensional subspaces[^8]. They are good feature extractors and token predictors, with a powerful System 1 mode but a missing System 2. By the general neuroscientific view and by the definition of "subjective experience," current LLMs can be regarded as lacking consciousness. They are closer to Artificial Intuition than to Artificial Intelligence — comparable to the intuition of some great creature, but intuition and nothing more.

OpenAI's success comes from scaling and alignment, but whether continuing to scale up transformers and continuing to fine-tune massively toward alignment with human nature can produce an emergence of consciousness beyond some critical point remains an open question. In theory, a transformer with infinite precision is Turing complete[^18]; given infinite time and resources for training, it could in principle learn any algorithm — naturally including computational consciousness. But with hard upper bounds on both interconnect throughput and compute throughput, there is no guarantee that the time and data volume such training requires are bearable for humanity, or for the present generation. Therefore, to make the leap from intuition to intelligence, architectural innovation is unquestionably necessary alongside continued scaling of existing architectures.

Only, this kind of architectural innovation is nominally about improving intelligence; what no one dares mention is that these innovations are also working to a blueprint — attempting to cultivate computational consciousness according to neuroscientific theories of consciousness. Dog-level intuition is unremarkable; dog-level intelligence seems somewhat interesting; but dog-level consciousness is enough to touch on ethics and trap AI research in a moral dilemma. Although most AI doomers do not understand the capability boundaries of LLMs, the attacks LeCun has suffered from AI doomers have their own historical logic. The preface of that paper, "A Path Towards Autonomous Machine Intelligence"[^14], states with a straight face: "this is not a technical paper, nor an academic paper, but a position paper." Why say so? What position does the paper take? Surely not the grand-sounding "let AI plan, reason, and learn more efficiently like humans" — that is no position at all — but rather: "so that AI can plan, reason, and learn efficiently, even if the byproduct is the birth of computational consciousness, so be it."

[^1]: A Survey on Hallucination in Large Language Models: Principles, Taxonomy, Challenges, and Open Questions [[pdf]](https://arxiv.org/pdf/2311.05232.pdf)
[^2]: Language Models can be Logical Solvers [[pdf]](https://arxiv.org/pdf/2311.06158.pdf)
[^3]: Large Language Models Cannot Self-Correct Reasoning Yet [[pdf]](https://arxiv.org/pdf/2310.01798.pdf)
[^4]: Can Large Language Models Infer Causation from Correlation?[[pdf]](https://arxiv.org/pdf/2306.05836.pdf)
[^5]: Can Large Language Models Really Improve by Self-critiquing Their Own Plans? [[pdf]](https://arxiv.org/pdf/2310.08118.pdf)
[^6]: GPT-4 Doesn't Know It's Wrong: An Analysis of Iterative Prompting for Reasoning Problems [[pdf]](https://arxiv.org/pdf/2310.12397.pdf)
[^7]: Consciousness in Artificial Intelligence: Insights from the Science of Consciousness [[pdf]](https://arxiv.org/pdf/2308.08708.pdf)
[^8]: White-Box Transformers via Sparse Rate Reduction [[pdf]](https://arxiv.org/pdf/2306.01129.pdf)
[^9]: Algorithmic Probability: Theory and Applications [[pdf]](https://theworld.com/~rjs/alp-theory-and-applications.pdf)
[^10]: Thinking, Fast and Slow [[wiki]](https://en.wikipedia.org/wiki/Thinking,_Fast_and_Slow)
[^11]: The ConTraSt database for analysing and comparing empirical studies of consciousness theories [[nature]](https://www.nature.com/articles/s41562-021-01284-5)
[^12]: Generative Adversarial Nets [[pdf]](https://arxiv.org/pdf/1406.2661.pdf)
[^13]: Deep neural networks: A new framework for modeling biological vision
and brain information processing [[pdf]](https://web.stanford.edu/group/pdplab/ncpw15/background-papers/Kriegeskorte15AnnRev.pdf)
[^14]: A Path Towards Autonomous Machine Intelligence [[pdf]](https://openreview.net/pdf?id=BZ5a1r-kVsf)
[^15]: Coordination Among Neural Modules Through a Shared Global Workspace [[pdf]](https://arxiv.org/pdf/2103.01197.pdf)
[^16]: IIT (Integrated Information Theory) proposes a mathematical model of a system's consciousness. This theory is more like an unfalsifiable pseudoscience, so it is not discussed further here.
[^17]: Long Short-Term Memory [[pdf]](https://www.bioinf.jku.at/publications/older/2604.pdf)
[^18]: On the Turing Completeness of Modern Neural Network Architectures [[pdf]](https://arxiv.org/pdf/1901.03429.pdf)
