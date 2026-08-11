# Writeups

> Long-form notes on systems, AI, math, and everything in between

---

LLMS index: [llms.txt](/llms.txt)

---

Section pages:

- [Music JEPA and the Three-Body Universe](/writeups/music_jepa_universe/): Dimensions of survival, dimensional collapse, resurrecting dead dimensions, the laws of the universe, the Edenic universe, surviving subspaces, and the deterrence term λ.
- [Music JEPA Regularizers](/writeups/music_jepa_reg/): Revisiting anti-collapse regularizers for Music LeJEPA
- [DSpark](/writeups/dspark/): DSpark: a low-rank bigram table buys back intra-block dependencies, and a confidence head applies admission control to verification length.
- [A Measurable Metric of Free Will](/writeups/free_will/): Self-origination = internal sensitivity - external sensitivity.
- [Kimi K3: Scaling LLMs Across Sequence, Depth, and Width](/writeups/llm_arch/): LLM架构演化，从dense transformer到Kimi K3。
- [Music LeJEPA](/writeups/music_lejepa/): First attempt at Music LeJEPA (to be continued)
- [The Isomorphism Between Lenses and Encoders](/writeups/lens_encoder/): Camera lenses and encoders exhibit a striking isomorphism.
- [Beyond Representation Learning: Prediction-Driven Encoders + Learnable Sensors](/writeups/beyond_representation_learning/): Existing multimodal representation learning is primitive, isolated, and incomplete — it still needs prediction-driven encoders and learnable sensors.
- [Cognitive Phase Transitions](/writeups/cognitive-phase-transitions/): The invariant of human intelligence's value across cycles is the capacity for cognitive phase transitions.
- [Semantic Depth & Music Re-ID](/writeups/reid/): This post proposes a semantic depth metric and discusses Music Re-ID from the angles of inductive bias, data-driven methods, performance engineering, and cascaded ranking.
- [Recommender Systems](/writeups/recomendation-systems/): Reading papers with the help of DeepSeek R1 to sort out recommender systems.
- [LLM Serving](/writeups/llm-serving/): A summary of the computational characteristics of LLM serving and its optimization opportunities.
- [Idiomatic Practices in C++ Systems Engineering](/writeups/engineering-practices/): A summary of the C++ systems engineering paradigms I currently consider sound.
- [Quantization and Pruning](/writeups/quantization-and-pruning/): A summary of two model compression techniques that matter a lot for inference optimization: quantization and pruning.
- [Optimizing AI Inference](/writeups/optimizing-ai-inference/): A summary of inference optimization problems in AI engineering.
- [Paged Attention](/writeups/paged-attention/): Paged Attention: improving memory utilization and throughput when serving many concurrent requests.
- [Flash Attention](/writeups/flash-attention/): Flash Attention in a nutshell: tiling + selective gradient checkpointing.
- [CAT02: Resources](/writeups/category-theory-2---resources/): The 2nd CAT write-up is about the categorical formalism of resources, and transforming one set of resources into another. It covers monoidal preorders, wiring diagrams, monoidal monotone maps, and V-categories.
- [Diffusion Probabilistic Models](/writeups/dpm/): This post is my notes on Professor Jun Zhu's talk — "Diffusion Models for Generating High-Dimensional Data". Notably, DPM practice makes clever use of analytical solutions: both the closed form $q(x_N|x_0)$ in the forward process and the analytic-form variance estimate in the reverse process greatly improve training performance — a testament to the elegance of mathematics.
- [Revisiting Recommender Systems](/writeups/revisiting-recomender-systems/): Notes on Professor Aixin Sun's talk, "Understanding the Current State of Recommender Systems Research" — a summary of the main content with links to the papers mentioned. A glimpse into the current state of academic research on recommender systems: incisive, interesting, and startling.
- [Reflecting on a Wake-up](/writeups/reflecting-on-a-wake-up/): Notes on a moment of waking before dawn.
- [Error Handling](/writeups/error-handling/): This post discusses error handling in modern C++.
- [Linkers & Loaders](/writeups/linkers-and-loaders/): Linkers & Loaders fills a niche body of knowledge — linking and loading.
- [CAT01: Orders](/writeups/category-theory-1---orders/): The 1st CAT write-up gives an order-theoretic warm-up for the full-fledged category theory. It covers preorders, meets/joins, monotone maps and Galois connections.
- [Tech Talk: Wall is Coming](/writeups/tech-talk---wall-is-coming/): Tech Talk transcript: traces the historical roots of the memory wall problem, attempts to frame an understanding of the optimization space, derives corresponding heuristics, and lists several memory-access optimization techniques.
- [Work Reduction vs Hardware Enablement](/writeups/work-reduction-vs-hardware-enablement/): Optimization can be divided into work reduction and hardware enablement.
- [Dash: Scalable Hashing](/writeups/dash/): The main focus of the Dash paper was on the once fashionable `persistent memory`, but in reality, any `memory bandwidth`-limited scenario can benefit from it. With Intel killing off its `pmem` business, the significance of the `Dash` approach has shifted to regular `DRAM` applications.
- [DPDK is All You Need](/writeups/dpdk/): For memory-intensive datacenter applications, DPDK offers an excellent performance engineering paradigm.
- [The Little Book Review & Internalization](/writeups/the-little-book-review/): Just as DDIA can be regarded as the go-to introductory text for distributed systems, LBDL is the ideal deep learning 101.
- [eBPF Tracing for Memory-Stalled Applications](/writeups/ebpf/): An introduction to eBPF — the cutting-edge observability technology for Linux — and eBPF-based off-CPU performance analysis.
- [Artificial Intuition: Reasoning Abilities of LLMs](/writeups/on-reasoning-abilities-of-llms/): A survey of recent research on LLMs, examining the boundaries of reasoning ability in current large language model architectures.
- [Order Emerges from Self-Assembly of Dissipative Structures](/writeups/on-order/): Order emerges from the self-organization of dissipative structures.
- [Computational Consciousness](/writeups/computational-conciousness/): Metaphysics, neuroscientific theories of consciousness, and computational consciousness.
- [Efficient ANNS at Scale](/writeups/efficient-anns-at-scale/): How to perform efficient vector search over feature stores with billions or tens of billions of vectors
- [Tech Talk: Evolution of Data Center Applications](/writeups/tech-talk---evolution-of-data-center-applications/): Script of a tech talk whose theme is white-boxing algorithm theory and infrastructure, offering a subjective take on the evolution of datacenter applications, datacenter AI in particular.
- [Our Rationality can be Divided into Induction & Deduction](/writeups/induction--deduction/): Human ignorance can be divided into mysteries and problems. Human rationality can be divided into induction and deduction.
- [Distributed Computing Systems At Scale](/writeups/distributed-computing-systems/): Messy distributed phenomena and tedious engineering practice can easily obscure the essence of distributed computing systems, so here is a systematic overview.
- [Knowledge is Embeddings of Reality](/writeups/reality--knowledge/): Reality is infinitely vast and infinitely deep; once embedded into our limited cognitive space, it becomes knowledge.
- [Federated Learning](/writeups/federated-learning/): Federated Learning is a machine learning paradigm in which many mobile devices collaboratively train a model under the orchestration of a central server. Training data stays decentralized, user data is never collected, and only client model updates are uploaded to the central server, where they are aggregated into a new global model. Compared with in-center distributed training, it has unique advantages and challenges.
- [On Transparency](/writeups/on-transparency/): Transparency — the white-box index of a program — is an ideal property long neglected in the software engineering practice of the internet industry.
- [String Lookups Reduce to Parsing](/writeups/string-lookups-could-reduce-to-parsing/): String lookup and string parsing both essentially extract state from a character stream using the most compact structure and the most efficient algorithm possible. So the NFA-to-DFA algorithm from the Dragon Book can be put to good use.
- [Paradigms of Generic Programming: Archetype, Ducktype, Subtype](/writeups/paradigms-of-generic-programming/): This post summarizes the three paradigms of generic programming: Archetype, Ducktype, and Subtype. All three names end in "type" — partly because it looks cool, with a sense of regularity and the architectural beauty of logic, and partly because systems language programming is itself about building types, while generic programming is about building type specifications plus the types that conform to them.
- [On NCO](/writeups/on-nco/): Non-convex optimization, more like art
- [On ABI](/writeups/on-abi/): This post summarizes the abstraction layer that sits between two minimal contract layers — the ISA and the language standard — and isolates a great deal of complexity: the ABI of system languages.
- [A Taxonomy of Stateful Distributed Systems](/writeups/a-taxonomy-of-stateful-distributed-systems/): This post discusses the limitations of the CAP theorem and lays out a more fine-grained and precise taxonomy of stateful distributed systems, based on the trade-off between the two ideal properties of consistency and availability.
