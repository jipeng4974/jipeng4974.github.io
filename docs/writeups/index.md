# Writeups

> Long-form notes on systems, AI, math, and everything in between

---

LLMS index: [llms.txt](/llms.txt)

---

Section pages:

- [Music JEPA 和三体宇宙](/writeups/music_jepa_universe/): 生存维数，维度坍塌，死维复活，宇宙规律，田园宇宙，幸存子空间，威慑项λ。
- [Music JEPA Regularizers](/writeups/music_jepa_reg/): 再试Music LeJEPA之抗坍缩正则
- [DSpark](/writeups/dspark/): DSpark：低秩bigram赎回块内依赖，置信度head给验证长度做admission control。
- [自由意志的可度量指标](/writeups/free_will/): 由己度 = 内敏性 - 外敏性。
- [Kimi K3: Scaling LLMs Across Sequence, Depth, and Width](/writeups/llm_arch/): LLM架构演化，从dense transformer到Kimi K3。
- [Music LeJEPA](/writeups/music_lejepa/): 初试Music LeJEPA（未完待续）
- [镜头与编码器的同构性](/writeups/lens_encoder/): 相机镜头和编码器呈现惊人的同构性。
- [超越表征学习：预测驱动的编码器+可学习的传感器](/writeups/beyond_representation_learning/): 现有的多模态表征学习原始、孤立且残缺，还需要预测驱动的编码器+可学习的传感器。
- [Cognitive Phase Transitions](/writeups/cognitive-phase-transitions/): 人类智能价值穿越周期的不变量，是认知相变能力。
- [Semantic Depth & Music Re-ID](/writeups/reid/): 本文提出语义深度指标，从归纳偏置、数据驱动、性能工程、级联排序等角度讨论Music Re-ID。
- [Recommender Systems](/writeups/recomendation-systems/): 借助DeepSeek R1读论文，梳理推荐系统。
- [LLM Serving](/writeups/llm-serving/): 本文总结LLM serving的计算形态和优化机会。
- [Idiomatic Practices in C++ Systems Engineering](/writeups/engineering-practices/): 本文总结当下我认为比较好的C++系统工程范式。
- [Quantization and Pruning](/writeups/quantization-and-pruning/): 总结推理优化问题中相当重要的模型压缩技术——量化和剪枝。
- [Optimizing AI Inference](/writeups/optimizing-ai-inference/): 总结AI工程中的推理优化问题。
- [Paged Attention](/writeups/paged-attention/): Paged Attention：提升同时处理多请求的显存利用率和吞吐。
- [Flash Attention](/writeups/flash-attention/): Flash Attention，一言以蔽之：tiling + selective gradient checkpointing。
- [CAT02: Resources](/writeups/category-theory-2---resources/): The 2nd CAT write-up is about the categorical formalism of resources, and transforming one set of resources into another. It covers monoidal preorders, wiring diagrams, monoidal monotone maps, and V-categories.
- [Diffusion Probabilistic Models](/writeups/dpm/): 本文是对朱军教授的分享——“用于生成高维数据的扩散模型”的笔记。值得注意的是DPM实践中巧妙使用了解析解，无论是前向过程的closed form $q(x_N|x_0)$，还是逆向过程中解析形式的方差估计，都大大提升了训练性能，体现了数学的精妙。
- [Revisiting Recommender Systems](/writeups/revisiting-recomender-systems/): 本文是对孙爱欣教授的分享——“推荐系统研究现状的理解”的笔记，对大致内容进行了摘要，并收集了提及文献的链接——可以一窥推荐系统领域学术研究的现状，精辟，有趣，令人震撼。
- [Reflecting on a Wake-up](/writeups/reflecting-on-a-wake-up/): 记黎明前的一次醒来。
- [Error Handling](/writeups/error-handling/): 本文讨论现代C++的错误处理问题。
- [Linkers & Loaders](/writeups/linkers-and-loaders/): 《Linkers & Loaders》填补了一个niche知识域——链接和加载。
- [CAT01: Orders](/writeups/category-theory-1---orders/): The 1st CAT write-up gives an order-theoretic warm-up for the full-fledged category theory. It covers preorders, meets/joins, monotone maps and Galois connections.
- [Tech Talk: Wall is Coming](/writeups/tech-talk---wall-is-coming/): Tech Talk文稿，梳理内存墙问题的历史渊源，尝试给出对优化空间的理解，推导出相应的启发性策略，并列举一些访存优化技术。
- [Work Reduction vs Hardware Enablement](/writeups/work-reduction-vs-hardware-enablement/): Optimization can be divided into work reduction and hardware enablement.
- [Dash: Scalable Hashing](/writeups/dash/): The main focus of the Dash paper was on the once fashionable `persistent memory`, but in reality, any `memory bandwidth`-limited scenario can benefit from it. With Intel killing off its `pmem` business, the significance of the `Dash` approach has shifted to regular `DRAM` applications.
- [DPDK is All You Need](/writeups/dpdk/): 对于访存密集的数据中心应用来说，DPDK提供了非常好的性能工程范式。
- [The Little Book Review & Internalization](/writeups/the-little-book-review/): 正如DDIA可被视为分布式系统方向的入门教程，LBDL是理想的深度学习101。
- [eBPF Tracing for Memory-Stalled Applications](/writeups/ebpf/): 介绍eBPF——前沿的Linux系统的可观测性技术，以及基于eBPF的off-CPU性能分析。
- [Artificial Intuition: Reasoning Abilities of LLMs](/writeups/on-reasoning-abilities-of-llms/): 对近期对LLM的调研文献进行梳理总结，讨论目前大语言模型架构的推理能力边界。
- [Order Emerges from Self-Assembly of Dissipative Structures](/writeups/on-order/): 秩序，涌现自耗散结构的自组织。
- [Computational Consciousness](/writeups/computational-conciousness/): 形而上学、神经科学意识论和计算意识。
- [Efficient ANNS at Scale](/writeups/efficient-anns-at-scale/): 如何在十亿、百亿级特征库上做高效的向量检索
- [Tech Talk: Evolution of Data Center Applications](/writeups/tech-talk---evolution-of-data-center-applications/): Tech Talk文稿，主旨是白盒化算法理论和基础设施，给出一个对Datacenter应用，尤其是Datacenter AI演进趋势的主观理解。
- [Our Rationality can be Divided into Induction & Deduction](/writeups/induction--deduction/): 人的无知，可分为神秘和问题。人的理性，也可分为归纳和演绎。
- [Distributed Computing Systems At Scale](/writeups/distributed-computing-systems/): 纷乱的分布式现象、繁琐的工程实践容易遮蔽对分布式计算系统本质的理解，遂做梳理。
- [Knowledge is Embeddings of Reality](/writeups/reality--knowledge/): 现实无限广博，无限深邃，将其嵌入我们有限的认知空间后，就形成了知识。
- [Federated Learning](/writeups/federated-learning/): 联邦学习(Federated Learning)是指许多移动设备在一个中央服务器的编排下协作训练模型，保持训练数据离散，避免对用户数据进行收集，仅将客户端模型更新上传中央服务器汇总成新的全局模型的机器学习模式。与in-center的分布式训练相比，有其独特的优势和挑战。
- [On Transparency](/writeups/on-transparency/): 透明度，或者说程序的白盒指数，是互联网软件工程实践中长期被忽略的一个理想属性。
- [String Lookups Reduce to Parsing](/writeups/string-lookups-could-reduce-to-parsing/): 字符串查找和字符串解析，本质都用尽可能紧凑的结构和高效的算法，从字符流中抽取状态。因此龙书中的NFA转DFA算法可以派上用场。
- [Paradigms of Generic Programming: Archetype, Ducktype, Subtype](/writeups/paradigms-of-generic-programming/): 本文总结泛型编程的三种范式：Archetype, Ducktype, Subtype。三者都以type结尾，一方面是因为这样比较帅，有规则感和逻辑上的建筑美，另一方面是因为系统语言编程本身就是在打造一个个类型，而泛型编程就是在打造一个个类型规约+遵循规约的类型。
- [On NCO](/writeups/on-nco/): 非凸优化(non-convex optimization)，more like art
- [On ABI](/writeups/on-abi/): 本文总结了介于ISA和语言标准这两个简约协议层之间隔离了大量复杂度的抽象层次——系统语言的ABI。
- [A Taxonomy of Stateful Distributed Systems](/writeups/a-taxonomy-of-stateful-distributed-systems/): 本文讨论了CAP Theorem的局限性，梳理了基于一致性、可用性这两个理想属性间的权衡的更细致精确的有状态分布式系统分类学。
