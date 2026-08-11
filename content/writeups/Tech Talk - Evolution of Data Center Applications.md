+++
title = "Tech Talk: Evolution of Data Center Applications"
date = "2023-08-05"
tags = ["Sys", "AI", "Talk"]
description = "Script of a tech talk whose theme is white-boxing algorithm theory and infrastructure, offering a subjective take on the evolution of datacenter applications, datacenter AI in particular."
showFullContent = false
+++

Datacenter applications (databases, message queues, retrieval services, parameter servers, live-streaming services, ad-serving services, and so on) form the bulk of internet services. They sit between end-user applications and hardware infrastructure, bridging the demand side and the hardware side, while also being constrained by factors such as cost, revenue, algorithms, and law.
With recent changes — even revolutions — on both the hardware and algorithm fronts, it is worth rethinking where datacenter applications are headed: their near-term trends, potential points of innovation, and the low-hanging fruit waiting to be picked.

## The Invariant Amid Change: Datacenter Power Consumption
Global datacenter energy consumption has barely grown over the past decade, while service demand, compute, storage, and total data transmission have grown by orders of magnitude over the same period. As practitioners, we are no strangers to the explosive growth of the internet, so the fact that global datacenter energy consumption rose only 6% from 2010 to 2018 is mildly counterintuitive — from an energy perspective, this is almost a stagnant industry.

Overall progress in CMOS process technology and in hardware/software techniques can explain the improvement in server energy efficiency, but the root cause of stagnant energy consumption lies elsewhere: the essence of internet companies' profitability is levying a service tax on the one billion people abroad and one billion at home who can afford to pay. Individual successful startups can expand rapidly, but the internet industry as a whole sees its spending growth on datacenter energy bounded by the income growth of ordinary people worldwide.

![DatacenterPower](https://jipeng4974.github.io/img/DatacenterPower.jpeg)

Once a startup has scaled up to internet scale, the growth rate of its machine resources should abruptly shift from explosive expansion to stagnation, entering a phase of natural turnover and replacement.
From this angle one can conclude that, once scale expansion is over, datacenter applications should likewise shift from an extensive, energy-hungry model to an intensive, fine-grained one: fully unleashing hardware potential through hardware-software co-design, and reducing total computation through algorithmic innovation.

## Algorithmic Iteration: From Deductive Reasoning to Inductive Inference
The trend on the algorithm side is "transformers getting even more attention."

Computer theory originated in symbolic logic. In the 1930s, Turing, Gödel, and Church independently proposed the Turing machine, general recursive functions, and lambda calculus, along with their respective notions of computability. The halting problem for Turing machines and Gödel's incompleteness theorems refuted the feasibility of the Hilbert program of the 1920s, proving that formal systems are incomplete: no finite set of axioms and rules can ever derive all truths.

Just as human ignorance can be divided into problems (computable) and mysteries (uncomputable), human rationality can be divided into deductive reasoning and inductive inference — the former solves problems, the latter analyzes mysteries.

Computer programs are built on the Turing machine model and are therefore inherently formal systems well suited to deductive reasoning. To this day, we still implement most datacenter applications with formal methods such as rule systems (hand-written conditional control), signal processing (e.g., audio feature extraction), and state machines — think of the various databases, retrieval systems, parameter servers, and game AI.

With the arrival of the AI era, we use more weights (as axioms) and deep neural networks composed of large numbers of operators (as rules) — still formal systems from an implementation standpoint — to simulate, or rather approximate, "inductive inference." (The purely Bayesian formalization is Solomonoff induction, a method that by itself never halts — the longer you run it, the less it wants to stop — so it can only be approximated, never implemented.) However crude and inefficient this approximation may be, it ultimately enhances the expressive power of computer languages and integrated circuits, completing the leap from reasoning to inference.

Once datacenter applications gain inductive inference capability, they acquire the ability to understand complex reality (pin down reality) — real-world signals such as images, corpora, and audio are high-dimensional as a whole, yet locally often contain structures of high data refinement, where refinement can be formally defined via Shannon entropy or Kolmogorov complexity. Hence a crop of knowledge-based AI systems has emerged, such as recommender systems, ad targeting, cover-song recognition, and chatbots. What they share is mapping reality onto low-dimensional representation subspaces, where the vectors — or embeddings — can effectively characterize reality from some angle. Transformers go a step further: they amount to mapping reality onto multiple low-dimensional subspaces with low mutual coherence (see https://arxiv.org/abs/2306.01129), thereby understanding reality from multiple angles and levels, and measuring the compactness and discriminative power of features by the difference between the coding rate of the representation space and the sum of the coding rates of the subspaces.

In games and game theory domains, reinforcement learning has also achieved what rule systems cannot, defeating professional players in StarCraft II, Dota 2 (fixed-lineup 5v5, and solo), and Go.

Moreover, inductive inference has one more important trait compared to deductive reasoning: it better matches the thinking habits of the human brain (or, generalizing, the brain of any Earth creature — a bee's 3D path-finding and obstacle avoidance, a mosquito's blood-vessel localization, are all based on natural Bayesian inductive inference; only well-trained humans can perform slow deductive reasoning). As a result, the traditions of human society, the uncertainty of human language, and human preferences are hard to describe with formal methods, yet can be described in probabilistic language and predicted by inductive inference.

The next topic in algorithmic progress is models that possess both deductive reasoning and inductive inference. Existing transformers do show the ability to fit simple multi-step arithmetic, but what they fit is a step-count-dependent, hyper-complex nonlinear system rather than the simple arithmetic rule, so accuracy typically plateaus around eighty percent and never goes higher. For a formal system, the number of reasoning steps does not change the total number of axioms and rules; it only slightly increases the amount of computation. But for an inductive-inference model like an LLM, once the number of reasoning steps grows and the computation becomes more complex, the LLM's symbolic-logic ability degrades rapidly — on dynamic programming problems whose exact execution path never appeared in the training data, it can drop straight to zero. Theoretically, an infinite-precision transformer is Turing-complete (see https://arxiv.org/abs/1901.03429), but anything short of infinite precision is not. In practice, the precision of model weights and computation is very low even by ordinary computing standards, the model architecture is not designed for multi-step logical reasoning, and deep learning itself is fitting an approximate function — these three factors combined leave current LLMs' deductive reasoning ability unsatisfactory.

From the perspective of algorithmic iteration, the development trend for datacenter applications is: while guaranteeing "computability" (in the literal sense, not the formal definition), pursue the two ideal properties that deductive methods cannot achieve — "humanization" (align to humanity) and "completeness" (pin down reality). Promising directions include intelligent content creation, personalized recommendation, smart agents (assistants, game NPCs, robots, autonomous driving), complex-system prediction (weather forecasting, trading systems), and large-model training (model parallelism, high-performance networks, C2C interconnect, optical I/O, heterogeneous computing).

Problems that rule systems already solve efficiently should not be force-fitted with inductive-inference models — that only adds cost and error rate. Examples: music fingerprinting, relational databases, image rotation and warping, lossless compression. Even in some seemingly "mysterious" areas, such as OOD text classification, white-box representation-learning research plus information theory can yield direct mathematical solutions. A recent study ("Low-Resource" Text Classification: A Parameter-Free Classification Method with Compressors) beat a 10-billion-parameter BERT with 14 lines of code (though its experimental code was later found to be buggy, so the actual results were not that good). The paper proposes using lossless compressors like gzip to approximate the Kolmogorov complexity of text data, and running KNN over information distance (similar in principle to the rate reduction proposed by Yi Ma) — very effective, and similar attempts could be made in the audio domain.

## Datacenter Hardware Infrastructure
First, a brief introduction to common datacenter hardware infrastructure:
- The older machines are Cascade Lake, 14nm, 24 cores per die — 24 cores per chip, 48 physical cores with two chips, i.e., what we usually call a 96-core machine. Other configurations exist too, up to 28 cores per die, 112 logical cores in total.
- The newer Ice Lake, on a 10nm process, usually has 32 cores, corresponding to 128 logical cores; the most expensive configuration goes up to 40 cores. It is still based on a monolithic die, with a mesh bus putting 40 cores together on one huge die.
- The latest 7nm Sapphire Rapids, and AMD Genoa which targets it, have both entered mass production this year.
- As for NICs, the commonly used ones are Mellanox 25G CX4/CX5 cards, and there are also quite a few 100G dual-port CX6 cards, all RDMA-capable; even without RDMA they deliver quite good high-speed Ethernet performance. For virtualization use cases, such as AWS or Alibaba Cloud, the SmartNIC capabilities of these cards can also be leveraged to offload virtualization overhead.

Compared to the previous Lakes, Sapphire Rapids changes a great deal. The monolithic-die route had indeed reached its limit, and Sapphire Rapids has entered the multi-die era: the chip is divided into 4 dies, which can also be seen as an evolution toward chiplets.

Sapphire Rapids supports up to 8 sockets, with up to 60 cores per chip. For IO technology it supports CXL 1.1, PCIe 5.0, UPI 2.0, and HBM2e (optional).

![Xeon](https://jipeng4974.github.io/img/4th_xeon.jpeg)

AMD's fourth-generation EPYC flagship, Genoa, also entered mass production in Q1 this year, basically going head-to-head with Sapphire Rapids on a 5nm process. Although AVX512 is criticized by many and its practical performance is mediocre, Zen4 supports it anyway — after all, the target customers for Xeon and EPYC are datacenter applications with growing audio/video processing and AI workloads. Genoa has 8 cores per die/chiplet, with 12 CPU cluster dies arranged around a central IO die — 96 cores per chip, a textbook chiplet-style design.

## Iteration in Chip Design: Chipletization
A notable recent change in chip design is the Chiplets + SiP (System in Package) paradigm replacing large-die SoC + PCB co-packaging.
Chiplets have drawn attention from both industry and academia, called "what's next in computing" by IBM Research. The discussions of compute, IO, and memory technologies in later sections all involve chiplets and co-packaging, so we first discuss iteration at the chip level.

Chiplet partitioning means splitting a circuit into modular subsystems, each an independent die — a chiplet — and packaging multiple chiplets into a single chip (package) using 2.5D/3D technology.

![Chiplet](https://jipeng4974.github.io/img/chiplet.png)

The advantages of the chiplet-reuse paradigm over traditional IP-reuse (in the chip context, IP refers to a circuit module with independent functionality and a mature design) are as follows:
1. Advanced CMOS processes (below 7nm) are unlikely to achieve high yield on large dies for technical reasons; the smaller the die size, the lower the cost.
2. On advanced CMOS processes, analog IP such as power management and high-speed IO SerDes cannot be scaled down at the same rate, so advanced CMOS is generally used only for processors and accelerators.
3. It enables modular design, letting designers focus on the extreme optimization of a single module and choose the most suitable technology: e.g., advanced processes for CPU and GPU, mature processes for analog modules, DRAM for high-bandwidth memory (HBM), and non-volatile memory for AI accelerators.
4. It enables heterogeneous integration at the chip/package level: general-purpose CPUs, optimized GPUs, embedded FPGAs, dedicated machine-learning circuits, optical IO modules, high-bandwidth memory, and other modules can be assembled like Lego bricks into a complete system, using advanced 3D packaging schemes based on through-silicon vias (TSV), micro-bumps, or even die-to-wafer hybrid bonding.

The era of heterogeneous computing we once envisioned — with all kinds of DSAs, AI chips, and FPGAs flourishing — did not arrive as expected; it was crushed by NVIDIA's GPU solution combining hardware and software with integrated compute and interconnect, leaving almost only TPU still iterating toward v5.

But future server chips themselves have both the possibility and the tendency toward diverse, heterogeneous co-packaged integration. Magical creatures like CPO (co-packaged optics) and HBM (high-bandwidth memory) get to move in thanks to chipletization.

More functionality also means higher programmability. Some functionality can even bring revolutionary change: the ultra-high communication bandwidth of optical IO, the ultra-high memory bandwidth of HBM, high-performance off-package interconnect (NVLink-C2C), and mesh interconnect among multiple chiplets (NVSwitch) — now used by NVIDIA to build the H100 and by Google to build TPU v4 — may in the future overturn the host-centric design paradigm of datacenter applications and usher in a new computing architecture of hardware resource disaggregation: operating systems adapted to resource disaggregation (LegoOS was one attempt based on early IB networks), system-language ABIs, new high-level languages, and new forms of network IO, storage, and computation may all hatch from it.

## Iteration in Compute and Memory Hierarchy: Scalable Many-Core NUMA Architectures 
In the commercial server space, part of the chiplet paradigm's vision has already been realized. AMD, for example, adopted chiplets very early and partially solved the inter-chiplet IO problem, achieving a scalable many-core NUMA architecture. Pre-SPR Xeon physical machines are also NUMA, though with only 2 NUMA nodes (Intel's NUMA nodes are currently too large, so they can't quite be called chiplets).

The μArch has a direct impact on the performance engineering of compute- and memory-intensive datacenter applications. The figure below shows a 96-core concept machine with 6 chiplets in one package. Clearly, once we open the black box of an integrated circuit, we see finer-grained components and the network they form (Network-on-Chip). This concept machine integrates various advanced designs: not only many cores, but also full cache coherency. Compared to past multi-core architectures, the memory hierarchy of many-core architectures has grown correspondingly deeper, and the cost of a cache miss has become higher — to the point that Rust's standard library implements maps with B-trees (whereas in C++ they are famously red-black trees). This is the result of the widening gap between processor and memory frequencies: (to exaggerate a bit) memory today is as slow as disks were back then.

![IntAct](https://jipeng4974.github.io/img/IntAct.png)

For NUMA architectures, the Linux kernel and KVM at the system layer have NUMA-aware schedulers; at the application layer, the networking framework Seastar, the database ScyllaDB, the in-memory database DragonFly, and others have all noticed that being aware of hardware topology greatly improves overall performance (ScyllaDB and DragonFly outperform their counterparts Cassandra and Redis by several times respectively), and have proposed share-nothing high-performance architectures: avoid locks and unnecessary shared memory, avoid unnecessary remote memory access, avoid unnecessary cross-die communication, design cache-friendly data structures, and make better use of the L1 cache local to each die — considering that the Cascade Lake machines we currently use are not fully cache coherent, and that even when full cache coherence is achieved in the future, the coherency mechanism for shared caches will almost certainly carry overhead. In short, in the era of complex topologies and deep memory hierarchies, beware of cache misses.

![NUMA](https://jipeng4974.github.io/img/NUMA.png)

## Hitting the I/O Bottleneck Again: Advanced Interconnect Once Again Central to HPC
Datacenter applications account for 76% of global IO traffic. Like computation, IO consumes power — and like computation, datacenter IO power consumption has also stayed flat for a decade, offset by hardware progress. Also like computation, interconnect is layered. Recently at the die-to-die (on-package) link layer there is the UCIe standard; at the off-package layer there is CXL 3.0 based on PCIe 6.0 and the 900GB/s NVLink-C2C; at the inter-node layer there is InfiniBand NDR. These are electrical interconnects; optical interconnects are more promising by comparison, but also harder, and still in early R&D.

The big-data era and the pre-LLM AI era had low IO demands; standard Ethernet sufficed for most datacenter applications, including parameter servers. Large-model training created new forms of computation and IO. Once models no longer fit in memory and model parallelism became unavoidable, IO became the bottleneck again: each of the H100's 8 GPUs needs 7.2Tbps of off-package bandwidth — for comparison, even a ToR switch only needs 10+Tbps. The bandwidth demand of AI-specific GPUs in large-model training scenarios is already very close to that of switches (switches, like GPUs, are giant ASICs and likewise a domain where co-packaged optics applies). In the switch domain, Google has already developed a practical, clearly beneficial all-optical link switch. For GPU interconnect, NVIDIA has also proposed a concept system of optically interconnected GPUs, even designing corresponding GPU racks with external laser sources and sparse cabling that happens to solve the cooling problem.

![optics](https://jipeng4974.github.io/img/optics.png)

Advanced IO technology is inseparable from the development of HPC (high-performance computing). Although HPC — or supercomputing — is always associated in the public imagination with extremely powerful processors and accelerators, the reality is actually the opposite: traditional HPC workloads (modeling- and simulation-type scientific computing) typically run on ordinary commercial nodes, while the interconnect must use high-performance HPC interconnect technology. The heterogeneity of traditional supercomputers lies in IO technology, not in the application of FPGAs or dedicated ASICs.

Later, in the era of big-data analytics and AI, standard Ethernet sufficed for AI training workloads matched to the model sizes of the time. Mainstream internet big-data applications could be implemented entirely on commercial IO technology and commercial compute nodes, while the heterogeneity of the few AI datacenters was mainly reflected in accelerator technology (GPUs, TPUs, dedicated AI chips) rather than IO.

Now heterogeneous workloads dominated by large-model training have emerged. Exploding model sizes outgrow memory, model parallelism becomes unavoidable, and die-to-die bandwidth, off-package bandwidth, and inter-node bandwidth once again become bottlenecks. Advanced (heterogeneous) interconnect technology is once again a core HPC topic.

Datacenter AI's return to a heterogeneous IO + heterogeneous compute architecture is essentially supercomputer-ization, so it happens to also suit traditional modeling-and-simulation HPC workloads. This actually gives the internet industry a new opportunity: while out-competing each other in large-model training, companies can conveniently enter the supercomputing industry, providing cheap, reliable, easy-to-use, always-on-call scientific computing capability to universities and research institutions — to some extent reversing, in public opinion, the negative image of internet companies contributing little to society, and seeking legitimacy for continuing to levy the internet service tax.

## Iteration in I/O Technology: Optical I/O Moves Ever Closer to Compute Endpoints
Advanced copper interconnect is the present; co-packaged optical interconnect is the future.

The co-packaging mentioned above is the key technology for advanced interconnect: on one hand, co-packaging multiple dies itself shortens IO links and reduces IO energy consumption; on the other hand, it enables the integration of co-packaged optics (CPO) modules.

One evolutionary trend of datacenter IO is "bring fiber closer to endpoints." Compared to electrical links, an obvious advantage of optical links is longer transmission distance (limited by frequency-dependent attenuation). Another advantage is that as bandwidth increases, electrical signals keep getting shorter and noise keeps growing; IB networks are already approaching the limits of copper, and further development can only go from copper to fiber. Moreover, at high frequencies, electrical interconnects and connectors both receive and transmit, experiencing significant crosstalk, which also limits the packaging density of electrical interconnect. As a signal transmission medium, fiber is nearly ideal; the only inefficiency lies in the electro-optical conversion at the two ends.

Today the mainstream in-rack connection scheme is copper cable, while inter-rack switching is based on Ethernet links. In hyperscale datacenters, cable runs reach several kilometers, so optical cable is used more and more — even short links increasingly use fiber now. In datacenters, fiber is getting closer and closer to the endpoints, closer and closer to CPUs and GPUs; the latest trend is integrating optical components directly onto the silicon die. CPO combines electrical and optical links, eliminating the intervening receive-and-retransmit process and skipping the optoelectronic conversion step. The first generation of CPO is pluggable optics, the second is On-Board Optics / Near-Package Optics, the third is 2.5D CPO, the fourth is 3D CPO, and the fifth is Integrated Laser.

The biggest innovation of Google's TPU v4 supercomputer is its all-optical link optical circuit switch (OCS), reconfigurable across 4k nodes, which saves the energy of optoelectronic conversion. InfiniBand pushed high-performance copper interconnect to its extreme, and its subsequent roadmap also goes from copper to optoelectronic co-packaging. Although NVIDIA has been pushing electrical-link solutions (NVLink), it has also signed an R&D partnership with Ayar Labs and begun supporting research into external laser sources and silicon-photonics interconnect — after all, NVLink is essentially still NUMA: it can scale to 8 GPUs, 16 GPUs, but there's no way to connect the datacenter-scale ten thousand GPUs. HP also partnered with Ayar Labs last year, trying to bring silicon photonics into its advanced HPC IO product, the Slingshot interconnect. Intel is likewise researching integration schemes that embed lasers inside the chip.

The figure below lists the power consumption, cost, density, and transmission distance of interposer, PCB, CPO, copper cable, and active optical cable. CPO's advantages are obvious.

![CPO](https://jipeng4974.github.io/img/CPO.png)

At the current state of the art, CPO is viewed merely as an electrical-optical (E/O) bridge to solve the interconnect bandwidth-density bottleneck of SiP. For application scenarios like distributed training, an E/O bridge — or rather, bringing fiber closer to endpoints — can already greatly reduce energy consumption and improve performance. But CPO's potential goes far beyond this. If you add just a bit of functionality to a CPO chiplet, it can offload some CPU work like a coprocessor or SmartNIC — for example, simple data pre-processing and post-processing; or, CPO can access HBM directly without going through the CPU, thereby providing DMA capability. This is very helpful for disaggregated architectures: no physical pooling is required, and it is faster than copper IB networks.

This means optical IO can not only solve the bandwidth problem brought by large-model training, but also makes it possible for datacenter applications to transition from a host-centric to a disaggregated architecture.

What is the disaggregation paradigm? Opposite to the traditional server-centric paradigm, disaggregation is a datacenter application architecture design paradigm that breaks the monolithic server apart into independent hardware resources — CPU, DRAM, disk, accelerators — for resource abstraction and management. Hardware disaggregation is not a new concept. The 2018 USENIX OSDI best paper, LegoOS, opened with "We believe that datacenters should break monolithic servers" — a line full of conviction. Back then, InfiniBand had not yet evolved to the NDR generation, and optical I/O was still far from datacenter-internal endpoints, yet it was already enough to support such a grand narrative.

With high-performance networks, disaggregated architectures can effectively improve the resource utilization of datacenter applications and alleviate the over-provisioning of CPU, accelerator, memory, disk, and other resources that is unavoidable under the host-centric paradigm.

## Evaluating the GH200 Grace Hopper Superchip 
NVIDIA claims the Grace Hopper Superchip is the world's first heterogeneous accelerated platform truly supporting HPC and AI workloads.
![GraceHopper](https://jipeng4974.github.io/img/GraceHopper.png)
As shown in the figure below, this superchip is an integration scheme that co-packages a Grace Arm Neoverse CPU + LPDDR5x memory and an H100 Tensor Core GPU + HBM onto a PCB via NVLink-C2C.
![GraceHopper](https://jipeng4974.github.io/img/GraceHopper2.png)
![GraceHopper](https://jipeng4974.github.io/img/GraceHopper3.png)

This is not an innovative solution. On one hand it runs against the chipletization trend (apart from HBM, which counts as a chiplet, the H100, Grace, and NVSwitch are all giant SoCs/ASICs — and even HBM here is PCB co-packaged, not SiP); on the other hand, it makes no exploration or attempt at CPU-GPU hyper-convergence (physical fusion into a single SoC, logical unification of page-table management, memory, cache, and concurrency models). (GPUs were designed from the start with too many choices incompatible with CPUs — cache model, memory model, concurrency model — and with CUDA's foundation now laid, it's hard to turn back.) It simply and crudely integrates the CPU, high-bandwidth memory, and H100 via PCB co-packaging, using NVLink-C2C to provide memory coherence and higher off-package bandwidth (without trying any advanced IO technology). On the software side it also fails to provide stronger programmability on top of CUDA, offering only coherent memory access; the programming model remains fully heterogeneous (also because CUDA was born as a graphics acceleration library and could not have anticipated future demand for a homogeneous programming model for such superchips).

But it is a low-risk, high-execution integration scheme. As Zuckerberg put it, "Move fast and break nothing": co-package the already-excellent components as-is, make no invasive modifications, and as long as you move fast enough, you can quickly capture the market, build the ecosystem, and sustain a premium. There are more perfect memory-coherence solutions on the market (e.g., AMD MI300X), better CPU-GPU hyper-convergence solutions, and AI chips more efficient than GPUs that have to compromise for graphics workloads — but none of them has the CUDA heterogeneous programming system, nor a complete solution like Grace Hopper that resolves compute, memory, and IO bottlenecks reasonably well all at once.

In short, NVIDIA's solution is the best-of-breed emerging from the interaction of an ecosystem (GPU + CUDA) and an organism (ChatGPT's training recipes tailored to the A100), but it is far from the ideal optimum and not even on the right technical route. AMD's so-called APUs and domestic AI DSAs (such as Biren) still have hope of overtaking on the curve.

New opportunities in computing systems
- Full-stack optimization from (application) end to (hardware) end, a.k.a. hardware-software co-design.
  - TVM: deep learning compiler stack for cpu, gpu and specialized accelerators
  - GPU + CUDA
  - GH200 Grace Hopper + new CUDA NUMA memory APIs + heterogeneous programming APIs
  - Our company's LavaRecord end-to-end optimization project, which interfaces downward (LavaUOS) with new storage hardware, trying to build an efficient user-space IO software stack on NVMe SSDs.
- Applying machine learning to auto-tune systems with large parameter spaces.
  - Parameter tuning for storage engines like rocksdb
  - AutoTVM for deep learning models on heterogeneous hardware
- Resource-disaggregation architecture design enabled by advanced interconnect technologies.
  - LegoOS
  - PolarDB-X's storage-compute separation and memory pooling
- Share-nothing architectures on compute nodes, and data-oriented design.
  - At the application-framework level there are precedents like libtorque, DragonFly, Seastar, and ScyllaDB, mostly for IO-intensive applications — though in fact most CPU applications with large memory footprints can be regarded as IO-intensive, because once cache misses rise, memory access tends to account for far more than computation.
  - In virtualization, SJTU's IPADS lab published CPS: A Cooperative Para-virtualized Scheduling Framework for Manycore Machines, proposing a cooperative para-virtualized scheduling mechanism that greatly improves the scalability of many-core virtual machines.
- Based on white-box research of deep models and existing mathematical tools, replace black-box model approximations with direct math solutions.
  - E.g., "Low-Resource" Text Classification: A Parameter-Free Classification Method with Compressors, with its neat solution of compression + information distance + KNN.
  - Measuring embedding quality by inter-class incoherence and intra-class compressibility (sparsity), without resorting to indirect measurement via some end-to-end application metric.
  

## References and Further Reading
- Learning One-hidden-layer Neural Networks with Landscape Design: even for the simplest non-convex optimization scenario in deep learning, explanation with mathematical tools (mathematical optimization methods) is extremely difficult.
- Functionality and performance of NVLink with IBM POWER9 processors: IBM Power9 (the US Department of Energy's Summit and Sierra supercomputers) was already using NVLink years ago, with a very mature hardware cache-coherence design (plus hardware atomic ops and address translation) — more mature than the Grace Hopper solution.
- Faith and Fate: Limits of Transformers on Compositionality: large language models exhibit emergent deductive-logic ability but perform poorly on multi-step compositional problems, and their accuracy drops even faster on dynamic programming problems where the same computation path in the computation graph never appeared in training samples. Compared with other empirical studies, this one is more rigorous and comprehensive, considering the effect of splits unseen during training in the computation graph. We have reason to believe that the emergent deductive reasoning of LLMs will be constrained by the inherent limits of transformers.
- Teaching Arithmetic to Small Transformers: small transformer-based language models suffice to learn simple arithmetic; providing training data containing correct computation steps (chain-of-thought style data) is the key to improving arithmetic learning — training crudely on problems and answers alone cannot raise accuracy by merely increasing model size.
- A Survey of Large Language Models: an up-to-date review of large language models.
- Variational Inference: A Review for Statisticians: a statistician's perspective for explaining and understanding VI, discussing the special case of VI applied to exponential-family models, giving an example of a Bayesian Gaussian mixture model, and deriving a VI variant that uses stochastic optimization to scale to massive data.
- Training language models to follow instructions with human feedback: OpenAI's experience report, focused on RLHF.
- GPT-4 Architecture, Infrastructure, Training Dataset, Costs, Vision, MoE: a leak from SemiAnalysis, quite credible.
- Efficiently Scale LLM Training Across a Large GPU Cluster with Alpa and Ray: LLM training.
- Scaling Language Model Training to a Trillion Parameters Using Megatron: Megatron (repo: https://github.com/NVIDIA/Megatron-LM , paper: https://arxiv.org/pdf/1909.08053.pdf )
- https://www.youtube.com/watch?v=eqWPyaRcILQ — a talk on Co-Packaged Optics by Ram Huggahalli of Microsoft Azure's hardware systems and infrastructure team.
- https://www.youtube.com/watch?v=Xt-GY8Pkt6g — a talk on Co-Packaged Optics and the Evolution of IO by Tony Chan Carusone, who researches optical communication and advanced interconnect technologies.
- Next-generation Co-Packaged Optics for Future Disaggregated AI Systems: insights into co-packaged optics and future disaggregated AI systems.
- TPU v4: An Optically Reconfigurable Supercomputer for Machine Learning with Hardware Support for Embeddings: Google's TPU v4, with emphasis on the all-optical link switch.
- LegoOS: A Disseminated, Distributed OS for Hardware Resource Disaggregation: LegoOS, an attempt at hardware resource disaggregation based on early InfiniBand high-speed networks.
- https://www.hpcwire.com/2020/11/16/nvidia-mellanox-debuts-ndr-400-gigabit-infiniband-at-sc20/ — Mellanox (NVIDIA)'s InfiniBand NDR generation, and its roadmap.
- Rack-scale disaggregated cloud data centers: The dReDBox project vision: an early attempt at disaggregated datacenter architectures. 
- White-Box Transformers via Sparse Rate Reduction: the white-box explanation of transformers by Yi Ma's team; Yi Ma had earlier given the more general rate reduction principle: Learning Diverse and Discriminative Representations via the Principle of Maximal Coding Rate Reduction.
- https://github.com/bgavran/Category_Theory_Machine_Learning — a category-theory account of deep learning. For deep-learning interpretability and reverse engineering, also see Christopher Olah's blog: colah.github.io. Olah has many profound insights, e.g. https://colah.github.io/posts/2014-03-NN-Manifolds-Topology/ — visualization of the manifold hypothesis and an explanation of deep-learning classification.
- IntAct: A 96-Core Processor With Six Chiplets 3D-Stacked on an Active Interposer With Distributed Interconnects and Integrated Power Management: a paper in advanced IC design, presenting a 96-core many-core prototype system integrating advanced concepts such as the chiplet paradigm, 3D packaging, and full cache coherence.
- "Low-Resource" Text Classification: A Parameter-Free Classification Method with Compressors: lossless compression to approximate Kolmogorov complexity, then computing information distance (similar to rate reduction — it characterizes the information gap between the whole and the classes: if a class has low coding length while the whole has high coding length, the classification has strong inter-class discrimination and intra-class compressibility), and simple KNN over information distance completes the classification. This study's code has bugs and does not actually beat BERT; see https://kenschutte.com/gzip-knn-paper/.
- M. Li and P.M.B. Vitányi, An Introduction to Kolmogorov Complexity and Its Applications: an introduction to Kolmogorov complexity and its applications.
- A Mathematical Theory of Communication: Shannon's original 1948 information-theory paper.
- hwloc doc: documentation for hwloc, a NUMA-discovery + cpu/memory-binding library.
- https://man7.org/linux/man-pages/man2/mbind.2.html: libnuma's NUMA memory policy functions.
- On the Turing Completeness of Modern Neural Network Architectures: proves that infinite-precision transformers are Turing-complete — i.e., any Turing machine can be simulated by an infinite-precision transformer — but any fixed precision is not Turing-complete.
