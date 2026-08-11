
+++
title = "工作减少 vs 硬件赋能"
date = "2024-02-08"
tags = ["sys", "en", "perf"]
description = "优化可以分为两类：工作减少与硬件赋能。"
showFullContent = false
+++

> 优化可以分为两类：工作减少（work reduction）与硬件赋能（hardware enablement）。

就计算机编程而言，“工作”（work）大体上是对程序需要做多少事情的一个总体度量。

工作优化的思路是减少程序需要做的事情的总量。常用的技术包括：`approximation`、`tail-recursion elimination`、`coarsening/refining recursion`、`inlining`、`loop fusion`、`loop unrolling`、`hoisting`、`short-circuiting`、`common-subexpression elimination`、`compile-time initialization`、`compile-time evaluation`、`exploiting sparsity`、`caching`、`pre-computation` 以及 `bit hacks`。

减少工作无疑是降低整体运行时间的重要启发式原则，但它并不是运行时间的唯一决定因素；它并没有涵盖计算机编程的全貌，因为它没有处理计算机硬件本身的复杂特性。

要深入研究能够释放硬件潜力的架构改进，我们必须深入到硬件与微架构（micro-architecture）的诸多方面：`the ISA`、`pipeline stages`、`superscalar processing`、`out-of-order execution`、`paging`、`caching`、`vectorization`、`speculation`、`hardware prefetching`、`branch prediction` 等等。

> 纵观历史，计算机架构提升性能的手段无外乎利用局部性（locality）或并行性（parallelism）。

为了利用局部性，内存层级（`registers`->`L1/L2/L3 caches`->`local DRAM`->`remote DRAM`->`PMem`->`SSD`）被做得更深，以掩盖性能问题；硬件预取器和分支预测器被用来预测即将到来的访问，把数据或指令移到离处理器更近的地方。作为程序员，我们工作在计算机架构层之上的一层，我们能做的是编写 NUMA 感知、缓存对齐、最好还能向量化的代码，保持规则的数据访问模式，并辅以恰当的软件预取。

为了利用并行性，人们引入了带 micro-ops 的超标量乱序流水线、向量硬件和多核。相应地，我们需要通过 `bit tricks`、`ILP`（指令级并行，Instruction-level parallelism）、`AVX`/`SSE`、`AMX`、多线程/多进程编程，以及将计算卸载到 `DPU` 或 `GPU` 这类加速器上，让所有这些硬件都保持忙碌。

让我们进一步探讨 `ILP`，因为它与处理器内部的 μ-arch 设计——如乱序执行、数据旁路（data bypassing）、寄存器重命名（register renaming）等等——联系更为紧密。要以编程方式利用 CPU 的 μ-arch，我们可以：(1) 使用相互独立的功能单元；(2) 为分支预测添加 likely/unlikely 提示；(3) 提前打破数据流图中的依赖，以减少数据冒险（data hazard）。
