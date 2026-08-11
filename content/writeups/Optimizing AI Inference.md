+++
title = "Optimizing AI Inference"
date = "2024-08-28"
tags = ["sys", "ai"]
description = "A summary of inference optimization problems in AI engineering."
showFullContent = false
+++

## Levels of Abstraction in Optimization
AI engineering on the application side can be roughly divided into MLOps and inference optimization. Inference optimization itself can be further divided into several levels of abstraction:
1. At the highest level is model optimization (see [^13]): quantization, knowledge distillation, parameter pruning, channel pruning.
2. The middle layer is graph optimization and general-purpose optimization: operator fusion/reconstruction, loop interchanges, data layout rewrites.
3. The lower layer is hardware operators: the most optimized low-level operators are necessarily the most specialized (tensor-specific + hardware-specific + framework-agnostic), so the mid-level IR must be bound to hardware operators (i.e., backend operator selection). These hardware operators typically need to exploit vectorization, parallelism, and locality based on device information, perform explicit cache management, hide memory latency through sensible instruction reordering, and use SIMD/AMX/DSP/GPGPU architectures for memory tiling and minibatch block GEMM.
4. At the very bottom there are also LLVM-level low-level codegen optimizations, responsible for finally generating optimized machine code.

The first level, model optimization, reduces the model's total amount of computation and is orthogonal to the lower-level optimizations. The second- and third-level inference optimizations are ultimately both about hardware enablement — except that the second level works for nearly all hardware, while the third level is refined according to device information. In terms of implementation, the second- and third-level optimizations can be further divided into AI-compiler-based optimization and manual optimization.

## Compilation-Based Optimization
Compilation — or DSLs + optimizing compilers — is a general solution to domain-specific optimization problems, providing portable optimizations for different hardware. A decade ago, Halide already offered a DSL + compiler for parallel computation on images and tensors, decoupling the algorithm specification itself from the optimization details. Even gcc/llvm, at an even lower level, are examples of decoupling algorithms from optimizations, doing optimization at the machine code codegen level.

### Strengths and Weaknesses of ML Compilers
Today's ML compilers are a continuation of this compilation-based approach[^2]. Their strengths are:
- Portability: on the one hand, hardware keeps iterating and updating; on the other hand, new hardware and new architectures keep emerging. Devices on the edge side are far more diverse. The datacenter scenario is better off, but it still faces the NVIDIA export ban and the domestic-substitution question; the domestic Chinese GPGPU market has not yet settled into a clear one-or-two-winners landscape[^10], so adapting to new hardware is something that must be solved in the near future.
- Clean decoupling of algorithms from optimizations: this improves engineering efficiency, and also helps reduce the risk of injecting defects as complexity explodes — a risk that would let the project gradually spiral out of control and become unmaintainable.

Their weaknesses are:
- Incompleteness: performance is good when handling most scenarios and routine problems, but there are always unexpected edge cases that fall back to the slow path, where performance degrades abruptly, and it is hard to generate better code by tweaking the DSL code.
- Inflexibility: given that compiler codegen output is often not very human-readable, it is difficult in practice to do flexible manual optimization for edge cases and ad-hoc requirements.
- Hard to truly beat expert manual optimization: just as there is still no functional-language compiler that can beat C/C++ in performance, an ML compiler only produces a good-enough solution, usually inferior to a fully expert-optimized C/C++ implementation.

### The ML Compiler Workflow
An ML compiler's workflow is a ``lowering`` process from high-level abstraction to low-level abstraction. The ML compiler backend contains various ``passes``; a pass is simply a lowering rule. Based on device information, it ultimately forms a hardware-specialized, tensor-specialized hardware operator description — called a ``schedule`` in TVM and a ``plan`` in Triton. The further CodeGen stage translates from the ML compiler's own language into a language compiler backend such as LLVM IR, which is then handed to LLVM to compile into executable machine code.

In MLIR, ``dialects`` can layer or categorize passes. A typical dialect layering (see [^1]), from top to bottom, looks like this:
```
OpGraph -> TSOWB(e.g. late hlo) -> CGASel -> HHO(e.g. Linalg) -> MHA(e.g. stripe/affine) -> HLTSIR(e.g. vector dialects) -> TSIR(e.g. llvm)
```

Triton's rough pipeline is as follows:
```
User-facing Python/C++ kernel code
--> [ML Compiler frontend; sometimes possibly just a dynamic-to-static tool: run forward once, then transcribe]
Device-agnostic high-level IR
--> [ML Compiler backend passes: graph optimization + operator selection + memory optimization]
Hardware-specialized low-level IR [Schedule/Plan]
--> [ML Compiler backend passes: translate the internal Schedule/Plan to LLVM IR]
LLVM IR 
--> [LLVM's NVPTX back-end, entering the language compilation layer]
PTX
--> [CUDA ptxas assembler]
CUBIN
```

The lowering pipeline of Intel's MLIR graph compiler is as follows:

Computation Graphs -> linalg [^4] -> layout propagation [^7] -> tiling [^3] -> fusion [^8] -> micro kernel [^9] -> vector [^10] -> bufferization [^12] -> memory planning -> LLVM IR -> handed over to LLVM.

The above lowering pipeline can also be divided into two large regions: tensor-land and memref-land; everything before bufferization is tensor-land. In tensor-land, all tensor operations are by default not in-place — even relu, which obviously could be in-place. Only in memref-land are further optimizations of memory access considered.

## Manual Optimization
In many cases, the target model's architecture is fixed, and there are only a few fixed target machine architectures, so the portability advantage of ML compilers — the advantage of automatic operator binding/operator selection — virtually disappears.

A typical example is llama.cpp, and the ggml that underpins it. Through a human-readable, minimal C/C++ project, llama.cpp+ggml implements the most direct and effective optimization mechanisms at every level of abstraction: quantization at various precisions, automatic differentiation, AVX/AVX2 optimizations, Metal optimizations, the flash-attention operator, multi-GPU pipeline parallelism, and so on — and it ultimately achieves quite good results, especially on local devices.

The strength of such a manually optimized system is that the system is transparent: programmers can read the entire system and pinpoint the exact line of code involved in a given problem, giving it a flexibility that ML compilation solutions lack. It is well suited to ad-hoc requirements and to scenarios where the model architecture and hardware devices are stable.

The weakness of manual optimization is that once a new architecture or new device appears, the code has to be rewritten.


[^1]: [Linalg Dialect Rationale: The Case For Compiler-Friendly Custom Operations](https://mlir.llvm.org/docs/Rationale/RationaleLinalgDialect/)
[^2]: TVM can be regarded as Halide's ... in the ML domain
[^3]: Operator tiling — or the matmul tiling layer — belongs to the scf dialect, i.e., structured control flow; it was previously called LoopOps.
[^4]: ``Linalg`` is a DSL(a high-level MLIR dialect) for expressing linear algebra operations in MLIR, designed to solve the High-level Hierarchical Optimization (HHO box) in MLIR and to interoperate nicely within a Mixture Of Expert Compilers environment (i.e. the CGSel box). 
[^5]: [MLIR — Lowering through LLVM](https://www.jeremykun.com/2023/11/01/mlir-lowering-through-llvm/)
[^6]: [A friendly introduction to machine learning compilers and optimizers](https://huyenchip.com/2021/09/07/a-friendly-introduction-to-machine-learning-compilers-and-optimizers.html)
[^7]: Adjust the layout — e.g., tile $M\times N$ into $32\times 32$ blocks.
[^8]: Operator fusion, e.g., elementwise+reduce op fusion.
[^9]: Micro kernels are usually handwritten — for example, the smallest-granularity matmul after tiling, typically 64*64. Handing it directly to the compiler does not work well: different hardware requires different instructions, different orderings, and different registers.
[^10]: Domestic Chinese AI chips are in a free-for-all stage: Huawei's Atlas series, Biren BR100, Rockchip RK NPU, Baidu Kunlunxin XPU, Bitmain (bm-se/sc), Cambricon MLU, Hygon DCU, Enflame GCU, etc.
[^11]: Various vector operations, which can be further divided into the GPU dialect, the Arm-Neon dialect, the x86vector dialect (AVX, AVX512), the AMX dialect for 4th-gen Xeon, and so on.
[^12]: Bufferization in MLIR is the process of converting ops with tensor semantics to ops with memref semantics. At this stage, the compiler tries as much as possible to make the memory usage of some tensor computations in-place; the ultimate goal is to use less memory and reduce the number of copies.
[^13]: [Quantization and Pruning](https://jipeng4974.github.io/writeups/quantization-and-pruning)
