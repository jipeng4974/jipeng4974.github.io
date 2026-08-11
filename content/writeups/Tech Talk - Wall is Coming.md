+++
title = "Tech Talk: Wall is Coming"
date = "2024-02-22"
tags = ["Sys", "Talk"]
description = "Tech Talk transcript: traces the historical roots of the memory wall problem, attempts to frame an understanding of the optimization space, derives corresponding heuristics, and lists several memory-access optimization techniques."
showFullContent = false
+++

# The Memory Wall and the Breakdown of Dennard Scaling
The "Memory Wall" and the "breakdown of Dennard Scaling" are the two core contradictions in the evolution of the computing ecosystem.
- To counter the breakdown of Dennard Scaling, compute hardware architecture broke out from single-core toward multi-core and many-core.
- To mask the memory wall problem, the memory hierarchy has grown deeper and deeper, and off-chip interconnect bandwidth has had to increase rapidly.

At first, the move from single-core to multi-core was a seismic shift: it forced painful restructuring of software architecture, and concurrent programming problems stood out like a tree above the forest — so they got focus-fired and taken down by nearly 20 years of industry practice plus academic research. Today we have plenty of tools: multithreading paradigms, async callback paradigms, goroutine-style stackful coroutines, C++/Rust async/await stackless coroutines, lockfree/waitfree data structures, and so on.
By contrast, over these decades the memory wall problem — owing to its hidden, non-urgent, and intractable nature — has not only failed to be properly solved but has become deeply entrenched, and it can no longer be concealed: it is now exposed right in front of software engineers. So the focus of this talk is the memory wall.

But before getting to the main topic, I need to give a bit of background on the evolution of data center hardware and microprocessor architecture.

# A Brief History of Data Center Hardware Evolution in the 21st Century
## The 00s: The Commodity Computing Era
In ancient times we didn't talk about "data center hardware," because the Internet industry in the modern sense didn't exist yet, so neither did data centers in the modern sense. Naturally there was no marketing speak like "for big data / cloud / Edge / AI" either; the emphasis was more on building distributed stateful systems out of cheap, unreliable, large-scale commodity hardware, and Google made pioneering attempts in this direction.

> Compared with IBM mainframes and supercomputers, the cheapness of x86 back then was plain to see. Google ran on Pentium II when it was first founded.
> The banking industry and DARPA made the IBM Power series. Today, although Power9/10 machines are completely crushed by Xeon/EPYC, they can still linger on thanks to the unchangeable ancestral laws of banking software and government orders, holding on to a niche market.

![commodity_computing](https://jipeng4974.github.io/img/commodity_computing.png)

Before the 10s, x86 microprocessors had no multi-core scalability at all.
- The `FSB` (front-side bus) was the chief culprit. In the architecture shown below, the memory bus and the PCIe bus had to share a single `FSB` to connect to the CPU, making the `FSB` the bottleneck and leaving the CPU count unable to scale out.
- Back then `PCIe` was still 1.0 (the first-generation `PCIe` from 2003), with quite limited bandwidth and lane counts. Even if you forced multi-core in, network IO and disk IO couldn't keep up.
![fsb](https://jipeng4974.github.io/img/fsb.png)

The 00s happened to coincide with Moore's Law gradually failing in the single-core context; focusing purely on improving chip performance was no longer viable after 2004, and hardware vendors had to break out architecturally toward multi-core.
![clock](https://jipeng4974.github.io/img/clock.png)

## The Early 10s: The Multi-Core Era
The Intel Xeon E5-2600 V1 32nm Sandy Bridge in 2012 was a milestone server product: it removed the `FSB` (this had actually already been done on the 2009 Nehalem machines), introduced `QPI`/`DMI` to replace the `FSB`, plus PCIe 2.0, giving the microarchitecture multi-core scalability.
> The E5 family is a household name; although most units are reaching the end of their lifespan, many companies should still be using them today.
> When buying a computer back then, the so-called second-gen i3/i5/i7 was `SandyBridge`, a full generation ahead of the first-gen i5/i7 `Nehalem`.

After `Sandy Bridge` came `Haswell`, which changed little. After `Haswell` came `Broadwell`, from which the ring topology `Broadwell Ring` takes its name.
![clock](https://jipeng4974.github.io/img/broadwell_ring.png)
After that came Skylake, which began to carry the Scalable name — moving from multi-core to many-core, from the early modern era into the modern era.

## The Late 10s to Early 20s: The Many-Core Era
In 2017 Intel released the 1st gen Xeon Scalable, `Skylake`, which adopted a Mesh Architecture. See Things are getting meshy
Around the same time AMD also launched the EPYC 7001, which more or less broke Xeon's monopoly. We have quite a few AMD machines in our US-TTP data center.

The successor to `Skylake`, `Ice Lake`, did not hatch smoothly, because 2018 brought the big Meltdown/Spectre news (security vulnerabilities in speculative execution). So the vulnerabilities were patched on `Skylake`, and `Cascade Lake` was released in 2019 as the 2nd-Gen Xeon. The original heir in line, `Ice Lake`, arrived belatedly in 2021 and became the third generation.

`Cooper Lake` is of the same generation as `Ice Lake` — both are called the third generation — but its architecture is actually different from `Ice Lake`'s; it was modified from `Skylake` and designed specifically for multi-socket (4~8s) systems. Compared with the general-purpose `Ice Lake`, `Cooper Lake` slightly exceeds the commodity hardware category; presumably it was meant to be sold to certain specialized computing fields, to replace aging UNIX systems such as `Oracle Solaris` and `IBM AIX`. In Internet scenarios we still prefer scaling out over scaling up.

Currently the mainstay models for data center applications are `Ice Lake` and `Cascade Lake` machines; starting from 2023~2024, compute/memory-intensive scenarios will gradually move to the fourth-generation Xeon: `Sapphire Rapids` machines.
In 2019 and 2020 we still had a scattered handful of 1st gen Xeon Scalable Gold machines, which were soon phased out.
![clock](https://jipeng4974.github.io/img/broadwell_ring.png)

Both `Ice Lake` and `Cascade Lake` use a monolithic mesh design.
- Here monolithic refers to the paradigm of a single huge die carrying many cores, as opposed to chiplet/tile-based designs;
- And mesh refers to the mesh topology, as opposed to the ring topology of the Broadwell Ring from the earlier E5 era.

The differences between the two lie mainly in the process node (10nm vs 14nm), the maximum core count, the number of PCIe lanes, and the number of memory channels (the number of DIMMs[^1] supported per socket: 16 vs 12).

The 4th-Gen `Sapphire Rapids`, which began shipping in 2024, transitions its chip architecture from mono-die to a more AMD-like multi-die design (Intel calls it tile-based); the microarchitecture was updated from sunny cove to golden cove (the tpause instruction can be used to optimize spinlocks). Its configuration is quite gorgeous: support for advanced interconnect protocols (`PCIe5`/`CXL`), DDR5, the new `AMX` instruction set, and the top-end configuration even has 3D-stacked on-package HBM[^2].

## Some Takeaways
### In the hardware ecosystem, life and environment also shape each other
PC users created the prosperous, accessible, standardized x86 commodity hardware market. Commodity hardware clusters happened to suit Internet workloads (no heavy floating-point computation, mostly integer servers, but an enormous volume of data), which made possible the modern data center supported by distributed systems; only then did hardware vendors start customizing and optimizing server hardware for data centers, such as Scalable Xeon.

PC gamers made the Nvidia GPU; GPUs happened to suit AI workloads, hence all kinds of MLSys and GPGPU applications. Only then came Nvidia's investment in accelerated computing. Once the A100 existed, ChatGPT training was done as large-scale model parallelism tailor-made on top of A100 + IB networks. The sensational effect of ChatGPT then fed back into the high-end GPU industry.

An ecological niche is hard to design artificially or create out of thin air. Take Intel Optane PMem: it occupied a very reasonable niche — a lot of systems research papers were published thanks to PMem, because it was just so reasonable, filling the gap between disk and memory. Yet it was still axed in 2022 because the demand side couldn't keep up. Fundamentally, PMem is not very useful in AI training/inference scenarios; it brings no significant cost advantage to mainstream search/ads/recommendation applications; and in transactional and analytical scenarios it either can't beat disk + memory or isn't worth the manpower for a major architectural overhaul.
### Characteristics of Modern Hardware
- Numerous cores
- Efforts to alleviate the Memory Wall problem — over the past 20 years, the yearly reduction rate of DRAM cycle time has been relatively stagnant compared with Moore's Law, and the consequences of a cache miss have grown ever more severe.
  - The memory hierarchy grows deeper: more caches; the latest top-end SPR machines even add on-package HBM; beyond local memory there are remote NUMA nodes; below memory there can be PMem and SSD; beyond the local node there are LAN/cloud nodes. In short, the Memory Wall problem is hidden by adding more levels.
  - Off-package/chip-to-chip interconnect bandwidth increases: to keep up with the IO demands brought by growing core counts and to alleviate the Memory Wall problem in bulk read/write scenarios.
### Black-Boxing and White-Boxing Happen Simultaneously
Modern hardware places harsher demands on engineering capability — only through off-CPU analysis, data-oriented cache-friendly design, manual memory management, and even manual prefetching can its performance potential truly be unleashed.

But this runs counter to the ever-simplifying direction of modern software engineering — freeing programmers' minds through runtimes, virtual machines, dynamic languages, the separation of responsibilities in the microservices paradigm, hypervisor-based virtualization, and containerization.

This divergence takes modern software practice down two forks: one is infrastructure building with white-boxing as the means — better hypervisors, better mlsys, high-performance retrieval, high-performance storage, high-performance networking; the other is upper-layer applications with black-boxing as the goal — exploiting the convenience of virtualization, containerization, microservicization, dynamic languages, and runtime languages to improve productivity.

# Performance Engineering Practice on Modern Hardware
## Understanding the Optimization Space
Performance engineering is software optimization practice guided by a systematic methodology.
An "optimization task" can be decomposed from two perspectives:
- Optimization = reducing total algorithmic work + hardware enablement
- Optimization = reducing runtime = reducing CPU time + reducing blocked time
"Algorithm improvement" and "hardware enablement" are nearly orthogonal, and "on-CPU time" and "off-CPU time" are largely complementary. So we can construct an orthogonal basis {hardware enablement $x$, algorithmic optimization $y$, arithmetic intensity $z$} for the optimization space $W$, giving $W = \{[x,y,z] \in R^3 | 0 \le z \le 1 \}$.

<div id="damn"><svg width="480" height="420"></svg></div>

## Some Derived Heuristics
Based on an understanding of the full picture of the optimization space, we can derive some performance engineering heuristics:

- You should first determine the arithmetic intensity of the program
  - Why is distinguishing on-CPU vs off-CPU crucial? Because your 100%-busy CPU may not actually be busy. CPU profiling only describes part of the full picture — sometimes even a small part. The commonly used CPU utilization metric is deceptive and misleading: it actually covers both on-CPU computation and off-CPU blocking. If there is a severe memory-access bottleneck, the superscalar pipeline of a 100%-utilized CPU is full of stall bubbles, and the CPU's various arithmetic logic units and dedicated compute hardware like SIMD/AMX are all waiting idly.
  - Of course, high off-CPU may also be caused by disk/network-IO intensity, but such IO-bound applications are usually not the bulk of the cost and don't warrant a big optimization campaign — only infra departments specializing in storage or networking need to care. Alternatively, the code itself may be problematic: lock granularity too coarse, locks held too long, which likewise causes an excessively high off-CPU ratio.
- Memory today should be treated as a peripheral
  - Data structures once designed for slow peripherals are now suitable for memory scenarios:
    - C++'s ordered map is implemented with a red-black tree, while Rust chose a B+ tree to exploit its better locality — because memory today is already as intolerably slow as disks were decades ago.
    - Data structures like DashTable, originally used on PMem (non-volatile memory, whose bandwidth is far lower than DRAM), are now used by DragonFly in its in-memory database, with performance far exceeding Redis/Memcached.
  - Therefore the memory hierarchy needs to be white-boxed to fully exploit hardware potential.
- Avoid colliding with compiler optimizations
  - There is no need to optimize multiplication and division with shifts or other assembly instructions. Multiplying by 8 will inevitably be auto-optimized into a left shift by 3. Division will also be optimized into multiply-and-add. The example below shows how a very old compiler handles volatile int y = x / 71. Compilers also use the LEA instruction to optimize multiplication: LEA was originally meant to speed up member address computation for arrays of small structs, but it can actually be used to speed up multiplication too — for example, multiplying by 5 can be written as lea eax, [eax*4 + eax]. Using existing circuitry is faster; this counts as work the compiler has already done for you in the realm of hardware enablement.
    ```assembly
    // volatile int y = x / 71;8b 0c 24        mov ecx, DWORD PTR _x$[esp+8] ; load x into ecx
    mov eax, -423447479 ; magic happens starting here...
    imul ecx            ; edx:eax = x * 0xe6c2b44903 d1           add edx, ecx        ; edx = x + edx
    sar edx, 6          ; edx >>= 6 (with sign fill)

    mov eax, edx        ; eax = edx
    shr eax, 31         ; eax >>= 31 (no sign fill)
    add eax, edx        ; eax += edx

    mov DWORD PTR _y$[esp+8], eax
    ```
  
  - You can safely do manual SIMD. Manual SIMD and LLVM's auto-vectorization sit at different abstraction levels, and basically you shouldn't place expectations on LLVM's auto-vectorization either. In the short term there is no hope of a good SIMD abstraction at the programming-language or library level.
    - Manual SIMD requires the programmer to choose appropriate instructions based on the target machine's microarchitecture model (the latency of various SIMD instructions differs across microarchitectures);
    - choose an appropriate SIMD size (bigger is not always better, and different sizes correspond to different shuffles);
    - handle the various corner cases caused by leftover data that doesn't fill a full batch at the end;
    - align data addresses, or tolerate unaligned data.
  - Use the latest compiler and use O3; then many details need no manual optimization, such as copy elision, tail-recursion elimination, even mutual-recursion elimination, inlining, and most loop optimizations — loop unrolling (+ stride) / fission (inverse fusion) / tiling (cache blocking) / unswitching (de-branching the inner loop) / auto-vectorization / interchange.
  - Optimizations that involve logic and concrete application scenarios, for which there is no standard optimization strategy, still have to be done yourself — e.g., loop fusion, adjusting recursion granularity, and adjusting the encoding strategy (the tradeoff between compactness and encode/decode overhead).


## Some Tricks for Memory-Access Optimization
- How to measure arithmetic intensity or memory-access intensity?
  - perf or perf_event_open: cache_miss/instructions, or ipc
  - Intel PCM
  - eBPF tools, e.g. https://github.com/iovisor/bcc
  - Static analysis: the proportion of load/store instructions can roughly indicate the arithmetic/memory-access intensity, but it is heavily affected by the cache hit rate.
- How to set appropriate object padding based on the memory configuration?
  - There are two important concepts in memory configuration that affect the application layer: memory channels and memory ranks. Under the x86 architecture, memory channels and memory ranks are interleaved across memory addresses, i.e., evenly distributed and increasing. Therefore RAM can be viewed as composed of $n_chan \times n_rank$ blocks. Its DIMM architecture is shown in the figure below.
[Image]
  - For the concrete object padding approach, refer to the pseudocode below, where 64B is the cache line size and also happens to be the size of one block. The general idea is to first ensure that all addresses in the memory pool are integer multiples of 64B, then ensure that the next object's block id is coprime with n_chan*n_rank.
  - This padding prevents objects' starting addresses from repeatedly hitting the same channel or the same rank, making the next object's starting address fall into a different channel/rank, fully utilizing the different memory channels and memory ranks, avoiding load imbalance across channels and ranks, and improving memory-access bandwidth.
    ```C  
    static unsigned int object_align(unsigned int obj_size)
    {
            unsigned nchan = get_nchannel();
            unsigned nrank = get_nrank();
            unsigned new_obj_size = (obj_size + 63) / 64; 
            while (get_gcd(new_obj_size, nrank * nchan) != 1)
                    new_obj_size++;
            return new_obj_size * 64; 
    }
    ```
- Cache optimization: align to cache lines to avoid false sharing; use cache blocking.
- Avoid lock bottlenecks
  - Based on static lock analysis or eBPF off-CPU analysis, find locks with excessively coarse granularity and locks held for excessively long.
  - Look for better concurrent data structures: lock-free implementations vary widely in quality; the best many-core scalability always comes from arrays.
  - Kernel bypassing: avoid certain kernel implementations that carry locks, e.g., replacing the kernel stack with a user-space network protocol stack.
  - Share-nothing: the most extreme approach is to emulate seastar — each core executes single-threaded code only on its own dedicated memory, avoiding CPU-to-CPU traffic as much as possible and eliminating locks from the code entirely.
- In-register storage: try to keep the parameters used by simple functions and the containers holding intermediate results small enough to fit entirely in registers; the compiler will then automatically optimize away all loads/stores.
- Consider pre-allocated memory and on-stack static structures: when memory access is unavoidable, long-lived objects can use pre-allocated memory; temporary containers can be designed — based on the online data distribution — as static structures partitioned by some rule and placed on the stack (stack allocation is just a stack pointer move, whereas malloc is far more complex; and the default malloc even has a global lock, so it doesn't scale).
- Consider compile-time evaluation and global static memory regions.
- Use 1GB huge pages to store data. Compared with the default 4KB pages, huge pages greatly reduce the total number of page table entries needed, which can significantly shrink page table size and TLB size, reduce TLB miss and page table walk overhead, and improve the continuity of memory allocation and the locality of memory access — all of which help improve memory bandwidth.
- Use 4MB large pages to store code: the .text segment can also use larger pages (though the maximum supported is only 4MB). The ITLB, like the DTLB, causes stalls on misses. ITLB problems can be diagnosed with the help of https://github.com/intel/iodlr; the solution is to move .text into existing large pages[^3], or to link statically and use the libhugetlbfs library. I haven't yet seen this optimization land in a real project, but it looks quite promising; see [Runtime Performance Optimization Blueprint: Large Code Pages](https://www.intel.com/content/dam/develop/external/us/en/documents/runtimeperformanceoptimizationblueprint-largecodepages-q1update.pdf).
- Respect the NUMA topology and avoid remote memory accesses, i.e., UPI traffic.
![numa](https://jipeng4974.github.io/img/NUMA.png)
- Take advantage of new architectural features and new instruction set extensions: for example, AMX-based GEMM precision and performance optimization is widely used in various training/inference frameworks; the AVX512_IFMA instruction extension for big-number multiplication has been used in newer versions of OpenSSL; and QAT (QuickAssist)-based hardware acceleration for cryptographic applications such as AES, RSA, and ECC.
- Use prefetching to make the cache smarter.
  - So-called hardware prefetching means prefetching data from memory into cache (usually the LLC). The hardware prefetcher has simple stride-pattern recognition logic; for example, a loop like a, a+2, a+4, a+6 can be recognized. There is no need to deliberately trigger hardware prefetching — normal code triggers it. But you need to avoid triggering it by mistake: for example, a large struct spanning multiple cache lines may only need its first few fields accessed, but the hardware prefetcher mistakenly thinks it will keep reading, causing cache pollution. Slightly adjusting the access order of those fields to break the constant stride pattern is enough.
  - Find the right timing to use software prefetching: prefetching data brings huge throughput improvements and can also hide latency, but exactly when to prefetch and which data to prefetch can only be found through repeated trial and error. And once the timing is chosen wrongly, it instead causes cache pollution and degrades performance.
    - Try to spread out prefetch instructions (ideally spread out from loads too), interleaving them among computation instructions. If you prefetch continuously, that — like a bunch of loads — will also create bubbles.
    - Choose an appropriate PSD (prefetch scheduling distance), i.e., how many iterations ahead to prefetch. For a loop with heavy computation, you can prefetch 1 iteration ahead; for one with light computation, you may need to prefetch multiple iterations ahead. In the example below, PSD=3.
    ```assembly    
    top_loop:
    prefetchnta [edx + esi + 128*3]
    prefetchnta [edx*4 + esi + 128*3]
    movaps xmm1, [edx + esi] 
    movaps xmm2, [edx*4 + esi]
    movaps xmm3, [edx + esi + 16] 
    movaps xmm4, [edx*4 + esi + 16]
    add esi, 128 
    cmp esi, ecx 
    jl top_loop
    ```

    - With doubly nested loops, you need to fill the bubbles when switching between the inner and outer loops, and also prefetch for the outer loop.

    ```C    
    for (i = 0; i < 100; i++) {  
    for (j = 0; j < 32; j+=8) { 
        prefetch a[i][j+8]  // on the last iteration, no prefetch needed
        computation a[i][j] // the first a[i+1][j] is not prefetched and will miss
    } 
    }
    // after optimization
    for (i = 0; i < 100; i++) {  
    for (j = 0; j < 24; j+=8) {
        prefetch a[i][j+8]  
        computation a[i][j]  
    }  
    prefetch a[i+1][0]  // prepare a[i][0] ahead of time, otherwise it will stall on a[i][j+8]
    computation a[i][j] // handle the last iteration separately, since it needs no prefetch
    }
    ```
- Take advantage of advanced interconnect technologies, such as `CXL`.

![cxl](https://jipeng4974.github.io/img/spr-cxl.png)


[^1]: DIMM (dual in-line memory module), i.e., a RAM stick — the physical embodiment of DDR (Double Data Rate) technology.
[^2]: https://www.intel.com/content/www/us/en/products/sku/232592/intel-xeon-cpu-max-9480-processor-112-5m-cache-1-90-ghz/specifications.html
[^3]: https://github.com/intel/iodlr/blob/master/large_page-c/large_page.c

<style type="text/css">
svg {
    box-shadow: 0 0 10px #999;
    border-radius: 5px;
}
</style>
<script type="module">
import {
  drag,
  color,
  select,
  range,
  randomUniform,
  randomNormal,
  scaleOrdinal,
  selectAll,
  schemePastel1,
} from "https://cdn.skypack.dev/d3@7.8.5";
import {
    gridPlanes3D,
    points3D,
    lineStrips3D,
} from "https://cdn.skypack.dev/d3-3d@1.0.0";
document.addEventListener("DOMContentLoaded", () => {
    console.log("dom loaded, starts to draw svg ...");
    const width = 480;
    const height = 420;
    const origin = { x: width/2, y: height/2 };
    const offset = origin.x - origin.y;
    const j = 10;
    const scale = 20;
    const key = (d) => d.id;
    const startAngle = Math.PI/2;
    // const startAngle = 0;
    const colorScale = scaleOrdinal(schemePastel1);
    let scatter = [];
    let yLine = [];
    let xLine = [];
    let zLine = [];
    let xGrid = [];
    let beta = 0;
    let alpha = 0;
    let mx, my, mouseX = 0, mouseY = 0;
    const svg = select("svg")
        .call(
          drag()
            .on("drag", dragged)
            .on("start", dragStart)
            .on("end", dragEnd)
        )
        .append("g");
    const grid3d = gridPlanes3D()
        .rows(20)
        .origin(origin)
        .rotateY(startAngle)
        .rotateX(-startAngle)
        .scale(scale);
  const points3d = points3D()
    .origin(origin)
    .rotateY(startAngle)
    .rotateX(-startAngle)
    .scale(scale);
  const yScale3d = lineStrips3D()
      .origin(origin)
      .rotateY(startAngle)
      .rotateX(-startAngle)
      .scale(scale);
  const xScale3d = lineStrips3D()
      .origin(origin)
      .rotateY(startAngle)
      .rotateX(-startAngle)
      .scale(scale);
  const zScale3d = lineStrips3D()
      .origin(origin)
      .rotateY(startAngle)
      .rotateX(-startAngle)
      .scale(scale);
  function processData(data, tt, recolor) {
    /* ----------- GRID ----------- */
    const xGrid = svg.selectAll("path.grid").data(data[0], key);
    xGrid
      .enter()
      .append("path")
      .attr("class", "d3-3d grid")
      .merge(xGrid)
      .attr("stroke", "black")
      .attr("stroke-width", 0.3)
      .attr("fill", (d) => (d.ccw ? "#eee" : "#aaa"))
      .attr("fill-opacity", 0.7)
      .attr("d", grid3d.draw);
    xGrid.exit().remove();
    /* ----------- POINTS ----------- */
    const points = svg.selectAll("circle").data(data[1], key);
    function GetColor(x, y){
      // console.log("x: %d, y: %d", x, y);
      // return (x > 0) ? 5 : -5 + (y > 0) ? 3 : -3;
      if (x >= 0 && y >= 0) return schemePastel1[0];
      if (x < 0 && y >= 0) return schemePastel1[1];
      if (x < 0 && y < 0) return schemePastel1[2];
      if (x >= 0 && y < 0) return schemePastel1[3];
    }
    if(recolor){
      points
      .enter()
      .append("circle")
      .attr("class", "d3-3d")
      .attr("opacity", 0)
      .attr("cx", posPointX)
      .attr("cy", posPointY)
      .merge(points)
      .transition()
      .duration(tt)
      .attr("r", 3)
      .attr("stroke", (d) => color(colorScale(d.id)).darker(3))
      .attr("fill", (d) => GetColor(d.projected.x - origin.x, d.projected.y - origin.y))
      .attr("opacity", 1)
      .attr("cx", posPointX)
      .attr("cy", posPointY);
    }else{
      points
      .enter()
      .append("circle")
      .attr("class", "d3-3d")
      .attr("opacity", 0)
      .attr("cx", posPointX)
      .attr("cy", posPointY)
      .merge(points)
      .transition()
      .duration(tt)
      .attr("r", 3)
      .attr("stroke", (d) => color(colorScale(d.id)).darker(3))
      .attr("opacity", 1)
      .attr("cx", posPointX)
      .attr("cy", posPointY);
    }
    points.exit().remove();
    /* ----------- x-Scale ----------- */
    const xScale = svg.selectAll("path.xScale").data(data[3]);
    xScale
      .enter()
      .append("path")
      .attr("class", "d3-3d xScale")
      .merge(xScale)
      .attr("stroke", "black")
      .attr("stroke-width", 1.5)
      .attr("d", xScale3d.draw);
    xScale.exit().remove();
    /* ----------- y-Scale ----------- */
    const yScale = svg.selectAll("path.yScale").data(data[2]);
    yScale
      .enter()
      .append("path")
      .attr("class", "d3-3d yScale")
      .merge(yScale)
      .attr("stroke", "black")
      .attr("stroke-width", 1.5)
      .attr("d", yScale3d.draw);
    yScale.exit().remove();
    /* ----------- z-Scale ----------- */
    const zScale = svg.selectAll("path.zScale").data(data[4]);
    zScale
      .enter()
      .append("path")
      .attr("class", "d3-3d zScale")
      .merge(zScale)
      .attr("stroke", "black")
      .attr("stroke-width", 1.5)
      .attr("d", zScale3d.draw);
    zScale.exit().remove();
    /* ----------- y-Scale Text ----------- */
    const yText = svg.selectAll("text.yText").data(data[2][0]);
    function GetYText(y){
      if (y==-11){
        return "[Arithmetic Intensity]";
      }else{
        return  (-y*10 + 100)/2+"%";
      }
    }
    function GetYWeight(y){
      if (y==-11){
        return 700;
      }else{
        return  350;
      }
    }
    yText
      .enter()
      .append("text")
      .attr("class", "d3-3d yText")
      .attr("font-family", "system-ui, sans-serif")
      .merge(yText)
      .each(function (d) {
        d.centroid = { x: d.rotated.x, y: d.rotated.y, z: d.rotated.z };
      })
      .attr("x", (d) => d.projected.x)
      .attr("y", (d) => d.projected.y)
      .style("font-weight", (d) => GetYWeight(d.y))
      .text((d) => GetYText(d.y))
      .attr("fill", "#78E2A0");
    yText.exit().remove();
    /* ----------- x-Scale Text ----------- */
    const xText = svg.selectAll("text.xText").data(data[3][0]);
    xText
      .enter()
      .append("text")
      .attr("class", "d3-3d xText")
      .attr("font-family", "system-ui, sans-serif")
      .merge(xText)
      .each(function (d) {
        d.centroid = { x: d.rotated.x, y: d.rotated.y, z: d.rotated.z };
      })
      .attr("x", (d) => d.projected.x)
      .attr("y", (d) => d.projected.y)
      .attr("z", (d) => d.projected.z)
      .text((d) =>  d.x == 10 ? "[Hardware Enablement]" : "")
      .style("font-weight", 700)
      .attr("fill", "#78E2A0");
    xText.exit().remove();
    /* ----------- x-Scale Text ----------- */
    const zText = svg.selectAll("text.zText").data(data[4][0]);
    zText
      .enter()
      .append("text")
      .attr("class", "d3-3d zText")
      .attr("font-family", "system-ui, sans-serif")
      .merge(zText)
      .each(function (d) {
        d.centroid = { x: d.rotated.x, y: d.rotated.y, z: d.rotated.z };
      })
      .attr("x", (d) => d.projected.x)
      .attr("y", (d) => d.projected.y)
      .attr("z", (d) => d.projected.z)
      .text((d) =>  d.z == 10 ? "[Work Reduction]" : "")
      .style("font-weight", 700)
      .attr("fill", "#78E2A0");
    zText.exit().remove(); 
    selectAll(".d3-3d").sort(points3d.sort);
  }
  function posPointX(d) {
    return d.projected.x;
  }
  function posPointY(d) {
    return d.projected.y;
  }
  function init() {
    xGrid = [];
    scatter = [];
    yLine = [];
    xLine = [];
    zLine = [];
    let cnt = 0; 
    for (let z = -j; z < j; z++) {
      for (let x = -j; x < j; x++) {
        xGrid.push({ x: x, y: 0, z: z}); // grid position
        scatter.push({
          x: x,
          y: randomNormal(0, 0.8)()*3,
          // y: randomUniform(9, -9)(),
          z: z,
          id: "point-" + cnt++,
        });
      }
    }
    range(-10, 12, 1).forEach((d) => {
      yLine.push({ x: 0, y: -d, z: 0 });
      xLine.push({ x: -d, y: 0, z: 0 });
      zLine.push({ x: 0, y: 0, z: -d });
    });
    const data = [
      grid3d(xGrid),
      points3d(scatter),
      yScale3d([yLine]),
      xScale3d([xLine]),
      zScale3d([zLine]),
    ];
    processData(data, 1000, true);
  }
  function dragStart(event) {
    mx = event.x;
    my = event.y;
  }
  function dragged(event) {
    beta = (event.x - mx + mouseX) * (Math.PI / offset);
    alpha = (event.y - my + mouseY) * (Math.PI / offset) * -1;
    const data = [
      grid3d.rotateY(beta + startAngle).rotateX(alpha - startAngle)(xGrid),
      points3d.rotateY(beta + startAngle).rotateX(alpha - startAngle)(scatter),
      yScale3d.rotateY(beta + startAngle).rotateX(alpha - startAngle)([yLine]),
      xScale3d.rotateY(beta + startAngle).rotateX(alpha - startAngle)([xLine]),
      zScale3d.rotateY(beta + startAngle).rotateX(alpha - startAngle)([zLine]),
    ];
    processData(data, 0, false);
  }
  function dragEnd(event) {
    mouseX = event.x - mx + mouseX;
    mouseY = event.y - my + mouseY;
  }
  selectAll("button").on("click", init);
  init();
});
</script>
