+++
title = "DPDK is All You Need"
date = "2024-01-05"
tags = ["Sys", "Perf"]
description = "For memory-intensive datacenter applications, DPDK offers an excellent performance engineering paradigm."
showFullContent = false
+++

For memory-intensive datacenter applications, DPDK offers an excellent performance engineering paradigm. This post only scratches the surface, gathering a few of its ideas worth borrowing.

## EAL: The Userspace Library
EAL (Envionmemt Abstraction Layer) is DPDK's user-facing userspace library, providing a variety of useful tools, such as runtime CPU feature detection, memory management better suited to modern hardware, and CPU pinning to run a task on a specific core.

`rte_eal_init()` is the function that initializes the EAL, with implementations for multiple platforms: Linux, FreeBSD, and Windows. Taking it as an example gives a rough idea of what DPDK's EAL actually does.

`rte_eal_init()` is a lengthy initialization procedure that includes the following steps: checking whether the CPU type is supported by DPDK, setting the log level, detecting the CPUs on each socket, enabling every logical core, initializing plugins (loading shared libraries, such as some PMD drivers), initializing the [tracing mechanism](https://doc.dpdk.org/guides/prog_guide/trace_lib.html), parsing configuration options for each device, initializing the global configuration (main core id, number of logical cores, number of NUMA nodes, IOVA mode[^1], memory topology configuration), initializing the interrupt handling mechanism, initializing the multi-process common channel, scanning devices on all buses, initializing the malloc heap, registering multi-process action callbacks (for hotplug support), initializing hugepage information[^9], initializing memory and memzones, initializing the HPET/TSC timers[^8], checking the memory on the local socket, creating the communication channel between the main thread and child threads, spawning worker threads, pinning cores, starting a dummy function on the worker threads, initializing services, probing devices and drivers on all buses, starting services, and enabling telemetry (which provides status queries such as ethdev stats, ethdev port list, and EAL parameters).

## Carving Out the Privileged Work
A prerequisite for running a DPDK program in userspace is having a kernel driver that helps with things like hardware device registration and interrupt mapping. Linux offers several usable kernel drivers, such as ``vfio-pci``, ``igb_uio``, and ``uio_pci_generic``. These are generic PCI kernel driver modules that work with any PCI device. ``igb_uio`` builds on Linux UIO to provide all types of interrupt support; it is fairly old and fairly simple, does not support IOMMU, and therefore can only use PA mode for IOVA mode. ``uio_pci_generic`` is similar to ``igb_uio``, except that it does not support MSI and MSI-X interrupts. ``vfio-pci`` supports IOVA mapping based on the IOMMU and is compatible with both VA mode and PA mode. If you can use `vfio`, use `vfio` — `uio` is now largely semi-deprecated.

Most devices must first be unbound from the Linux kernel driver and then bound to the DPDK kernel driver. Before running a DPDK program, the user needs to use the ``dpdk-devbind.py`` script in the `usertools` directory to unbind and bind devices to and from kernel modules — this kind of preparatory work that requires root privileges is likewise stripped out of the userspace library.

## Use Hugepages — 4KB Is a Relic of a Bygone Era
DPDK uses mmap to allocate hugepage physical memory from hugetlbfs. Compared with the default 4KB pages (on x86, DPDK currently supports 2MB or 1GB hugepages[^7]), using larger memory pages drastically reduces the total number of page table entries required, which significantly shrinks the page table size and TLB size, lowers the cost of TLB misses and page table walks, and improves the contiguity of memory allocation and the locality of memory access — all of which help improve memory bandwidth.

## Respect the NUMA Node Topology
Every operation in DPDK is NUMA-aware, and the APIs it provides are NUMA node-affine by default, which makes it hard for users to accidentally write code that performs remote memory accesses.

## Respect the Hardware Topology of Memory
DPDK's memory allocation is extremely meticulous, taking advantage of memory configurations such as the hardware topology of memory.

Two concepts in memory configuration matter to the application layer: memory channels and memory ranks.

A memory channel is the communication channel between the CPU and memory; in theory, memory bandwidth is proportional to the number of channels. A single channel is 64 bits wide, so two channels give 128 bits. The number of memory channels usually equals the number of DIMMs[^3] supported per socket — after all, having enough memory also requires enough channels to guarantee its interconnect with the CPU.

The interface between the CPU and memory is 64 bits wide, but an individual DRAM chip may be only 4 or 16 bits wide, so multiple DRAM chips are ganged together to form a single 64-bit memory rank, connected to the same chip select, ensuring that all the chips can be accessed simultaneously. A memory module must form at least one rank to communicate with the CPU. The "2R×8" on a memory stick's label means 2 ranks with 8-bit-wide chips, for a total of 16 chips.

> Depending on memory configuration on x86 arch, objects addresses are spread between channels and ranks in RAM

On the x86 architecture, memory channels and memory ranks are interleaved across memory addresses — that is, evenly distributed and increasing — so RAM can be viewed as consisting of $n_{chan}\times n_{rank}$ blocks, with a DIMM architecture as shown in the figure below.
![2chan4rank](https://jipeng4974.github.io/img/2chan4rank.svg)

As the figure shows, a memory pool should avoid letting object start addresses repeatedly hit the same channel or the same rank; instead, it should make full use of different memory channels and different memory ranks, avoiding load imbalance across channels and ranks to improve memory access bandwidth.

DPDK's `mempool` adds appropriate padding to the object size so that the start address of the next object in the pool lands on a different memory channel and rank. See the code below for the concrete implementation, where 64B[^4] is the `cache line size` on x86 — and, as it happens, also exactly one `block size`, or `memory bus width`, or `channel width`. Either way, addresses in the memory pool must first be aligned to a multiple of 64B so they fit neatly into the cache, making them cache-friendly — in fact also block friendly and memory bus width friendly — and only then is the next object's `block id` made coprime with $n_{chan}\times n_{rank}$.

```C 
static unsigned int arch_mem_object_align(unsigned int obj_size)
{
	unsigned nchan = rte_memory_get_nchannel();
	unsigned nrank = rte_memory_get_nrank();
	unsigned new_obj_size = (obj_size + 63) / 64; 
	while (get_gcd(new_obj_size, nrank * nchan) != 1)
		new_obj_size++;
	return new_obj_size * 64; 
}
```

## The Contiguity of IOVA and VA Modes
Hardware doesn't know VAs, and userspace doesn't know PAs. One of DPDK's roles is to bridge physical addresses (PA) and virtual addresses (VA), so providing an IOVA (IO Virtual Address) is a natural design.

DPDK has two IOVA modes: PA mode and VA mode. With PA mode, every IOVA address assigned to DPDK is a physical address — or rather, it is also an IO virtual address, just one whose memory layout is exactly identical to the physical addresses. The downsides of PA mode are that it requires root privileges to read the page tables, and it may inherit the fragmentation of physical memory. DPDK therefore introduced the new VA mode, which on one hand needs no root privileges, and on the other hand remaps physical memory through the IOMMU[^2], guaranteeing the contiguity of IO virtual addresses and matching their encoding layout to the format of ordinary virtual addresses — which permits allocating large swaths of contiguous IOVA memory. From the hardware's perspective as well as from userspace's, the IOVA memory region under VA mode is contiguous.

![iova](https://jipeng4974.github.io/img/iova.png)

## Fixed Physical Addresses Are Just Right for DMA
Respecting NUMA topology, respecting memory layout, VA mode for IOVA, and using hugepages — these features stacked together naturally dictate that in DPDK's design, the underlying physical addresses behind every virtual address used by a userspace process are fixed and immutable — in other words, those addresses can be used for DMA. A DPDK userspace program need not get involved in IO transactions at all; it lets the hardware do the work autonomously, via DMA transactions on those fixed physical addresses.

## The Multi-Process Paradigm
DPDK also offers dedicated multi-process support, allowing one primary process to manage all DPDK resources while multiple secondary processes share access to them. DPDK's extra effort lies in guaranteeing that the addresses a secondary process sees are exactly the same as those seen by peer processes and the primary process — meaning even pointers can be passed across processes. That sounds rather dangerous, but the performance certainly beats all the safe communication and coordination mechanisms. In addition, DPDK supports cross-process global locks, making multi-process programming feel closer to multi-threaded programming.

[^1]: [Memory in DPDK Part 2: Deep Dive into IOVA](https://www.intel.com/content/www/us/en/developer/articles/technical/memory-in-dpdk-part-2-deep-dive-into-iova.html)
[^2]: An IOMMU is dedicated hardware sitting between the DMA-capable IO bus and main memory that maps devices' physical addresses into the virtual address space. Physical machines generally support an IOMMU; on Intel, for example, the IOMMU technology is VT-d: Intel® Virtualization Technology for Directed I/O.
[^3]: DIMM (dual in-line memory module), i.e., a RAM stick — the physical embodiment of DDR (Double Data Rate) technology.
[^4]: On both i686 and x86_64, the cache line size is 64B, though in some scenarios a sensible cache padding size is 128, because the prefetcher fetches two cache lines at a time.
[^5]: [Memory in DPDK Part 4: 18.11 and Beyond](https://www.intel.com/content/www/us/en/developer/articles/technical/memory-in-dpdk-part-4-1811-and-beyond.html)
[^6]: [Memory in DPDK Part 3: 17.11 and Earlier Releases](https://www.intel.com/content/www/us/en/developer/articles/technical/memory-in-dpdk-part-4-1811-and-beyond.html)
[^7]: [Memory in DPDK Part 1: General Concepts](https://www.intel.com/content/www/us/en/developer/articles/technical/memory-in-dpdk-part-1-general-concepts.html)
[^8]: The EAL accesses the HPET kernel time counter from userspace via `mmap`, exposing a high-precision timer interface to the service layer.
[^9]: The EAL uses `mmap` to allocate hugepage physical memory and exposes that physical memory to the service layer through the memory pool API.
