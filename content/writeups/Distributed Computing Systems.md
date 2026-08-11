+++
title = "Distributed Computing Systems At Scale"
date = "2023-06-07"
tags = ["sys"]
description = "Messy distributed phenomena and tedious engineering practice can easily obscure the essence of distributed computing systems, so here is a systematic overview."
showFullContent = false
+++
# Distributed Computing Systems At Scale
Once a distributed computing system makes the leap to scale, its core challenges are correctness and system efficiency. The former corresponds to consensus abstraction, the latter to performance engineering.

## Consensus Abstraction
The correctness of a distributed system actually subsumes ideal properties such as stability, consistency, and availability.
Distribution inherently means concurrent execution, process failures, and unreliable message delivery — all of which make correctness extremely hard in common distributed application contexts:
- Concurrency means the execution-trace space explodes dimensionally, proportional to the factorial of (number of processes × number of steps). It is very hard to guarantee that every execution path is correct.
- Process failures can be classified into: crashes (the process is gone — NIC or CPU failure), dropped requests (no crash, but network congestion or service degradation causes temporary unavailability), crash-after-recovery (repeated restarts before eventually recovering to a correct state, usually based on logs), and Byzantine failures (unpredictable, arising from cosmic rays or malicious attacks).
- Message delivery faces risks such as tampering, packet loss, retransmission, and reordering; even reliable network protocols are not completely reliable.

Consensus means that multiple processes eventually reach a common decision on some problem (the 2-general problem, the global ordering guaranteed by a replicated state machine, distributed transactions). An ideal consensus algorithm should have safety and liveness. Safety includes validity (only a proposed value may be chosen), agreement (correct nodes stay consistent), and integrity (each node chooses at most once). Liveness means termination: every correct node eventually chooses some value instead of hanging.

If safety is set aside, TCP is actually the simplest, crudest, yet effective consensus protocol: at its core, TCP lets the two communicating processes reach consensus on the state of the connection so they can communicate reliably. For example, when a process closes a connection, it sets its local state to TIME_WAIT and waits a few minutes before truly releasing resources, to make sure the peer receives the ACK — using a simple wait to achieve a high-probability consensus, avoiding errors from receiving stale packets after both sides have started the next connection.

If full safety is required, the simple case is negotiating and deciding on a single value — single-value consensus.

The simplest scenario under single-value consensus assumes perfect failure detection; flooding consensus (decide only after collecting proposals from all processes) and uniform hierarchical consensus are simple solutions for this scenario.

With only eventual failure detection, the notion of epoch change must be introduced, and the problem is solved with epoch-based consensus (usually a Paxos variant). Paxos is the only known completely-safe & largely-live fault-tolerant consensus algorithm. 2PC is also a form of epoch-based consensus, except that in 2PC a coordinator crash blocks the whole system — the classic example of safe but not live. Paxos can even implement multi-dimensional successive epochs by changing the partial order on rounds from < to divisibility, thereby also covering the functionality of 2PC.

If we further generalize single-value consensus to sequence consensus (more common in distributed storage, because writes to a log-structured store are an append-only sequence rather than a single value), Paxos evolves into some form of Multi-Paxos, or Raft. Raft essentially solves the replicated state machine problem, and the sequence consensus problem happens to reduce to the replicated state machine problem.

If Byzantine failures are also considered, then BFT, PBFT, or PoW are required.

## Performance Engineering
The importance of performance is beyond doubt: whether the latency of on-device applications (speech recognition, intelligent customer service) is low enough directly determines whether a project is viable at all, while the throughput of large-scale IDC applications (search, ads and recommendation, storage, online inference, training) is directly tied to money and environmental impact.

The performance of computing systems is inseparable from distribution, because almost every computing system is distributed — including a single machine, a single GPU, or a single device. A single machine (in-node) still has a network structure, such as NVLink and NVSwitch; even a Xeon CPU is many cores connected by a bus, and devices like NICs and SSDs, once opened up, are all networked devices. The methods for gaining performance through partitioning, parallelism, hardware-software co-design, and removing unnecessary layers are universal.

If a baseline exists, the main work of performance engineering is optimization, iteration, and incremental innovation.
When the baseline is not good enough, performance engineering is divergent: it spans layers — bottlenecks can appear at any abstraction level, in any module, so there should be no optimization blind spots; when necessary, de-layering is required, such as moving the entire IO path into userspace, or exposing hypervisor state to the VM for efficient scheduling.
When the baseline is already very good, the performance path converges: it points clearly at hardware-software co-design. After all, performance ultimately comes from tailoring to hardware resources and squeezing them to the limit; if co-design is also good enough, the only direction left is hardware upgrades: bigger and stronger NVIDIA cards, DSAs like TPUv4, optical routing like OCS, and further AI DC build-out.

If the problem is brand new, or someone gains an extraordinary insight into an existing problem and imposes a more effective structure on it, a new architecture — a new form of computation and IO — is born:
- After the advent of the large-model era, the model parameters alone no longer fit into a top-tier GPU, let alone the extra memory required for neural network computation. This challenged the Parameter Server architecture, which had struck a very good balance between system efficiency and correctness, and forced entirely new architectures and strategies for large-model training: weight sharding and model parallelism within a Node (8 GPUs forming one Node), data and instruction parallelism within operators, subgraph partitioning, operator placement and pipeline parallelism (8 Nodes forming an 8-stage pipeline), and data parallelism with Batch16 across 16 groups of 8×8 — which produced the Alpa-over-Ray solution for training large models. This in turn indirectly caused bandwidth demand to surge, to the point of requiring network solutions like Google's optical circuit switch OCS, which avoids the overhead of optical-electrical modules and packet-parsing computation — the downside of optical switching is slow reconfiguration, but during a training cycle the main traffic routes for data exchange are fixed, so building an AI DC with it neatly sidesteps this weakness.
- Based on experience with hypervisor performance bottlenecks on NUMA systems, the author of KVM readily saw that the application-layer shared-memory parallel model is flawed — especially severe in the many-core era — and therefore created seastar. Exploiting hardware locality and avoiding thread switching, data copies, and NUMA remote memory access is a critical performance path for memory-access-intensive, CPU-bound applications.
