+++
title = "A Taxonomy of Stateful Distributed Systems"
date = "2021-07-08"
tags = ["Sys"]
description = "This post discusses the limitations of the CAP theorem and lays out a more fine-grained and precise taxonomy of stateful distributed systems, based on the trade-off between the two ideal properties of consistency and availability."
showFullContent = false
+++


## The CAP Theorem Has a Narrow Scope
In the field of distributed systems, the CAP theorem is widely cited, and it is often applied to problems beyond the boundaries of what it actually addresses.

The CAP theorem, as formally proven (see [Brewer's Conjecture and the Feasibility of Consistent, Available, Partition-​Tolerant Web Services](https://users.ece.cmu.edu/~adrian/731-sp04/readings/GL-cap.pdf)), is in fact confined to the read-write storage scenario: a storage system with only two operations, get and set(x). Such a system is called a register.

In an asynchronous network (meaning message delivery time is unbounded), a register cannot simultaneously satisfy all of the following properties:
1. Availability: every request sent to the register eventually completes. This differs from the definition used by most real systems, which do not require 100% of requests to complete — they only need a high enough SLA, and usually under some time constraint, returning a timeout error once the deadline passes.
2. Consistency: all read and write operations are linearizable: if operation B executes successfully after operation A, the system state seen by B cannot be older than the state at the time A completed.
3. Partition tolerance: packet loss in the network is allowed.

Partition tolerance is taken as given, so the choice is between CAP-availability and CAP-consistency.
Once we leave the formally proven setting, does the CAP theorem still have any guiding value? The answer is no — unless availability and consistency are redefined and generalized to more universal scenarios: from single-object, single-operation systems to transactional systems with multiple objects and multiple operations.
A taxonomy of distributed systems based on redefined availability and consistency may look very similar to the CAP theorem, but it cannot be called the CAP theorem.


## Consistency
The more general notion of "consistency" should be defined as "the visibility of shared-state updates in a concurrent system."
What modern microprocessors, distributed systems, and databases have in common is that they are all concurrent systems with shared data.
When we talk about consistency, we may mean the consistency models of microprocessor architecture and systems programming, or replica consistency in distributed systems, or transaction isolation in databases. These fields operate at different levels of abstraction; what they share is that the systems under discussion are all concurrent systems.

A consistency model describes, in multi-core concurrent scenarios in microprocessor architecture, the degree of reordering each processor is allowed — the fewer the constraints on reordering, the higher the efficiency, and the harder it is to guarantee the correctness of concurrent programs.
1. The strongest, strict consistency, means any write is immediately visible to any processor in any clock cycle; it obviously cannot be generalized to distributed systems.
2. Next is sequential consistency, which means the order of write operations is the same for every replica: the program order within each process is preserved, while the interleaving of operations from different processes may differ. This concept was also originally proposed by Lamport when discussing how a multi-processor computer can correctly execute concurrent programs. It has nothing to do with replica consistency in distributed systems. In C++, std::memory_order_seq_cst guarantees program order within a thread.
3. The looser causal consistency means the order of the subset of write operations that have dependency relationships is preserved, i.e., the dependency order within each process is consistent. Modern CPUs are essentially all out-of-order pipelines: as long as the bottom line of dependency order is preserved, they reorder as aggressively as possible. In C++, pairing a load(A) with std::memory_order_consume and a store(B) with std::memory_order_acquire guarantees that the part of all writes before this store that load(A) depends on is visible to load(A). If every dependency guarantees Release-Consume ordering, the dependency chain is ordered, and causal consistency holds overall.
4. Beyond these well-known models, there are dozens of consistency models applied in different approaches and fields. The figure below covers the various consistency models of non-transactional distributed storage systems (see [Consistency in Non-Transactional Distributed Storage Systems](https://arxiv.org/pdf/1512.00168.pdf) for details).

![Consistency1](https://wujipeng.com/img/1.png)

Concurrent programs can obviously be generalized quite easily to distributed replicated state machines — only network latency is added. Therefore, consistency models can be generalized to replica consistency in distributed systems. Take sequential consistency as an example: with a real-time constraint added, it becomes linearizability, which is more widely cited in the distributed systems field. It states that a single operation on a single replicated object satisfies: if A is a write operation, B is a read on a replica, and A happened-before (precedes in the causal sense, see https://en.wikipedia.org/wiki/Happened-before) B, then A's write is always visible to B's read. Compare this with the C++ definition of sequentially consistent ordering: everything that happened-before a store in one thread becomes a visible side effect in the thread that did a load. The two are equivalent.

Note that everything discussed so far is limited to a single operation on a single object replicated across different replicas. A distributed storage system cannot possibly store just one object, and many distributed stores support transactions or BatchWrite, which involve multiple operations on multiple objects. Generalizing the visibility of a single operation on a single object to multiple operations on multiple objects is not hard either — the ACID isolation levels of transactions are essentially a generalization of the visibility of a single operation on a single shared object to a group of operations on multiple objects. The left side of the figure below shows the isolation levels familiar from the database field. Just as in distributed systems, microprocessor architecture, and multi-core programming: the fewer the constraints on reordering, the higher the efficiency, and the harder it is to guarantee the correctness of concurrent programs.

![Consistency2](https://wujipeng.com/img/2.png)


## Availability
The more general notion of "availability" should be defined as "after imposing some constraint, the system can still eventually respond to every request, no matter how long a network partition lasts."
For a taxonomy of distributed systems more complete than the CAP theorem, see [Highly Available Transactions: Virtues and Limitations](https://arxiv.org/pdf/1302.0309.pdf).
This paper gives new definitions of availability:

1. High availability: every request a user sends to a running system eventually receives a reply, no matter how long a network partition lasts. This is the standard definition of CAP-availability, or traditional availability.
2. Sticky availability: whenever a user's transactions execute against a copy of a database state (one that reflects all of that user's previous operations), they eventually receive a reply, no matter how long a network partition lasts. This is a stronger requirement than CAP-availability.
  - If you only pursue high availability, a user can access any replica in the system, and it doesn't matter if different operations are served by different replicas; but with sticky availability, the user must ensure that a consecutive series of operations always goes to the same replica. For example, in a multi-writer distributed store like Dynamo, you cannot write to node A for a while and then switch to node B.
3. Transactional availability: consistency models in the distributed systems literature mostly consider single operations on a single object, whereas the database literature focuses on transactions: multiple operations on multiple objects grouped together as one transaction. Clearly, the CAP-availability definition does not apply to transactions either.
  - Replica availability for transactions: a transaction can reach at least one replica for every object it needs to access. This requirement is weaker than CAP-availability.
  - Liveliness of transactions: suppose we abort every transaction — then we could guarantee 100% timely responses and perfectly achieve CAP-availability, but what would be the point? So we also need to ensure that transactions commit rather than abort as much as possible.
  - Therefore, the resulting definition of transactional availability is: for every piece of data in a transaction, replica availability is guaranteed, and the transaction can eventually commit successfully within N retries, or internal abort (an abort actively chosen by the transaction itself, rather than one imposed by the system implementation).
  - Going one step further, we can define sticky transactional availability: if a system can guarantee sticky availability, then it can guarantee transactional availability.

With these definitions, we can compare the consistency (isolation levels) of existing transactional systems against their availability, yielding the results in the figure below: in the new taxonomy, it holds that the higher the availability requirement, the looser the consistency requirement.

![Availability](https://wujipeng.com/img/3.png)
