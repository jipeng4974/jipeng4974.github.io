+++
title = "Dash: 可扩展哈希"
date = "2024-01-26"
tags = ["Sys", "En", "Perf"]
description = "Dash 论文的主要关注点是曾经风靡一时的 `persistent memory`，但实际上，任何受 `memory bandwidth` 限制的场景都能从中受益。随着 Intel 砍掉其 `pmem` 业务，`Dash` 方案的意义已经转移到了普通的 `DRAM` 应用上。"
showFullContent = false
+++

Dash 论文的主要关注点是曾经风靡一时的 `persistent memory`，但实际上，任何受 `memory bandwidth` 限制的场景都能从中受益。随着 Intel 砍掉其 `pmem` 业务，`Dash` 方案的意义已经转移到了普通的 `DRAM` 应用上。

# 动态哈希
论文提出的可扩展哈希表 `Dashtable`，由 `extendible hashing` 演化而来。

`Extendible hashing` 是一种哈希体系，它使用哈希值的前 $N$ 位，在 trie 结构的 `directory` 中查找 bucket。

`global depth` 为 $N$ 的 `directory` 可以容纳 $2^N$ 个 bucket。这意味着 $N$ 是映射该 `directory` 的键长。

每个 bucket 还有一个 `local depth` $M(M \le N)$，即此前映射该 `directory` 的键长。`local depth` 为 $M = N$ 的 bucket 恰好被一个 `directory` 条目指向；`local depth` 为 $M \lt N$ 的 bucket 则被多个 `directory` 条目指向。

要保证每个 item 都有唯一的 bucket 索引，2 个 item 所需的最小 $N$ 为 1，4 个 item 所需的最小 $N = 2$。每当有新 item 加入 bucket，如果 bucket 中的 item 数量超过某个阈值，就会触发一次 rehashing：将该 bucket 一分为二。因此，这种方案中的 rehashing 不需要 stop the world、做全表扫描和拷贝，而是增量完成的。

与 `extendible hashing` 类似，`linear hashing` 同样使用 `directory` 来组织和寻址 bucket。区别在于 split 的控制方式。在 `linear hashing` 中，通常只有当 load factor 超过阈值时才会发生 split，且待 split 的 bucket 是以“线性”的方式选出的。

# 面向 Extendible Hashing 的 Dash
## 概览
![dash_eh](https://jipeng4974.github.io/img/dash_eh.png)

在 `Dash-EH` 中，每个 `directory` 条目指向一个 `segment`，它由固定数量的普通 bucket 和 stash bucket 组成。一个 `segment` 可以看作一个大小恒定的子哈希表。所谓 stash bucket 与普通 bucket 布局相同，负责存储溢出记录。

![dash_eh](https://jipeng4974.github.io/img/dash_eh_bucket.png)

`Dash-EH` 的核心思想是在元数据上多花一点空间，换来基于 fingerprint 的更快 probing，以及基于 version lock 的轻量级并发控制。

如上图所示，在一个 `Dash-EH` bucket 内部，前 32 字节是元数据，包括 version lock、计数器、alloc bitmap、用于负载均衡的 membership bitmap，以及 18 个用于 bucket probing 的单字节 fingerprint（其中 14 个对应 bucket 内的 slot，4 个对应原本哈希到此 bucket 的溢出记录）。紧随其后的是 $16(Bytes) \times 14 (records) = 224 Bytes$ 的 payload，存放 14 条 16 字节的记录。

## Fingerprinting
Bucket probing 指在 bucket 中查找某个 slot，是哈希表的基本操作，`search`、`insert` 和 `delete` 都需要它来定位特定的 key。传统的 probing 需要线性扫描，这在 `PMem` 上天然就慢，而且当被查找的 key 不存在时，扫描可能完全是多余的。`Dash-EH` 采用 fingerprinting 来减少不必要的扫描。fingerprint 是 key 的哈希值的最低一个字节。probing 线程在查找某个 key 时，会先检查 bucket 元数据中是否有与该 key 匹配的 fingerprint，从而跳过没有任何 fingerprint 匹配的 bucket。

Fingerprinting 主要惠及 negative（key 不存在）的 `search`。`Dash` 论文还声称，fingerprinting 使得使用跨越 2 条以上 cacheline 的更大 bucket 成为可能。但对此我得持保留态度。论文自己用的是 256B 的设置，DragonFly 的实现[^1] 也是如此。理论上，更大的 bucket 确实能容忍更多冲突、提高 load factor；然而，这可能以在一定程度上牺牲 locality 为代价——在哈希表中，你不会希望访问一次 bucket 要加载多次。

## Bucket 负载均衡
Segmentation 通过减小 `directory` 的体积来降低其 cache miss。在 `extendible hashing` 方案中，如果 segment 里任意一个 bucket 满了，整个 segment 都需要 split，即使其他 bucket 可能还有很多空闲空间。

为了避免频繁的 segment split，`Dash-EH` 的算法设计引入了 bucket 负载均衡。对于 `insert` 操作，`Dash-EH` 会同时探测 bucket $B_b$ 和 $B_{b+1}$，然后插入较不满的那个。如果 $B_b$ 和 $B_{b+1}$ 都满了，`Dash-EH` 会尝试把 $B_{b+1}$ 中的一条“native record”挪到 $B_{b+2}$，或者把 $B_b$ 中的一条“rebalanced record”挪回它原本所属的 $B_{b-1}$。

每个 bucket 的 membership bitmap 用于判断一条记录是 rebalanced 还是 native。如果 membership bitmap 中某一位被置位，那么对应的 key 并非直接哈希进这个 bucket（native），而是由于重均衡被放到这里的（rebalanced）。

如果 `insert` 和 displacement 都失败了，`Dash-EH` 就使出最后的手段——stashing。每个 segment 有固定数量的 stash bucket 来容纳这些溢出记录。探测 stash bucket 会给 negative `search` 和 `insert`（需要做唯一性检查）带来显著开销。为了解决这个问题，每个普通 bucket 预留了若干元数据字段：为存放在 stash bucket 中的溢出记录保留 4 个溢出 fingerprint；一个溢出 bit 表示是否存在溢出。这样，如果一个 bucket 没有溢出，`search`/`insert` 操作就不必探测 stash bucket。不过，保持少量的 stash bucket 仍然是明智的。论文声称“每个 segment 使用 2–4 个 stash bucket，可以将 load factor 提升到 90% 以上，而不会带来显著开销”。在 Dragonfly 的 Dashtable 中，每个 segment 有 56 个普通 bucket 和 4 个 stash bucket。

## 轻量级并发控制
`Dash-EH` 的轻量级并发控制在当今的 `many-core` 架构上天然具有良好的扩展性，性能优于传统的 bucket 级共享锁。

写操作沿用传统的 bucket 级加锁方式，通过对一个 lock bit 执行 CAS 来锁住受影响的 bucket。写完成后，写线程复位 lock bit，并将该 bucket 的版本号加一。

另一方面，读操作被设计为 lock-free 的。读之前，读线程先获取 lock word 的快照，等待锁被释放，然后在不持有任何锁的情况下继续读取。读完后，它会再次检查 lock word，确认版本号保持不变。如果版本号变了，就重试整个操作。

# 面向 Linear Hashing 的 Dash
Dash 论文还提出了 `Dash-LH`，一种支持 Dash 的 linear hashing 方案，构建于 `Dash-EH` 所用的构建块之上，比如均衡的 `insert`/`displacement`、fingerprinting 和乐观并发——毕竟它们大体上是正交的。主要区别在于，`Dash-LH` 以线性的方式 split 指针所指向的 segment。

传统的 `linear hashing` 用链表（linklist）把溢出记录串起来。在 `Dash-LH` 中，改用 stash bucket，对 cache 更友好。不过它仍需要把这些 stash bucket 串成链，但这仍然比把单条记录串成链好得多。



[^1]: [Dragonfly 中的 Dashtable](https://github.com/dragonflydb/dragonfly/blob/main/docs/dashtable.md)
