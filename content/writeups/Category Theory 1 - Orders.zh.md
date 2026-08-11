+++
title = "CAT01: Orders"
date = "2024-02-26"
tags = ["En", "Math"]
description = "CAT 系列第 1 篇，以序理论（order theory）为完整的范畴论做热身。内容涵盖 preorder、meet/join、monotone map 与 Galois connection。"
showFullContent = false
+++

## 预序（Preorders）
从 `sets` 和 `subsets` 出发，我们可以把 $A$ 与 $B$ 之间的 `relation` 定义为一个 `subset` $R \in A\times B$。

每个 `function` 都是一个 relation，满足 2 个性质：
1. $\forall a \in A$，存在 $b \in B$，使得 $(a,b)\in \mathbb{R}$
2. $\forall a, b_1, b_2$，若 $(a,b_1) \in R$ 且 $(a, b_2) \in \mathbb{R}$，则 $b_1 = b_2$。

> 序（order）、等价（equivalence）、容忍（tolerance）都是 relation。

`function` $f: A\rightarrowtail B$ 称为 `injection`（单射），如果 $\forall a_1, a_2, b$，当 $(a_1, b), (a_2, b) \in R$ 时，有 $a_1 = a_2$。

`function` $f: A\twoheadrightarrow B$ 称为 `surjection`（满射），如果 $\forall b \in B$，存在一个 $a \in A$，使得 $f(a) = b$。

`function` $f: A \overset{\cong} \rightarrowtail B$ 称为 `bijection`（双射），如果它既是满射又是单射。

集合 $X$ 上的 `identity function` 记作 $id_X$，即双射函数 $id_X(x) = x$。

$A$ 上的一个 `partition`（划分）无非就是到另一个集合 $P$ 的满射：$A \twoheadrightarrow P$。

我们可以给 `partitions` 排序：$A \twoheadrightarrow P_1$，$A \twoheadrightarrow P_2$。如果存在一个 `function` $P_1 \rightarrow P_2$ 使下图交换：$A \twoheadrightarrow P_1 \rightarrow P_2$，则称 $P_1 \leqslant P_2$。

因此 $A \twoheadrightarrow A$ 是最小的 `partition`，而 $A \twoheadrightarrow \underline{1}$ 是最大的 `partition`。

`preorder`（预序）是
1. 一个集合 $S$，以及
2. 一个关系 $≤ \: \subseteq S \times S$[^1]

并满足 2 个性质：
1. $S ≤ S$。[^2]
2. $\forall S_1$、$S_2$、$S_3$，若 $S_1 ≤ S_2$ 且 $S_2 ≤ S_3$，则 $S_1 ≤ S_3$。[^3]

> `preorder` 就是任意两个对象之间至多只有一个 `morphism` 的 `category`。稍微复杂一点的说法是：`preorder` 是一种 `Bool-enriched category`。

## Meet 与 Join
序（order）催生了 `meets` 和 `joins`。

将 $A$ 与 $B$ 做 join，记作 $A\vee B$，得到的是同时大于 $A$ 和 $B$ 的最小 `partition`。即 $A ≤ (A\vee B)$ 且 $A ≤ (A\vee B)$。并且对任意 C，若 $A ≤ C$ 且 $B ≤ C$，则 $(A\vee B) ≤ C$。

形式化地说。设 $(P, ≤)$ 是一个 `preorder`，$A \subseteq P$ 是一个 `subset`。若元素 $p \in P$ 满足以下条件，则称 $p$ 为 $A$ 的一个 `meet`：
1. $\forall a\in A$，都有 $p≤ a$[^4]，并且
2. 对所有满足"对任意 $a\in A$ 都有 $q≤ a$"的 $q$，都有 $q≤ p$[^5]。

记作 $P = \wedge A =  \underset{a\in A}{\wedge} a = a_1 \wedge a_2 \wedge ... \wedge a_n$

类似地，若满足以下条件，则称 $p$ 为 $A$ 的一个 `join`：
1. $\forall a\in A$，都有 $a≤ p$[^6]，并且
2. 对所有满足"对任意 $a\in A$ 都有 $a≤ q$"的 $q$，都有 $p≤ q$[^7]。

此时记作 $P = \vee A =  \underset{a\in A}{\vee} a = a_1 \vee a_2 \vee ... \vee a_n$

接下来讨论一些例子。

**例 1：布尔值 $\mathbb{B}$=\{T,F\}$ 的真值表。**

例如，$\{T,F\}(F \leq T)$ 的两两 `meets` 表恰好就是初等逻辑中 `AND` 的真值表。类似的二元 join 计算会生成 `OR` 的真值表。

**例 2：幂集 $(P(x), ≤)$。**

取一个具体的集合，令 $X = \{\square, \times, \heartsuit\}$，然后考虑它的幂集：

![幂集](https://jipeng4974.github.io/img/power_sets.png)

在这种情况下，显然 $\wedge$ = 交集（intersection），$\vee$ = 并集（union）。

**例 3：$(\mathbb{N}, |), a ≤ b$ 当且仅当 $a|b$**

![整除关系](https://jipeng4974.github.io/img/divisible.png)

1 整除所有数，所以我们从 1 开始。这里 $\wedge$ = gcd，$\vee$ = lcm。

**例 4：`meet`/`join` 可能不止一个。**
![Hasse 图](https://jipeng4974.github.io/img/hasse_diagram.png)

这张 Hasse 图给出了一个 `preorder`，其中 $c$ 和 $d$ 都是 $A$ 的 `meets`。我们有 $c≤ d$ 且 $d≤ c$，所以 $c \cong d$，即 $c$ 与 $d$ 是 `isomorphic`（同构）的，这一点后文会讲到；不过一般来说，我们把它们当作相等也不会遇到什么麻烦。

**例 5：`meet` 或 `join` 可能不存在。**

显然，事物并不总是有下界或上界。也可能存在多个下界/上界，但这些下界/上界之间不可比较。

纵观这些例子，许多熟悉的事物——无论是 gcd/lcm、max/min、limits/colimits，还是 intersection/union——都可以用这个相当简单的泛性质（universal property）来刻画。

事物可以由泛性质来刻画，这一想法意味着我们现在离范畴论更近了。

## 单调映射（Monotone Maps）
`Preorders` 自身之间也可以相互关联。`monotone map` 就是 `preorders` 之间保持结构的映射。

`preorders` $(A, ≤_A)$ 与 $(B, ≤_B)$ 之间的 `monotone map` 是一个 `function` $f : A \rightarrow B$，使得对所有元素 $x, y ∈ A$，若 $x ≤_A y$，则 $f (x) ≤_B f(y)$。
![单调映射](https://jipeng4974.github.io/img/monotone_map.png)

设 $\mathbb{B}$ 为布尔值的 preorder，$\mathbb{N}$ 为自然数的 preorder。把 false 映到 17、把 true 映到 24 的映射 $\mathbb{B} \rightarrow \mathbb{N}$ 是一个 `monotone map`，因为它保持了序。

![b2n](https://jipeng4974.github.io/img/b2n.png)

若 $\forall p,p' \in P, f(p\vee p') = f(p) \vee f(p')$，则称 `monotone map` $f: P \rightarrow Q$ 保持 `joins`。

对任意 `preorder` $(P, ≤_P)$，恒等函数都是单调的。
若 $(Q, ≤_Q)$ 和 $(R, ≤_R)$ 是 preorders，且 $f : P → Q$ 与 $g : Q → R$ 都是单调的，则 $(f ; g): P → R$ 也是单调的。

设 $(P, ≤_P)$ 与 $(Q, ≤_Q)$ 为 preorders。若存在一个 `monotone function` $g : Q → P$ 使得 $f;g = id_P$ 且 $g;f = id_Q$，则称 `monotone function` $f : P → Q$ 为一个 `isomorphism`（同构）。这意味着对任意 $p ∈ P$ 和 $q ∈ Q$，都有 $p=g(f(p))$ 且 $q 
=f(g(q))$。

我们称 $g$ 为 $f$ 的逆，反之亦然：$f$ 是 $g$ 的逆。

若存在一个 `isomorphism` $P → Q$，则称 $P$ 与 $Q$ 是同构的（isomorphic）。

> preorders 之间的同构，基本上只是对元素的重新标记（relabeling）。

## Galois 连接（Galois Connections）
`preorders` $P$ 与 $Q$ 之间的 `Galois connections` 是一对 `monotone maps` $f : P → Q$ 和 $g : Q → P$，满足 $f(p) ≤ q iff p ≤ g(q).$

我们称 $f$ 为左 `adjoint`（左伴随），g 为该 `Galois connection` 的右 `adjoint`（右伴随）。

> `Galois connections` 的理论是一个更一般的理论——`adjunctions`（伴随）理论——的特例。

**例 1：$P = Q = \underline{3}$**
![togc](https://jipeng4974.github.io/img/Galois_connections.png)

此时 $P$ 和 $Q$ 都是全序（total order），只要箭头不交叉，$f$ 就是 $g$ 的左伴随。

**例 2：$\mathbb{Z} \xrightarrow[f]{3\times\square} \mathbb{R}$，$\mathbb{R} \xrightarrow[g]{\lfloor\square/3\rfloor} \mathbb{Z}$**

也就是说我们有 $5 \xrightarrow[f]{3\times\square}15$，$13.3 \xrightarrow[g]{\lfloor\square/3\rfloor}4$。

由于 $3n ≤ x$ 当且仅当 $n ≤ \lfloor x/3\rfloor$，所以 $f$ 是 $g$ 的左伴随。

归根结底，我们可以断言：`monotone map` $f$ 是左/右 `adjoint`，当且仅当它保持 `joins`/`meets`。




[^1]: 这里我们用符号 $≤$ 而不是 $R$，因为它暗示了一个 preorder，而且中缀记法 $S_1 ≤ S_2$ 看起来比 $(S_1,S_2) \in R$ 更自然。
[^2]: 自反性（reflexivity）
[^3]: 传递性（transitivity）
[^4]: $p$ 是 $A$ 的下界（lower bound）
[^5]: $p$ 是最大下界（greatest lower bound）
[^6]: $p$ 是 $A$ 的上界（upper bound）
[^7]: $p$ 是最小下界（least lower bound）
