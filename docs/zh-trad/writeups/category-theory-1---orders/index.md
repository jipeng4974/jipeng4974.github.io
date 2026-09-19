# CAT01: Orders

> CAT 系列第 1 篇，以序理論（order theory）爲完整的範疇論做熱身。內容涵蓋 preorder、meet/join、monotone map 與 Galois connection。

---

LLMS index: [llms.txt](/llms.txt)

---

## 預序（Preorders）
從 `sets` 和 `subsets` 出發，我們可以把 $A$ 與 $B$ 之間的 `relation` 定義爲一個 `subset` $R \in A\times B$。

每個 `function` 都是一個 relation，滿足 2 個性質：
1. $\forall a \in A$，存在 $b \in B$，使得 $(a,b)\in \mathbb{R}$
2. $\forall a, b_1, b_2$，若 $(a,b_1) \in R$ 且 $(a, b_2) \in \mathbb{R}$，則 $b_1 = b_2$。

> 序（order）、等價（equivalence）、容忍（tolerance）都是 relation。

`function` $f: A\rightarrowtail B$ 稱爲 `injection`（單射），如果 $\forall a_1, a_2, b$，當 $(a_1, b), (a_2, b) \in R$ 時，有 $a_1 = a_2$。

`function` $f: A\twoheadrightarrow B$ 稱爲 `surjection`（滿射），如果 $\forall b \in B$，存在一個 $a \in A$，使得 $f(a) = b$。

`function` $f: A \overset{\cong} \rightarrowtail B$ 稱爲 `bijection`（雙射），如果它既是滿射又是單射。

集合 $X$ 上的 `identity function` 記作 $id_X$，即雙射函數 $id_X(x) = x$。

$A$ 上的一個 `partition`（劃分）無非就是到另一個集合 $P$ 的滿射：$A \twoheadrightarrow P$。

我們可以給 `partitions` 排序：$A \twoheadrightarrow P_1$，$A \twoheadrightarrow P_2$。如果存在一個 `function` $P_1 \rightarrow P_2$ 使下圖交換：$A \twoheadrightarrow P_1 \rightarrow P_2$，則稱 $P_1 \leqslant P_2$。

因此 $A \twoheadrightarrow A$ 是最小的 `partition`，而 $A \twoheadrightarrow \underline{1}$ 是最大的 `partition`。

`preorder`（預序）是
1. 一個集合 $S$，以及
2. 一個關係 $≤ \: \subseteq S \times S$[^1]

並滿足 2 個性質：
1. $S ≤ S$。[^2]
2. $\forall S_1$、$S_2$、$S_3$，若 $S_1 ≤ S_2$ 且 $S_2 ≤ S_3$，則 $S_1 ≤ S_3$。[^3]

> `preorder` 就是任意兩個對象之間至多隻有一個 `morphism` 的 `category`。稍微複雜一點的說法是：`preorder` 是一種 `Bool-enriched category`。

## Meet 與 Join
序（order）催生了 `meets` 和 `joins`。

將 $A$ 與 $B$ 做 join，記作 $A\vee B$，得到的是同時大於 $A$ 和 $B$ 的最小 `partition`。即 $A ≤ (A\vee B)$ 且 $A ≤ (A\vee B)$。並且對任意 C，若 $A ≤ C$ 且 $B ≤ C$，則 $(A\vee B) ≤ C$。

形式化地說。設 $(P, ≤)$ 是一個 `preorder`，$A \subseteq P$ 是一個 `subset`。若元素 $p \in P$ 滿足以下條件，則稱 $p$ 爲 $A$ 的一個 `meet`：
1. $\forall a\in A$，都有 $p≤ a$[^4]，並且
2. 對所有滿足"對任意 $a\in A$ 都有 $q≤ a$"的 $q$，都有 $q≤ p$[^5]。

記作 $P = \wedge A =  \underset{a\in A}{\wedge} a = a_1 \wedge a_2 \wedge ... \wedge a_n$

類似地，若滿足以下條件，則稱 $p$ 爲 $A$ 的一個 `join`：
1. $\forall a\in A$，都有 $a≤ p$[^6]，並且
2. 對所有滿足"對任意 $a\in A$ 都有 $a≤ q$"的 $q$，都有 $p≤ q$[^7]。

此時記作 $P = \vee A =  \underset{a\in A}{\vee} a = a_1 \vee a_2 \vee ... \vee a_n$

接下來討論一些例子。

**例 1：布爾值 $\mathbb{B}$=\{T,F\}$ 的真值表。**

例如，$\{T,F\}(F \leq T)$ 的兩兩 `meets` 表恰好就是初等邏輯中 `AND` 的真值表。類似的二元 join 計算會生成 `OR` 的真值表。

**例 2：冪集 $(P(x), ≤)$。**

取一個具體的集合，令 $X = \{\square, \times, \heartsuit\}$，然後考慮它的冪集：

![冪集](https://wujipeng.com/img/power_sets.png)

在這種情況下，顯然 $\wedge$ = 交集（intersection），$\vee$ = 並集（union）。

**例 3：$(\mathbb{N}, |), a ≤ b$ 當且僅當 $a|b$**

![整除關係](https://wujipeng.com/img/divisible.png)

1 整除所有數，所以我們從 1 開始。這裏 $\wedge$ = gcd，$\vee$ = lcm。

**例 4：`meet`/`join` 可能不止一個。**
![Hasse 圖](https://wujipeng.com/img/hasse_diagram.png)

這張 Hasse 圖給出了一個 `preorder`，其中 $c$ 和 $d$ 都是 $A$ 的 `meets`。我們有 $c≤ d$ 且 $d≤ c$，所以 $c \cong d$，即 $c$ 與 $d$ 是 `isomorphic`（同構）的，這一點後文會講到；不過一般來說，我們把它們當作相等也不會遇到什麼麻煩。

**例 5：`meet` 或 `join` 可能不存在。**

顯然，事物並不總是有下界或上界。也可能存在多個下界/上界，但這些下界/上界之間不可比較。

縱觀這些例子，許多熟悉的事物——無論是 gcd/lcm、max/min、limits/colimits，還是 intersection/union——都可以用這個相當簡單的泛性質（universal property）來刻畫。

事物可以由泛性質來刻畫，這一想法意味着我們現在離範疇論更近了。

## 單調映射（Monotone Maps）
`Preorders` 自身之間也可以相互關聯。`monotone map` 就是 `preorders` 之間保持結構的映射。

`preorders` $(A, ≤_A)$ 與 $(B, ≤_B)$ 之間的 `monotone map` 是一個 `function` $f : A \rightarrow B$，使得對所有元素 $x, y ∈ A$，若 $x ≤_A y$，則 $f (x) ≤_B f(y)$。
![單調映射](https://wujipeng.com/img/monotone_map.png)

設 $\mathbb{B}$ 爲布爾值的 preorder，$\mathbb{N}$ 爲自然數的 preorder。把 false 映到 17、把 true 映到 24 的映射 $\mathbb{B} \rightarrow \mathbb{N}$ 是一個 `monotone map`，因爲它保持了序。

![b2n](https://wujipeng.com/img/b2n.png)

若 $\forall p,p' \in P, f(p\vee p') = f(p) \vee f(p')$，則稱 `monotone map` $f: P \rightarrow Q$ 保持 `joins`。

對任意 `preorder` $(P, ≤_P)$，恆等函數都是單調的。
若 $(Q, ≤_Q)$ 和 $(R, ≤_R)$ 是 preorders，且 $f : P → Q$ 與 $g : Q → R$ 都是單調的，則 $(f ; g): P → R$ 也是單調的。

設 $(P, ≤_P)$ 與 $(Q, ≤_Q)$ 爲 preorders。若存在一個 `monotone function` $g : Q → P$ 使得 $f;g = id_P$ 且 $g;f = id_Q$，則稱 `monotone function` $f : P → Q$ 爲一個 `isomorphism`（同構）。這意味着對任意 $p ∈ P$ 和 $q ∈ Q$，都有 $p=g(f(p))$ 且 $q 
=f(g(q))$。

我們稱 $g$ 爲 $f$ 的逆，反之亦然：$f$ 是 $g$ 的逆。

若存在一個 `isomorphism` $P → Q$，則稱 $P$ 與 $Q$ 是同構的（isomorphic）。

> preorders 之間的同構，基本上只是對元素的重新標記（relabeling）。

## Galois 連接（Galois Connections）
`preorders` $P$ 與 $Q$ 之間的 `Galois connections` 是一對 `monotone maps` $f : P → Q$ 和 $g : Q → P$，滿足 $f(p) ≤ q iff p ≤ g(q).$

我們稱 $f$ 爲左 `adjoint`（左伴隨），g 爲該 `Galois connection` 的右 `adjoint`（右伴隨）。

> `Galois connections` 的理論是一個更一般的理論——`adjunctions`（伴隨）理論——的特例。

**例 1：$P = Q = \underline{3}$**
![togc](https://wujipeng.com/img/Galois_connections.png)

此時 $P$ 和 $Q$ 都是全序（total order），只要箭頭不交叉，$f$ 就是 $g$ 的左伴隨。

**例 2：$\mathbb{Z} \xrightarrow[f]{3\times\square} \mathbb{R}$，$\mathbb{R} \xrightarrow[g]{\lfloor\square/3\rfloor} \mathbb{Z}$**

也就是說我們有 $5 \xrightarrow[f]{3\times\square}15$，$13.3 \xrightarrow[g]{\lfloor\square/3\rfloor}4$。

由於 $3n ≤ x$ 當且僅當 $n ≤ \lfloor x/3\rfloor$，所以 $f$ 是 $g$ 的左伴隨。

歸根結底，我們可以斷言：`monotone map` $f$ 是左/右 `adjoint`，當且僅當它保持 `joins`/`meets`。




[^1]: 這裏我們用符號 $≤$ 而不是 $R$，因爲它暗示了一個 preorder，而且中綴記法 $S_1 ≤ S_2$ 看起來比 $(S_1,S_2) \in R$ 更自然。
[^2]: 自反性（reflexivity）
[^3]: 傳遞性（transitivity）
[^4]: $p$ 是 $A$ 的下界（lower bound）
[^5]: $p$ 是最大下界（greatest lower bound）
[^6]: $p$ 是 $A$ 的上界（upper bound）
[^7]: $p$ 是最小下界（least lower bound）
