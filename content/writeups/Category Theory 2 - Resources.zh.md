+++
title = "CAT02: Resources"
date = "2024-06-14"
tags = ["en", "math"]
description = "第二篇 CAT 笔记讨论资源的范畴论形式化，以及如何将一组资源转化为另一组资源。内容涵盖 monoidal preorder（幺半预序）、wiring diagram（接线图）、monoidal monotone map 和 V-category。"
showFullContent = false
+++

## 对称幺半预序

**定义 2.1:** 预序 $(X,≤)$ 上的一个 `对称幺半结构`（symmetric monoidal structure）由两个成分构成：(i) 一个元素 $I \in X$，称为 `幺半单位`（monoidal unit）；(ii) 一个函数 $\otimes: X \times X \rightarrow X$，称为 `幺半积`（monoidal product）。这些成分必须满足以下性质：
- 单调性（monotonicity）：$\forall x_1, x_2, y_1, y_2 \in X$，若 $x_1 \le y_1$ 且 $x_2 \le y_2$，则 $x_1 \otimes x_2 \le y_1 \otimes y_2$。
- 单位性（unitality）：$\forall x \in X$，$I \otimes x = x$ 且 $x \otimes I = x$ 成立。
- 结合律（associativity）：$\forall x,y,z \in X$，$(x\otimes y)\otimes z = x \otimes (y\otimes z)$。
- 对称性（symmetry）：$\forall x,y \in X, x\otimes y = y\otimes x$。[^1]

**定义 2.2:** 配备了对称幺半结构的预序 $(X,\le,I,\otimes)$ 称为 `对称幺半预序`（symmetric monoidal preorder）。

**例 2.1（布尔值）:** $\mathbb{B} = \{true, false\}$ 配合 $false < true$ 是最简单的非平凡预序。我们可以定义幺半单位为 true，幺半积为 $\wedge$（与，AND）。这样我们就得到了一个幺半预序，记作 $Bool := (\mathbb{B}, \le ,true, \wedge )$。

## 接线图
`接线图`（wiring diagram）是从旧关系构建新关系的可视化表示。在没有幺半结构的预序中，关系是串联起来的。
![接线图 1](https://jipeng4974.github.io/img/wiring_diagrams_1.png)

有了对称幺半结构，关系也可以并联排列。
![接线图 2](https://jipeng4974.github.io/img/wiring_diagrams_2.png)
上面整张接线图表达的是"若 $t\le v, w+u\le x+z, v+x\le y$，则 $t+u\le y+z$"。

我们可以并排画两条线来表示两个标签的幺半积。
![接线图 3](https://jipeng4974.github.io/img/wiring_diagrams_3.png)
上图中方框的合法性对应于 $x_1\otimes x_2 \le y_1 \otimes y_2 \otimes y_3$。

TBD

[^1]: 更严谨一点说，把 **定义 2.1** 中的 $=$ 全部替换为 $\cong$ 往往更有用。
