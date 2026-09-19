# CAT02: Resources

> 第二篇 CAT 筆記討論資源的範疇論形式化，以及如何將一組資源轉化爲另一組資源。內容涵蓋 monoidal preorder（幺半預序）、wiring diagram（接線圖）、monoidal monotone map 和 V-category。

---

LLMS index: [llms.txt](/llms.txt)

---

## 對稱幺半預序

**定義 2.1:** 預序 $(X,≤)$ 上的一個 `对称幺半结构`（symmetric monoidal structure）由兩個成分構成：(i) 一個元素 $I \in X$，稱爲 `幺半单位`（monoidal unit）；(ii) 一個函數 $\otimes: X \times X \rightarrow X$，稱爲 `幺半积`（monoidal product）。這些成分必須滿足以下性質：
- 單調性（monotonicity）：$\forall x_1, x_2, y_1, y_2 \in X$，若 $x_1 \le y_1$ 且 $x_2 \le y_2$，則 $x_1 \otimes x_2 \le y_1 \otimes y_2$。
- 單位性（unitality）：$\forall x \in X$，$I \otimes x = x$ 且 $x \otimes I = x$ 成立。
- 結合律（associativity）：$\forall x,y,z \in X$，$(x\otimes y)\otimes z = x \otimes (y\otimes z)$。
- 對稱性（symmetry）：$\forall x,y \in X, x\otimes y = y\otimes x$。[^1]

**定義 2.2:** 配備了對稱幺半結構的預序 $(X,\le,I,\otimes)$ 稱爲 `对称幺半预序`（symmetric monoidal preorder）。

**例 2.1（布爾值）:** $\mathbb{B} = \{true, false\}$ 配合 $false < true$ 是最簡單的非平凡預序。我們可以定義幺半單位爲 true，幺半積爲 $\wedge$（與，AND）。這樣我們就得到了一個幺半預序，記作 $Bool := (\mathbb{B}, \le ,true, \wedge )$。

## 接線圖
`接线图`（wiring diagram）是從舊關係構建新關係的可視化表示。在沒有幺半結構的預序中，關係是串聯起來的。
![接線圖 1](https://wujipeng.com/img/wiring_diagrams_1.png)

有了對稱幺半結構，關係也可以並聯排列。
![接線圖 2](https://wujipeng.com/img/wiring_diagrams_2.png)
上面整張接線圖表達的是"若 $t\le v, w+u\le x+z, v+x\le y$，則 $t+u\le y+z$"。

我們可以並排畫兩條線來表示兩個標籤的幺半積。
![接線圖 3](https://wujipeng.com/img/wiring_diagrams_3.png)
上圖中方框的合法性對應於 $x_1\otimes x_2 \le y_1 \otimes y_2 \otimes y_3$。

TBD

[^1]: 更嚴謹一點說，把 **定義 2.1** 中的 $=$ 全部替換爲 $\cong$ 往往更有用。
