# Flash Attention

> Flash Attention，一言以蔽之：tiling + selective gradient checkpointing。

---

LLMS index: [llms.txt](/llms.txt)

---

由於`self-attention`的時、空複雜度都是序列長的平方，長序列LLM和高分辨率ViT都是非常喫內存的。

此前針對`self-attention`的優化大多是近似計算，其核心是優化FLOPs，例如把理論上的時間複雜度降低到O(N)，但這並不能有效加速`self-attention`，因爲該操作（以及transformer中多數操作）的實際瓶頸在訪存——更準確地說是HBM和SRAM之間的IO。

`FlashAttention`的原理就是基於`tiling`，確保內層循環計算fit in SRAM，減少了HBM和SRAM之間的IO頻次，因而真切有效地提升了transformer性能，解鎖了更長的context，迅速在各種高性能框架中得到應用。

## 傳統的Self-Attention實現
Attention Layer具有將局部信息和張量中較遠位置的信息結合起來的能力——通過爲所得張量的每個組件與輸入張量的每個組件計算注意力得分，不受局部性約束地，在整個張量範圍內對特徵進行平均。

給定$N^Q\times D^{QK}$維queries張量$Q$，$N^{KV}\times D^{QK}$維keys張量$K$，$N^{KV}\times D^V$維values張量$V$，通過```Attention操作```$att(K,Q,V)$計算得到$N^Q\times D^V$維的張量Y：

$$Y = att(K,Q,V) = \underbrace{softargmax(\frac{QK^T}{\frac{1}{\sqrt{D^{QK}}}})}_A V$$

整個過程分兩步，第一步先計算每個query index $q$和每個key index $k$的注意力得分，即queries和keys點乘後的```softargmax```結果：$A_{q,k} = \frac{exp(\frac{1}{\sqrt{D^{QK}}} Q_q \cdot K_k ) }{\sum_l exp(\frac{1}{\sqrt{D^{QK}}} Q_q \cdot K_l)}$，其中$\frac{1}{\sqrt{D^{QK}}}$是一個縮放參數，用於保證取值範圍在$D^{QK}$變大時大體不變。

![att](https://wujipeng.com/img/attention.png)

得到注意力得分$A_{q,k}$後，再進行第二步計算：$Y_q = \sum_k A_{q,k}V_k$。注意力得分即queries和keys之間的匹配程度，匹配程度越高，該權重就越高。如果某個query和某個key匹配度達到極限，注意力得分接近1，則直接拿到這個key對應的value。如果和好幾個key都有中等水準的匹配，則按注意力得分做加權平均。

## FlashAttention的IO優化
`FlashAttention`注意到事實上並不需要完整的輸入K、V、Q，可以分批讀入、分批計算、分批把結果寫回O，即所謂`tiling`：
- 在外層循環，遍歷K、V矩陣。每次只需加載一個block的 $K^T$ 和 $V$ 到片上SRAM中。
- 在內層循環，遍歷Q的各個block，加載到SRAM中，進行 $att(K,Q,V)$ 計算，把局部結果寫回HBM上 $N\times d$ 的結果矩陣[^3]。
- 對softmax歸一化因子做相應調整，即可保證最後加起來得到的最終結果和標準實現等價，具體代數見[^1]附錄。
- 設置K、V的block size爲 $\lceil \frac{M}{4d} \rceil$，Q、O的block size爲$min(\lceil \frac{M}{4d} \rceil, d)$[^4]。

![fast_att](https://wujipeng.com/img/fast_attention.png)

此外，對於訓練負載來說，`FlashAttention`還在反向傳播中複用前饋過程暫存的softmax歸一化因子$\frac{1}{\sqrt{D^{QK}}}$，這也比從HBM讀$N\times N$的巨大attention中間矩陣要快得多。這可被視作selective gradient checkpointing。

## FlashAttention2: 改進並行和工作切分
相比GEMM，`FlashAttention`只達到了25~40%理論FLOPs，可優化空間巨大。`FlashAttention2`[^2]在原版基礎上做了並行化和工作切分優化。
- 減少非矩陣乘算子，因爲GPU矩陣乘是高度優化的，其他算子與之差距非常大。
    - 避免循環中算O時每次都rescale，而是在算最終結果時施加softmax歸一化因子。
    - 對反向傳播時保持的狀態進行了精簡。
- 在不同線程塊[^8]上進行並行計算，從而充分利用GPU資源。
    - 原版一個線程塊處理一個head/一個batch，每個線程塊跑在一個SM上。不過在長序列場景，head數目和batch size可能都偏小，導致二者相乘後都未必能打滿A100的128個SM。
    - 顯而易見可以並行的部分是外層循環，可以隨便調度到不同線程塊上，相互之間完全沒有通信需求。
    - 反向傳播時並行化外層循環也只有dQ更新時需要簡單的通信/同步，這是一個順序不重要的加法，atomic add足以解決。

既然有了線程塊，就要考慮在每個線程塊內，不同warps之間如何進行工作的切分。
![work_part](https://wujipeng.com/img/work_part.png)

如上圖所示，`FlashAttention`的切分方式導致內層循環中各個warp都需要把結果寫到共享內存且做一個同步加，存在一定的通信開銷，而`FlashAttention2`的切分方式可以保證warp之間完全沒有通信需求。


[^1]: T. Dao, et al. FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness. [[pdf]](https://arxiv.org/pdf/2205.14135)
[^2]: Tri Dao. FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning. [[pdf]](https://arxiv.org/pdf/2307.08691)
[^3]: 其中，d爲head dimension。N爲序列長度。$N \gg d$。GPT2中 $N=1024, d=64$。
[^4]: 其中，M爲SRAM大小。
[^5]: M. Milakov, N. Gimelshein. Online Normalizer Calculation for Softmax. [[pdf]](https://arxiv.org/pdf/1805.02867)
[^6]: Markus N. Rabe, Charles Staats. Self-attention Does not Need $O(n^2)$ Memory. [[pdf]](https://arxiv.org/pdf/2112.05682)
[^7]: W. Kwon, et al. Efficient Memory Management for Large Language Model Serving with PagedAttention. [[pdf]](https://arxiv.org/pdf/2309.06180)
[^8]：多個線程塊，一個線程塊(thread block)內的多個warp可分時複用同一個SM。
