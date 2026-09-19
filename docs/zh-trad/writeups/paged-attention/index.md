# Paged Attention

> Paged Attention：提升同時處理多請求的顯存利用率和吞吐。

---

LLMS index: [llms.txt](/llms.txt)

---

在LLM serving場景，通過恰當的batching積攢足夠多的請求，可提升LLM吞吐。但每個請求對應的KV Cache非常巨大——在原始實現中，KV Cache需要爲`max_tokens`預留內存，但每個請求實際上攜帶的tokens數目普遍遠小於`max_tokens`，造成大量內存浪費和反覆的動態內存分配。

`PagedAttention`[^1]的目標就是消除這種內存浪費、在請求內部和請求之間靈活共享一些KV cache。vLLM通過`PagedAttention`達到此前sota的`FasterTransformer`和`Orca`的2~4倍吞吐。

此前，樸素的KV Cache實現如下圖所示，簡短的prompt和並不長的當前iteration只佔據了7+4個slots，剩餘2038個slots不得不預留在內存裏以支撐最大序列長達2048的承諾（採樣結束才能知道實際tokens數目），在這之後才能是下一個請求的slots，中間預留的部分就完全浪費掉了。

![naive_kv_cache](https://wujipeng.com/img/naive_kv_cache.png)

`PagedAttention`把連續的kv存在不連續的內存空間上，借用類似頁表的機制（引入一個block_table）規避內存碎片化問題。具體來講，`PagedAttention`把KV cache分區成若干個K blocks和V blocks，每個K/V block容納固定數量tokens所對應的K/V向量，因此attention計算也被轉化爲blockwise計算。這和`FlashAttention`有點像，不過應用的尺度不同，前者是爲了大規模serving時克服碎片化、按需取用，後者是爲了單次self-attention計算能全部fit in SRAM。

![block_table](https://wujipeng.com/img/block_table.png)

這個物理kv blocks顯然是支持多個請求複用的。如下圖所示，爲每個請求維護一個小的block table即可。

![vllm_two_requests](https://wujipeng.com/img/vllm_two_requests.png)


[^1]: W. Kwon, et al. Efficient Memory Management for Large Language Model Serving with PagedAttention. [[pdf]](https://arxiv.org/pdf/2309.06180)
