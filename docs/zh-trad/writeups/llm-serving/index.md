# LLM Serving

> 本文總結LLM serving的計算形態和優化機會。

---

LLMS index: [llms.txt](/llms.txt)

---

LLM推理有別於小模型推理，更加memory-bound，輸入輸出也有可以利用的獨特模式，自然湧現出系統層面可優化的點。我將這些系統層面的優化機會粗略分爲batching優化、sampling優化、模型壓縮3類。

其中batching優化負責輸入端將GPU等加速器的計算單元喂的更飽和、更好地利用SRAM locality[^5]，sampling優化[^6]負責輸出端更快速拿到LLM的結果，模型壓縮負責讓大模型不要那麼大。

## LLM Serving的計算形態
LLM serving相比此前主流的深度模型serving，有一些獨有的特性和約束：
- 一次LLM調用中很多時間耗費在加載模型參數上（HBM->SRAM）。
    - batching可以讓一次參數加載負責多個seq的推理，顯著減少了總體的模型參數加載開銷。
- Seq2Seq：變長輸入、變長輸出。
    - 因此batching機制必須適應batch size，seq len皆爲變量。
- 自迴歸（通過若干次迭代才能處理一個請求），且爲每個對話維護一個跨迭代的KV。
    - 純粹無狀態的方法需每次重新計算所有KV，開銷會大得無法接受，因此必須基於KV caching做incremental decoding。
    - RNN也是自迴歸，transformer相比RNN，還有一個不同之處是每次迭代KV cache都會變大。
    - 非自迴歸模型往往以request爲粒度做batching，對自迴歸場景並不適用，後者明顯更適合以iteration爲粒度。
- GPT不同於原版transformer，是decoder-only的。
    - 此前encoder-only和encoder-decoder架構的變長batching padding策略不適用於LLM。
- GPT多了一個前置的計算密集的prefill階段（也叫infill），用於處理prompt。
    - prefill階段和增量迭代階段的序列不好batching（計算要完全一樣纔好batching）。
    - 這個階段消化prompt，考慮到很多LLM應用都有非常巨大、內容也差不多的prompt，prefill階段很多計算都是重複的。
- GPT在生成token概率分佈後，還有一個後置的sampling階段：基於概率密度從vocab中選擇token。
    - token選擇完成後，當前迭代選中的token還需要再作爲下次迭代的forward pass輸入參數。
    - 若seq[A~A+5]在此前的迭代中已經進行了採樣，在新一輪迭代中除了需要對seq[A+6]採樣之外，仍然要對seq[A~A+5]再算一遍。
    - 已處理tokens的decoding結果可以cache下來。
    - 很多LLM應用會要求結構化輸出，輸出的格式比較固定。
- Transformer的attention算子的input張量形狀和已處理tokens長度有關。
    - 不同序列，序列長度不同，attn計算的輸入形狀不統一，就不好做batching。
    - 好在attn計算做不做batching影響不大，因爲attn計算並不涉及任何模型權重，也就不存在一次參數加載重用於多序列推理的加速效應。
- LLM在GPU上跑的一個關鍵瓶頸是將數據（請求+參數）加載進HBM的開銷。LLM serving吞吐明顯會受限於batch size——即一次性能放入多少數據而不至於爆顯存。
    - 在簡單的靜態batching實現中，seq_len * nr_seqs + 模型大小決定了顯存佔用。seq_len假設定的比較高，又用不完，那也會讓nr_seqs縮小很多。
    - 顯存如此緊張，自然使量化、剪枝等模型壓縮技術在LLM serving中變得尤爲重要，比如awq, gptq, gguf, smooth quant。

## Continuous Batching
上文提及LLM做batching好處雖多，困難更多，並不存在特別簡單、顯而易見的GPT特供batching機制。

Orca[^1]首先爲GPT模型的batching提出了一個完整的解決方案，在迭代層面進行調度，並通過一個必要的selective batching機制剔除不能做batching的操作（若不做剔除，兩個請求整體能做batching的幾率可以忽略不計），這些操作雖然不能做batching，但對整體性能提升的影響很小。

![orca](https://wujipeng.com/img/orca.png)

## Paged Attention
Orca方案並未考慮KV cache的HBM佔用，默認預分配max_seq_len。

但monolithic KV cache導致HBM碎片化，爲每個seq預分配巨大內存進而導致併發不足也是真實瓶頸所在，vLLM中就對此進行了優化，提出了Paged Attention[^2]，一種近似頁表的分塊kv cache技術：
- 在prefill階段允許kv cache在非連續內存中以頁的方式組織起來，因此不必爲max_seq_len提前分配內存，運行時再分配就好。
- 大多數seq顯然不會觸及max_seq_len，paged attention因此節省了大量內存，也就允許batch size大大提高。
- vLLM的實現中並未採納Orca的selective batching，主要是因爲它的paged attention算子是自己寫的cuda，可以與非attn算子一起兼容batching。vLLM將prefill和decoding分開做batching，整體上就不需要實現selective batching這麼麻煩的機制了。
    - 但這種做法也阻止了prefill和decoding step的融合。如果某個prompt過長，prefill開銷太大，確實會出現block後續所有decoding batch的情形。

![block_table](https://wujipeng.com/img/block_table.png)

詳見[Paged Attention](https://jipeng4974.github.io/writeups/paged-attention)。

## Dynamic SplitFuse
DeepSpeed-FastGen[^3]中提出了SplitFuse，也是Continuous Batching的一個演化版本。思路是切分長prompt請求成若干個小的step，這些小的step開銷較低，可填充調度縫隙，同時還保證prefill(prompt generation)和decode(token generation)的steps開銷一致，就可以確保不存在大小不同的workload，取得一定的吞吐提升，但最主要的優勢是能穩住tail-latency，在線服務場景下下限更高。

![split_fuse](https://wujipeng.com/img/split_fuse.png)

## Quantization and Pruning
由於Nvidia架構上天然偏向圖形負載，顯存天然就給的不足，各種量化、剪枝技術除了降低計算量外，在LLM serving方向上還起到了關鍵性的降低顯存佔用的效果。

詳見[Quantization and Pruning](https://jipeng4974.github.io/writeups/quantization-and-pruning)

## Radix Attention
SGLang採用了Radix attention[^4]技術，將common prefix的KV以radix tree的形式保留下來，使kv cache的生命週期不侷限於一次請求，而是真正構成跨多次請求的LRU cache，適應prompt巨大且大多有相同前綴的實際應用場景。

## Flash Attention
Continuous batching提升了非attn操作的SRAM locality，針對kv計算，Flash Attention[^8]則令attn計算內層循環fit in SRAM。

![fast_att](https://wujipeng.com/img/fast_attention.png)

詳見[Flash Attention](https://jipeng4974.github.io/writeups/flash-attention)。

## Speculative Decoding
Speculative decoding[^7]的思路是選用tokenizer相同，大小不同的兩個模型。假設大模型的latency大體上是小模型的N倍。小模型輸出N個token的時間內，大模型把這N個token拿過來append到seq上形成input，再輸出1個token，總計生成N+1個token。基於greedy decoding對這N+1個token進行採樣，採樣結果如果和小模型的結果match，就直接用了，不match就停下，在停下的地方把原本的小模型token改成大模型採樣結果對應的token。整個過程中，大模型實際上只需要一個forward pass，運氣好就能一下子輸出N+1個token，運氣差就輸出1個token。

![speculative_decoding](https://wujipeng.com/img/speculative_decoding.png)


## Structured Decoding
SGLang基於一個壓縮有限狀態機實現了structured decoding[^4]，用於對特定結構化輸出（比如支持regex的JSON模板）進行加速，一次性decode多個token。假設這個結構化輸出的JSON模板中總是有一個key是"top5 candidate"，那就可以把"top5 candidate"這個multi-token詞組當成一個token一輪迭代處理掉。

![structured_decoding](https://wujipeng.com/img/structured_decoding.png)


[^1]: Gyeong-In Yu and Joo Seong Jeong. Orca: A Distributed Serving System for Transformer-Based Generative Models. OSDI 22. [[pdf]](https://www.usenix.org/system/files/osdi22-yu.pdf)
[^2]: W. Kwon, et al. Efficient Memory Management for Large Language Model Serving with PagedAttention. [[pdf]](https://arxiv.org/pdf/2309.06180)
[^3]: C. Holmes, et al. DeepSpeed-FastGen: High-throughput Text Generation for LLMs via MII and DeepSpeed-Inference. [[pdf]](https://arxiv.org/pdf/2401.08671)
[^4]: L. Zheng, et al. SGLang: Efficient Execution of Structured Language Model Programs. [[pdf]](https://arxiv.org/pdf/2312.07104)
[^5]: GPU/NPU/TPU等加速器需將模型參數從off-chip memory加載到on-chip SRAM才能進行底層硬件算子的計算，對較大的模型，這種加載開銷往往纔是瓶頸所在。因此batching不僅僅能提升加速器計算單元的利用率，還能通過一份模型參數在多個請求中重用，更好地利用SRAM locality。
[^6]: sampling指的基於density做token-selection的過程，decoding指的是整個decoder-only transformer的inference過程。
[^7]: Y. Leviathan, et al. Fast Inference from Transformers via Speculative Decoding. [[pdf]](https://arxiv.org/pdf/2211.17192)
[^8]: T. Dao, et al. FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness. [[pdf]](https://arxiv.org/pdf/2205.14135)
