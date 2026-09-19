# 隨筆

> 爲現實建立結構

---

LLMS index: [llms.txt](/llms.txt)

---

Section pages:

- [表徵幾何](/zh-trad/writeups/representation_geometry/): 表徵學習的反覆迭代中，應嘗試估計嵌入子流形的內稟維數，在輸入和輸出環節上設計合理的 ambient space。
- [Music JEPA 和三體宇宙](/zh-trad/writeups/music_jepa_universe/): 生存維數，維度坍塌，死維復活，宇宙規律，田園宇宙，倖存子空間，威懾項λ。
- [Music JEPA Regularizers](/zh-trad/writeups/music_jepa_reg/): 再試Music LeJEPA之抗坍縮正則
- [DSpark](/zh-trad/writeups/dspark/): DSpark = 半自迴歸 draft（重並行骨幹 + 輕量順序頭） + 置信度感知的動態驗證調度
- [自由意志的可度量指標](/zh-trad/writeups/free_will/): 由己度 = 內敏性 - 外敏性。
- [Kimi K3：沿序列、深度與寬度擴展 LLM](/zh-trad/writeups/llm_arch/): LLM架構演化，從dense transformer到Kimi K3。
- [Music LeJEPA](/zh-trad/writeups/music_lejepa/): 初試Music LeJEPA
- [鏡頭與編碼器的同構性](/zh-trad/writeups/lens_encoder/): 相機鏡頭和編碼器呈現驚人的同構性。
- [超越表徵學習：預測驅動的編碼器+可學習的傳感器](/zh-trad/writeups/beyond_representation_learning/): 現有的多模態表徵學習原始、孤立且殘缺，還需要預測驅動的編碼器+可學習的傳感器。
- [認知相變](/zh-trad/writeups/cognitive-phase-transitions/): 人類智能價值穿越週期的不變量，是認知相變能力。
- [語義深度和音樂識別](/zh-trad/writeups/reid/): 本文提出語義深度指標，從歸納偏置、數據驅動、性能工程、級聯排序等角度討論Music Re-ID。
- [推薦系統](/zh-trad/writeups/recomendation-systems/): 藉助DeepSeek R1讀論文，梳理推薦系統。
- [LLM Serving](/zh-trad/writeups/llm-serving/): 本文總結LLM serving的計算形態和優化機會。
- [C++系統工程範式](/zh-trad/writeups/engineering-practices/): 本文總結當下我認爲比較好的C++系統工程範式。
- [量化和剪枝](/zh-trad/writeups/quantization-and-pruning/): 總結推理優化問題中相當重要的模型壓縮技術——量化和剪枝。
- [AI推理優化](/zh-trad/writeups/optimizing-ai-inference/): 總結AI工程中的推理優化問題。
- [Paged Attention](/zh-trad/writeups/paged-attention/): Paged Attention：提升同時處理多請求的顯存利用率和吞吐。
- [Flash Attention](/zh-trad/writeups/flash-attention/): Flash Attention，一言以蔽之：tiling + selective gradient checkpointing。
- [CAT02: Resources](/zh-trad/writeups/category-theory-2---resources/): 第二篇 CAT 筆記討論資源的範疇論形式化，以及如何將一組資源轉化爲另一組資源。內容涵蓋 monoidal preorder（幺半預序）、wiring diagram（接線圖）、monoidal monotone map 和 V-category。
- [Diffusion Probabilistic Models](/zh-trad/writeups/dpm/): 本文是對朱軍教授的分享——“用於生成高維數據的擴散模型”的筆記。值得注意的是DPM實踐中巧妙使用瞭解析解，無論是前向過程的closed form $q(x_N|x_0)$，還是逆向過程中解析形式的方差估計，都大大提升了訓練性能，體現了數學的精妙。
- [筆記：推薦系統研究現狀的理解](/zh-trad/writeups/revisiting-recomender-systems/): 本文是對孫愛欣教授的分享——“推薦系統研究現狀的理解”的筆記，對大致內容進行了摘要，並收集了提及文獻的鏈接——可以一窺推薦系統領域學術研究的現狀。
- [Reflecting on a Wake-up](/zh-trad/writeups/reflecting-on-a-wake-up/): 記黎明前的一次醒來。
- [錯誤處理](/zh-trad/writeups/error-handling/): 本文討論現代C++的錯誤處理問題。
- [Linkers & Loaders](/zh-trad/writeups/linkers-and-loaders/): 《Linkers & Loaders》填補了一個niche知識域——鏈接和加載。
- [CAT01: Orders](/zh-trad/writeups/category-theory-1---orders/): CAT 系列第 1 篇，以序理論（order theory）爲完整的範疇論做熱身。內容涵蓋 preorder、meet/join、monotone map 與 Galois connection。
- [Tech Talk: Wall is Coming](/zh-trad/writeups/tech-talk---wall-is-coming/): Tech Talk文稿，梳理內存牆問題的歷史淵源，嘗試給出對優化空間的理解，推導出相應的啓發性策略，並列舉一些訪存優化技術。
- [工作減少 vs 硬件賦能](/zh-trad/writeups/work-reduction-vs-hardware-enablement/): 優化可以分爲兩類：工作減少與硬件賦能。
- [Dash: 可擴展哈希](/zh-trad/writeups/dash/): Dash 論文的主要關注點是曾經風靡一時的 `persistent memory`，但實際上，任何受 `memory bandwidth` 限制的場景都能從中受益。隨着 Intel 砍掉其 `pmem` 業務，`Dash` 方案的意義已經轉移到了普通的 `DRAM` 應用上。
- [DPDK is All You Need](/zh-trad/writeups/dpdk/): 對於訪存密集的數據中心應用來說，DPDK提供了非常好的性能工程範式。
- [The Little Book 書評](/zh-trad/writeups/the-little-book-review/): 正如DDIA可被視爲分佈式系統方向的入門教程，LBDL是理想的深度學習101。
- [內存瓶頸應用的eBPF Tracing](/zh-trad/writeups/ebpf/): 介紹eBPF——前沿的Linux系統的可觀測性技術，以及基於eBPF的off-CPU性能分析。
- [人造直覺](/zh-trad/writeups/on-reasoning-abilities-of-llms/): 對近期對LLM的調研文獻進行梳理總結，討論目前大語言模型架構的推理能力邊界。
- [論秩序](/zh-trad/writeups/on-order/): 秩序，湧現自耗散結構的自組織。
- [計算意識](/zh-trad/writeups/computational-conciousness/): 形而上學、神經科學意識論和計算意識。
- [高性能大規模向量檢索](/zh-trad/writeups/efficient-anns-at-scale/): 如何在十億、百億級特徵庫上做高效的向量檢索
- [Tech Talk: Evolution of Data Center Applications](/zh-trad/writeups/tech-talk---evolution-of-data-center-applications/): Tech Talk文稿，主旨是白盒化算法理論和基礎設施，給出一個對Datacenter應用，尤其是Datacenter AI演進趨勢的主觀理解。
- [歸納和演繹](/zh-trad/writeups/induction--deduction/): 人的無知，可分爲神祕和問題。人的理性，也可分爲歸納和演繹。
- [大規模分佈式計算](/zh-trad/writeups/distributed-computing-systems/): 紛亂的分佈式現象、繁瑣的工程實踐容易遮蔽對分佈式計算系統本質的理解，遂做梳理。
- [知識是現實的Embedding](/zh-trad/writeups/reality--knowledge/): 現實無限廣博，無限深邃，將其嵌入我們有限的認知空間後，就形成了知識。
- [聯邦學習](/zh-trad/writeups/federated-learning/): 聯邦學習(Federated Learning)是指許多移動設備在一箇中央服務器的編排下協作訓練模型，保持訓練數據離散，避免對用戶數據進行收集，僅將客戶端模型更新上傳中央服務器彙總成新的全局模型的機器學習模式。與in-center的分佈式訓練相比，有其獨特的優勢和挑戰。
- [論透明度](/zh-trad/writeups/on-transparency/): 透明度，或者說程序的白盒指數，是互聯網軟件工程實踐中長期被忽略的一個理想屬性。
- [String Lookups Reduce to Parsing](/zh-trad/writeups/string-lookups-could-reduce-to-parsing/): 字符串查找和字符串解析，本質都用盡可能緊湊的結構和高效的算法，從字符流中抽取狀態。因此龍書中的NFA轉DFA算法可以派上用場。
- [泛型編程的三種範式：Archetype, Ducktype, Subtype](/zh-trad/writeups/paradigms-of-generic-programming/): 本文總結泛型編程的三種範式：Archetype, Ducktype, Subtype。三者都以type結尾，一方面是因爲這樣比較帥，有規則感和邏輯上的建築美，另一方面是因爲系統語言編程本身就是在打造一個個類型，而泛型編程就是在打造一個個類型規約+遵循規約的類型。
- [非凸優化](/zh-trad/writeups/on-nco/): 非凸優化(non-convex optimization)，more like art
- [論ABI](/zh-trad/writeups/on-abi/): 本文總結了介於ISA和語言標準這兩個簡約協議層之間隔離了大量複雜度的抽象層次——系統語言的ABI。
- [有狀態分佈式系統分類學](/zh-trad/writeups/a-taxonomy-of-stateful-distributed-systems/): 本文討論了CAP Theorem的侷限性，梳理了基於一致性、可用性這兩個理想屬性間的權衡的更細緻精確的有狀態分佈式系統分類學。
