# 推薦系統

> 藉助DeepSeek R1讀論文，梳理推薦系統。

---

LLMS index: [llms.txt](/llms.txt)

---

## 搜推架構概述
傳統搜推採用三層漏斗結構(matching召回/preranking粗排/ranking精排)。

召回階段兼顧相關性和一定的排序能力，往往多路召回。
- 傳統三層結構中，召回和排序的分離導致商業回報較低，因此召回時可通過相關性模型做訓練樣本增強[^1]，或將相關性badcase作爲負反饋建模[^2]。
- 樣本空間設計需考慮SSB問題[^7]：
    - 做in-batch負採樣[^8]和logQ流行度糾偏[^9]。
    - 構建中等難度負樣本[^10]。
    - 可參考Pinterest廣告檢索的MUDA（Modified Unsupervised Domain Adaption）方法[^11]。
- 模型結構：樸素雙塔、MVKE[^12]、QARM[^13]。

粗排階段在相關性有保障的前提下，主要考慮排序能力，篩選出優質結果。
- 粗排同樣需要應對SSB。[^15]
- 除了傳統粗排模型外，還可以考慮和精排對齊的方法：COPR[^16]。

精排階段的目標是通過精細化建模與多維度優化，從候選集中選出最優內容，精準預測用戶行爲，最大化某種商業目標。
- 精排：精細化打分百級候選（DeepFM、DIN、MMOE），側重精準預測（CTR/CVR）和特徵交互。
- 重排序：調優Top結果（PRM、DPP、MMR），側重列表級優化（多樣性、上下文）。

## Mobius
Mobius項目旨在通過將CPM[^3]、CTR[^4]、CPA[^5]、ROI[^6]等商業指標整合到匹配層，同時保持低延遲和計算資源限制，從而提升效果。

Mobius有兩項關鍵技術：主動學習的CTR模型和快速廣告檢索。前者通過教師-學生框架，利用已有相關性模型生成合成數據，增強CTR模型對長尾查詢和廣告的泛化能力。後者就是我們做內容檢索方向非常熟悉的ANNS/MIPS和OPQ向量壓縮等檢索技術。

Mobius中，CTR模型是直接集成到了召回階段（傳統搜推的CTR往往在排序階段，和召回割裂）的雙塔DNN（query塔+ad塔），在召回時就同時考慮相關性和商業價值，對數十億的廣告候選也能做出高效篩選。

ANNS不必贅述，OPQ（Optimized Product Quantization）是一個新穎的高維向量低損壓縮機制，對PQ稍作升級，對原始向量施加了正交矩陣R，使變換後的向量在子空間中分佈更均勻，減少子向量之間的相關性，從而減少了PQ量化誤差（PQ將向量分割成多個子向量，這些子向量之間可能存在相關性）。

## "Not-to-Recommend" Loss
Google在[^2]中提出一個利用用戶負反饋進行訓練的"Not-to-Recommend" loss，顯式地利用負反饋信號優化不推薦該項目的log-likelihood。這個loss和標準的正反饋交互交叉熵loss合併，形成一個兼顧正、負反饋的聯合學習機制。

此前大多數利用負反饋的工作往往是ad-hoc的特徵工程。

## SimANS 
SimANS是針對密集文本檢索任務剔除的負樣本採樣方法，旨在解決現有方法中存在的非信息性負樣本（過於簡單）和錯誤負樣本（過於困難）問題。其核心思路是通過選擇模糊負樣本（ambiguous negatives）——即與正樣本相關性分數接近的負樣本，平衡樣本的難度與信息量，從而提升模型訓練效果。

其創新點在於首次明確將中等難度負樣本（ambiguous negatives）作爲核心採樣目標，提出其梯度特性（均值大、方差小）的理論依據，並通過實驗驗證其對模型訓練的重要性。

此外，SimANS引入了基於指數函數的概率分佈，通過分數差異動態調整採樣權重，避免固定策略（如隨機或Top-k）的缺陷，同時兼顧計算效率。

## MUDA
傳統推薦系統在訓練數據中存在選擇偏差，訓練數據分佈往往和實際推理時的數據分佈不一致。廣告檢索領域，訓練數據主要來自後續階段（拍賣勝出者），這些數據是多輪篩選後的結果，和真實全量候選廣告集有很大分佈差異。

傳統的基於用戶行爲的建模忽視了大量未被選中廣告的信息，MUDA[^11]將精排結果（即僞標籤）轉化爲二元標籤（正負例），用二元交叉熵loss替代迴歸損失，避免過擬合低置信度的僞標籤，從而在訓練中間接利用到哪些未被標記卻與真實推理時數據分佈更接近的數據。

## MVKE
此前用戶畫像建模普遍使用多個獨立雙塔模型預測CTR和CVR[^14]，但存在數據稀疏（單一動作模型，比如僅點擊、僅轉化無法互補學習，導致數據稀疏動作的建模效果差）、特徵交互不足（雙塔模型兩塔分離，用戶和標籤的交互不足，難以捕捉多主題偏好）的問題。

MVKE模型中設置了虛擬內核專家（VKE）和虛擬內核們（VKG門控結構）聯合學習用戶在不同動作（點擊、付費）和主題（體育、汽車）上的偏好，提升用戶畫像多樣性和準確性。MKVE中一個用戶塔包含多個VKE，每個VKE負責用戶偏好的一個特定領域，並通過所謂虛擬內核（某種可學習參數）指導特徵交互。VKG根據標籤embedding和虛擬內核動態融合不同VKE的輸出。

在“點擊->轉化”序列中，點擊作爲淺層任務，可以爲轉化這一深層任務提供基礎信息。

## QARM
QARM是基於MLLM（多模態大語言模型）的推薦架構。論文指出工業界推薦系統的MLLM有兩個問題，一個是表示不匹配，一個是表示無法學習。前者指的是預訓練的多模態模型與下游推薦任務的目標不一致（比如說圖片表徵本來是用來做image-text匹配的），後者指的是多模態表示在推薦模型中無法通過梯度更新進行優化（往往MLLM生成的表徵只用做推薦模型的額外輸入）。

QARM框架提出了Item Alignment（項目對齊機制）和Quantitative Code（量化編碼機制）。
- 對齊：通過業務特定的用戶-項目交互數據（如高質量Item2Item對），微調預訓練多模態模型，使表徵與下游任務對齊。例如，針對電商場景調整因果商品關係，而非通用語義對齊。
- 量化：將對齊後的多模態表徵轉換爲可學習的離散編碼ID（如VQ向量量化與RQ殘差量化），支持端到端訓練。編碼ID作爲推薦模型的輸入特徵（如用戶興趣序列、物品屬性），替代靜態表徵。

## ASH & ASMOL
傳統粗排往往被視爲迷你精排，用輕量模型，依賴AUC等有序列表評估指標，但這種方法的離線評估和在線A/B測試結果不一致。而且粗排的目標其實不是生成有序列表，而應該是高質量無序候選集。

論文[^15]認爲盲目模仿精排模型會限制粗排潛力，需專注集合質量而非單純一致性。

論文引入ASH(All-Scenario Hitrate)指標，將推薦、購物車等不同場景的購買正樣本都放在一起，緩解單一場景樣本偏差，更全面地衡量粗排候選集質量。實驗證明ASH與在線GMV強相關，優於傳統指標AUC(Area Under the ROC Curve)[^17]、ISPH(In-Scenario Purchase Hitrate)。

論文還提出來多目標學習框架ASMOL：
- 全空間訓練樣本：整合曝光樣本、排序候選樣本、預排序候選樣本，覆蓋更全面的數據分佈。
- 多目標損失：聯合優化曝光、點擊、購買任務，學習優先級（購買 > 點擊 > 曝光）。
- 知識蒸餾：從排序模型蒸餾知識，但僅針對曝光樣本（避免噪聲干擾）。
- 單模型架構：統一模型處理多任務，優於傳統多模型策略。

## COPR
COPR[^16]察覺到相對輕量的粗排模型的能力不如複雜精排模型，因此以對齊精排結果爲目標建模，通過分塊採樣和排序對齊，將目標從分數對齊放寬爲排序對齊，有效緩解了模型容量差異與投標誤差放大的問題。其即插即用設計可適配多種預排序模型，並在淘寶廣告系統中成功落地，顯著提升了廣告效果與平臺收益。該研究爲級聯架構中的模型協同優化提供了新思路。

COPR缺乏探索策略。分塊採樣可能保留不同優先級的廣告組，ΔNDCG加權更關注頭部廣告的排序正確性，這主要影響排序的準確性而非多樣性。如果頭部廣告本身多樣性不足（如重複推薦同類高CTR廣告），則可能引發信息繭房。



[^1]: MOBIUS: Towards the Next Generation of Query-Ad Matching in
Baidu’s Sponsored Search. [[pdf]](https://arxiv.org/pdf/2409.03449)
[^2]: Learning from Negative User Feedback and Measuring
Responsiveness for Sequential Recommenders. [[pdf]](https://arxiv.org/pdf/2308.12256) 
[^3]: CPM(Cost per Mile)爲千次廣告展示的成本，適用於品牌曝光。
[^4]: CTR(Click-through Rate)爲用戶點擊率，適用於搜索廣告。
[^5]: CPA(Cost per Action)爲每次用戶行爲成本（入下載、購買），適用於深度目標轉化。
[^6]: ROI(Return on Investment)爲投資回報率，收益和成本的比值。
[^7]: SSB(Sample Selection Bias)是指訓練數據中樣本選擇過程不隨機或不符合真實數據分佈，影響模型效果的問題。推薦系統中，熱門項目往往佔據大部分用戶交互數據（數據分佈highly skewed纔是常態），長尾項目（冷門項目）的數據過於稀疏，若模型過度依賴熱門項目的數據，則容易導致對長尾項目的推薦效果不佳。負樣本（用戶未交戶的項目）若隨機採樣，很容易導致佔比高的熱門項目倍過度採樣爲負樣本，又會對熱門項目不公平。此外，訓練數據若來自特定時間段，而測試數據又來自另一時間段，則可能導致對歷史時期的過擬合，無法適應當下潮流。
[^8]: In-batch負採樣是簡化負樣本採樣的方法，對當前訓練批次中每個正樣本（user-item pair）之外的其他項目都作爲負樣本。這樣做可以直接從當前批次獲取負樣本，無需額外採樣操作，也不必維護一個全局負樣本池，降低計算開銷（大規模推薦系統的項目太多，每次訓練都從整個數據集隨機採樣負樣本非常耗時，也容易導致熱門項目倍過度採樣爲負樣本）。這麼做也有缺點，比如負樣本多樣性不足。
[^9]: 估計項目的流行度（被採樣的概率）並用logQ修正模型得分，減少負採樣中對熱門項目的過度懲罰，詳見[Sampling-Bias-Corrected Neural Modeling for Large Corpus Item Recommendations](https://dl.acm.org/doi/abs/10.1145/3298689.3346996)。這裏的logQ correction受啓發自[sampled softmax model](https://www.iro.umontreal.ca/~lisa/pointeurs/importance_samplingIEEEtnn.pdf)。
[^10]: 簡單in-batch隨機負採樣生成的非信息性負樣本往往過於簡單，更均衡的負採樣方法參考：SimANS: Simple Ambiguous Negatives Sampling for Dense Text Retrieval. [[pdf]](https://arxiv.org/pdf/2210.11773)
[^11]: An Empirical Study of Selection Bias in Pinterest Ads Retrieval. [[pdf]](https://dl.acm.org/doi/pdf/10.1145/3580305.3599771)
[^12]: Mixture of Virtual-Kernel Experts for Multi-Objective User Profile Modeling. [[pdf]](https://arxiv.org/pdf/2106.07356)
[^13]: QARM: Quantitative Alignment Multi-Modal Recommendation at Kuaishou. [[pdf]](https://arxiv.org/pdf/2411.11739)
[^14]: CVR(Conversion Rate)：轉化次數和點擊次數的比值。
[^15]: Rethinking the Role of Pre-ranking in Large-scale E-Commerce Searching System. [[pdf]](https://arxiv.org/pdf/2305.13647)
[^16]: COPR: Consistency-Oriented Pre-Ranking for Online Advertising. [[pdf]](https://arxiv.org/pdf/2306.03516)
[^17]: ROC(Receiver Operating Characteristic Curve)曲線即受試者工作特徵曲線，是一種用於評估二分類模型性能的工具，以真陽率TPR爲縱軸、假陽率FPR爲橫軸，對每一個閾值，計算對應的（FPR, TPR）座標點，將這些點按閾值從高到低連接起來，即形成ROC曲線。AUC即ROC曲線下面積，AUC=1即爲完美分類器，AUC=0.5爲隨機猜測。
[^18]: ISPH@k，即場景內購買命中率，衡量的是粗排輸出的前k個候選項中是否包含用戶實際購買的商品。侷限性在於：(1)k等於粗排輸出的全部候選數時，ISPH@k恆爲1，就失去評估意義了。（2）粗排候選集在精排後才能曝光，ISPH@k反映的是整個排序階段的選擇，而非粗排候選集本身的質量。（3）僅捕捉單個場景內的偏好。
