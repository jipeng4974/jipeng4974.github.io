# 筆記：推薦系統研究現狀的理解

> 本文是對孫愛欣教授的分享——“推薦系統研究現狀的理解”的筆記，對大致內容進行了摘要，並收集了提及文獻的鏈接——可以一窺推薦系統領域學術研究的現狀。

---

LLMS index: [llms.txt](/llms.txt)

---

推薦問題可定義爲在合適的時間把合適的物品（items）推薦給合適的用戶。

通常來說，推薦任務所用數據集即用戶-物品交互矩陣。推薦系統基於這個數據集推測用戶興趣，把結果放到線上進行測試（學術界沒有這個條件，只能線下評測），進行評測。

## What's Missing in User-Item Interaction Datasets
> 我們通常將推薦任務理解爲如何在一個靜態的用戶-物品交互矩陣中預測缺失的數據，而不是在特定場景中的動態環境下預測用戶的下一次交互。

值得注意的是，在用戶-物品交互矩陣數據集中，時間的維度坍縮了，真實互動場景中的種種約束也坍縮了。

推薦系統文章中，70%使用了MovieLens數據集，但它真的能還原推薦場景嗎？未必，因爲MovieLens在一次初始化過程中完成用戶對過去多年觀影體驗的回憶——可能有很多遺漏、遺忘，也忽略了上映時間、價格等現實因素。然而現實場景中用戶的興趣是逐漸形成的，受制於各種現實約束，其觀影決策還不可避免地受到上映時間、票價、以及興趣是否發生變化等諸多因素影響。

## A Worrying Analysis of Current Practice 
> 基於我們過去五年在推薦系統評測和數據集分析方面的工作，我們重新審視推薦系統的問題定義，併爲缺乏共識的現象提供一種解讀。

Dacrema et al., 2019[^1]認爲深度學習新模型的效果一般般。

Bauer et al., 2024[^2]做了一個綜合，分析了數據集，發現用的數據集非常集中，集中在MovieLens、Amazon Reviews等主流數據集，這些數據集普遍偏舊。大部分文章都是提出新的模型。還有一些專注在評測標準。

Ivanova et al., 2023[^3]認爲推薦領域用哪個baseline其實並沒有共識，一部分原因是每年推薦系統裏有5k篇文章，沒人能讀完所有文章，於是審稿者之間未形成共識，有的審稿者認爲某些方法特別好，一定要納入baseline，另一些審稿者則有不同觀點。比如nearest neighbor雖然簡單，經過良好調參後卻是非常強的baseline，在很多場景比複雜模型表現得好很多，但很多文章不會把nearest neighbor列入baseline。大家都認爲它是幾十年前的方法，不值得比較。

不僅baseline不統一，即使baseline統一，也有一個調參的問題，如Shehzad, Jannach, 2023[^4]所說，當你提出自己的模型時，非常注重調參，和別人比時卻沒有非常精細地調參。

McElfresh et al., 2022[^5]做了非常大規模的調研，在85個數據集上比較了24個算法的315個指標，得出了令人震撼的結論：這些算法並不能泛化，在某個數據集上很好，下一個可能就不好。每個算法都可能排第一第二，最差都能排到倒數第幾。最後發現最強的算法竟然是Item-kNN！


## Data Leakage in Train/Test Split
Sun, 2022[^6]中對20~22年88篇RecSys會議文章的``train/test split``做了一個梳理，發現34%的論文用的是``random split``（隨機分），25%用的是``leave-one-out``（最後一個交互做成測試，之前的是訓練），19.5% ``single time point``，17% ``simulation-based online``，4.5% ``sliding window``。理論上最完美的train/test split方法是嚴格遵循時間線，在時間線上取某個時間點之前的做訓練集，之後的做測試集，然後慢慢把這個時間點往後推，每個時間點的選取生成了相應的訓練集和測試集，時間點越往後推，訓練集越多測試集越少——可惜實際操作起來很難做到，大部分文章採用的``random split``和``leave-one-out``有很強的信息泄露。

以``leave-one-out``爲例，每個用戶都只取最後一次交互做測試集，問題就在於不同用戶的最後一個交互的時間可能大不同——假設某個物品在特定時間非常火，``popularity``(流行度是推薦系統中最簡單的baseline，就是對物品的交互次數進行排序)排序非常高，但某個用戶最後一次交互時間甚至在這個特定時間之前，那給這個用戶推薦一個未來爆火的物品顯然並不合理。Ji et al., 2020[^7]就重新審視了這一問題，將``popularity``修正爲用戶最後一次交互時間點的“當時流行度”後，``popularity``置信度可以提升70%。

在Ji et al., 2020[^8]的研究中指出：幾乎所有的ML/DL模型中，尤其是離線評測的推薦系統模型中，都存在類似的數據泄露問題——模型無意中使用了未來數據進行訓練，學到了本不應該存在的用戶-物品交互。``BPR``、``NeuMF``、``LightGCN``、``SASRec``均未從機制上避免這種數據泄露。這個研究也通過實驗證明了這種數據泄露確實會顯著影響模型的推薦準確率，且這種對準確率的影響是不可預測的。

## Recommendation should be Task-specific & Dynamic
用戶決策涉及通用偏好和當下context因素。當下context是非常task-specific且非常動態的。這也使得推薦任務具備了這一特徵。

很大程度上，現有的推薦系統都只侷限在通用偏好層面。回顧數據層面，現有數據集，即各種用戶-物品交互矩陣，顯然丟掉了context，只能捕捉通用偏好。模型層面，訓練是基於決策結果的，而非決策過程本身，這也決定了模型只能學到用戶的通用偏好。而評測端，自然而然也只能對通用偏好進行評測。

工業實踐中，推薦系統應該是一個檢索問題。在這個檢索問題中，query包含通用偏好、當前context這兩種動態更新的信息；item collection也是動態更新的；ranking則旨在提升決策質量。

考慮到context因素，不同場景的推薦系統，如食物推薦、電影推薦、電商推薦、賓館推薦，也應該有非常不同的實現，分開建模。有的場景選項固定，有的場景就是適合重複，有的場景則偏重探索。不同場景的輸入甚至也有區別。比如外賣推薦除了user id之外，必需提供收貨地址、早餐/午餐/晚餐這種信息。

## Conclusions and What's Next
孫教授近期的一篇文章Sun, 2024[^9]，對推薦系統的問題定義進行了重新思考，認爲當前的推薦系統研究對推薦問題做了過度簡化，以至於學術界提出的幾乎所有的方案都不太適應現實世界中的具體任務。對未來的研究方向進行了一些研判：
- 預計不會有贏家通喫的模型，未來依舊是每個模型在自己的論文裏無敵。
- 應將推薦系統問題細化，短視頻有短視頻賽道，電商有電商賽道，新聞有新聞賽道，針對不同賽道，設計、評測新的模型。別再用MovieLens去評測電商推薦了！
- Item-kNN仍然會是很強的baseline。只不過對nearest和neighbor的定義需要更好的特徵工程。

[^1]: Are we really making much progress? A worrying analysis of recent neural recommendation approaches [[arxiv]](https://arxiv.org/abs/1907.06902)
[^2]: Exploring the Landscape of Recommender Systems Evaluation: Practices and Perspectives [[pdf]](https://arxiv.org/pdf/2311.05232.pdf) 
[^3]: RecBaselines2023: a new dataset for choosing baselines for recommender models [[arxiv]](https://arxiv.org/abs/2306.14292)
[^4]: Everyone’s a Winner! On Hyperparameter Tuning of Recommendation Models [[pdf]](https://dl.acm.org/doi/pdf/10.1145/3604915.3609488)
[^5]: On the Generalizability and Predictability of Recommender Systems[[arxiv]](https://arxiv.org/abs/2206.11886)
[^6]: Take a Fresh Look at Recommender Systems from an Evaluation Standpoint [[arxiv]](https://arxiv.org/abs/2210.04149)
[^7]: A Re-visit of the Popularity Baseline in Recommender Systems [[arxiv]](https://arxiv.org/abs/2005.13829)
[^8]: A Critical Study on Data Leakage in Recommender System Offline Evaluation  [[arxiv]](https://arxiv.org/abs/2010.11060)
[^9]: Beyond Collaborative Filtering: A Relook at Task Formulation in Recommender Systems [[arxiv]](https://arxiv.org/abs/2404.13375)
