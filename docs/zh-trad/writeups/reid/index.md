# 語義深度和音樂識別

> 本文提出語義深度指標，從歸納偏置、數據驅動、性能工程、級聯排序等角度討論Music Re-ID。

---

LLMS index: [llms.txt](/llms.txt)

---

從基於純先驗的手工特徵檢索，到純數據驅動的深度度量學習，再到給模型提供一些先驗領域知識輔助學習，是螺旋上升的過程。

即便是代表極致數據驅動的預訓練MLLM，也普遍將log mel頻譜而非原始波形送入模型的audio encoder，這本身就是一種基於人耳聽覺和心理聲學的歸納偏置。

音樂是天然高度結構化的，音樂檢索領域的標註數據天然是稀缺且不公開的，因此基於先驗知識引入歸納偏置是合乎邏輯的，可以讓模型不必從聲學、信號處理、音樂理論學起。


# 歸納偏置

## 選取合適的語義深度
語義深度(semantic depth)是我生造的指標。

從現實世界採樣信號到人的主觀認知之間存在若干個抽象層次，自信號層面每下潛一個抽象層次，則語義深度+1。

音樂表徵領域中，不妨將語義深度定義如下：

- 語義深度爲0的通用音頻信號特徵，無音樂語義，如peaks、spectra flux、shazam/quad fingerprints、mfcc。

- 語義深度爲1的音樂微觀結構特徵，攜帶音樂語義，如harmonic peaks/ridges、attack onsets、beat positions、spectral envelope、pitch contour。

- 語義深度爲2的音樂中觀結構特徵，往往覆蓋5~30s有記憶點的motif，如melody、rhythm、groove。

- 語義深度爲3的音樂宏觀結構特徵，往往覆蓋全曲，能挖掘更深層的敘事、情感、結構、風格、關聯信息、特定語境下的信息。


## 選取恰當的輸入形態
如果確定應用場景中query和doc幾乎都是音樂，輸入層面自然可以選用合乎樂理的cqt頻譜，並根據該場景所需的語義深度選取對應的音頻切片粒度。

## 捕捉不變量的歸納偏置
其次可以將某些手工特徵作爲輔助信息提供給模型。這些手工特徵大多屬於頻譜的某種壓縮表示，關注特定語義深度爲1的音樂不變量，例如pitch contour，harmonic peaks的拓撲關係。

# 數據驅動
## 硬樣本挖掘
確定一個fpr足夠低的得分閾值，基於KNN檢索取top k，得分超閾值則爲positive(用於擴充sim group)，其他爲negative。

從top k中negative cands中剔除掉hard negative（得分大於sim group平均相似度而小於閾值，這部分只能依賴人工標註），留下得分低於sim group平均相似度的semi-hard negatives。

## 生成數據
音樂或視頻生成模型續寫、改編樣本。

## 數據增強
變速變調、擾動加噪、混音。

# 性能工程

## Encoder輕量化
當query足夠短，比如普遍是僅10s的音頻clip時，可fit in cnn[^1]感受野，且該cnn架構進行過恰當的attention hybrid改造[^2]，transformer encoder的長程相關性建模優勢就不存在了。

## MRL
MRL（套娃表徵學習）允許表徵向量的前若干維構成的子向量是直接可用於檢索的，這顯然優於獨立的PCA降維或訓練時加個線性層降維，更加靈活，值得嘗試。

MRL主要用於超大規模檢索中降低向量索引的存儲和檢索開銷。

MRL在推理階段也降低了分類頭實際激活參數量，在class數量特別多的場景可稍稍降低檢索成本。

## QAT
和MRL相比，QAT固然也減小向量尺寸，但主要是爲了發揮硬件低精度推理的性能優勢，大幅降低推理成本。QAT和MRL的收益和損失均正交。

# 級聯排序
多數場景下，metrics learning + 大規模ANN向量檢索足以解決音頻檢索問題。

少數困難場景，需要引入級聯檢索，用embedding模型做粗召，用某種reranker做精排。
- reranker可以是一個基於原有embedding模型的cross-encoder reranker。
- 或基於預訓練MLLM在rerank任務上做後訓練。

[^1]: 這裏的CNN只是大類，實際會使用Resnet中的ResNeSt變種。ResNeSt比Resnet多了Split-Attention Conv。

[^2]: 在ResNeSt-50基礎上，還可以在layer2, layer3插入NonLocal自注意力模塊，進一步提升上下文理解能力。此外，低層還會用IBN替代BN，即一半通道走InstanceNorm，另一半SyncBN。引入InstanceNorm可去風格，抑制模型對能量包絡的學習。
