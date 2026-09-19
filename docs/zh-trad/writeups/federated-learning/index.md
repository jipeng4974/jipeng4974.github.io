# 聯邦學習

> 聯邦學習(Federated Learning)是指許多移動設備在一箇中央服務器的編排下協作訓練模型，保持訓練數據離散，避免對用戶數據進行收集，僅將客戶端模型更新上傳中央服務器彙總成新的全局模型的機器學習模式。與in-center的分佈式訓練相比，有其獨特的優勢和挑戰。

---

LLMS index: [llms.txt](/llms.txt)

---

# 聯邦學習
[聯邦學習](https://arxiv.org/pdf/1602.05629.pdf)由McMahan於2016年提出，指的是許多移動設備在一箇中央服務器的編排下協作訓練模型，保持訓練數據離散，避免對用戶數據進行收集，僅將客戶端模型更新上傳中央服務器彙總成新的全局模型的機器學習模式。

聯邦學習的naive實現如下：
1. 中央服務器選定一些clients，讓這些client下載模型。
2. 每個client根據自己的數據計算更新。
3. 每個client將更新（即新的完整模型）上傳到中央服務器。
4. 中央服務器用某種方式（比如取平均）聚合這些模型得到一個全局模型。

## 應對上傳模型開銷大的問題
大量文獻對聯邦學習進行了討論。[Federated Learning: Strategies For Improving Communication Efficiency](https://arxiv.org/pdf/1610.05492.pdf)指出聯邦學習的naive實現的第3步很容易出現通信瓶頸，這是首當其衝的問題，提出了降低上行鏈路通信成本的兩種方法：結構化更新，縮略更新。

聯邦學習的問題可以被形式化地表述爲：學習模型中的參數。全連接層的參數可用一個實數矩陣（這只是爲了簡化問題，所以討論單個矩陣）表示 W ∈ R^(d1×d2)，shape爲(#input × #output)，d1和d2表示輸出維度和輸入維度。卷積層kernel是4d tensor(#input × width × height × #output) ，需reshape到 (#input × width × height) × #output 。

用W(t)表示當前回合(t)的模型，W(i,t)表示本地更新後的模型，所謂更新就是H(i,t) = W(i,t) - W(t)。中央服務器可聚合得到新模型：W(t+1) = W(t) + η(t)H(t)，其中H(t) = Sum(H(i,t))/N，η(t)是learning rate。

### 結構化更新
結構化更新是指直接從一個用少量變量參數化的受限空間裏學習更新，而不是學習整個模型的更新。

所謂結構化更新，也就是impose structure on updates，論文裏提出兩種結構，一種是低秩矩陣，另一種是隨機掩碼。

在結構化更新之低秩矩陣方法中，本地模型的更新H(i,t)必須是一個rank < k的低秩矩陣，這裏k是一個固定的數字。將H(i,t)表示爲H(i,t)=A(i,t)B(i,t)，其中A(i,t)∈ R^(d1×k)，B(i,t)∈ R^(k×d2)，在後續計算中，隨機生成一個A(i,t)視爲常數，然後優化B(i,t)。這樣A的數據就不必上傳，可以坍塌成一個隨機種子，只需上傳B(i,t)。這種優化，本質上是利用降維的壓縮技術，使用矩陣將原始數據進行降維處理，然後使用重構矩陣將降維後的數據重新構建爲原始數據。
A(i,t)每回合都爲每個client隨機生成一次。立刻就得到d1/k的上傳開銷減少。固定B，訓練A，或者同時訓練A和B都試過，效果不如固定A，訓練B。對此的解釋是A可視作重構矩陣（從變換後的向量中重新構建出原始向量），B是投影矩陣（將一個向量投影到另一個向量的子空間上）。固定A訓練B實際上相當於解決這樣一個問題：給定一個隨機的重構矩陣，什麼樣的投影矩陣可以恢復最多的信息？

在結構化更新之隨機掩碼方法中，本地模型的更新H(i,t)必須是用某個預定義的隨機掩碼生成的稀疏矩陣。同樣是每回合對每個client重新生成一次。稀疏掩碼可以坍塌成一個隨機種子，因此只需上傳H(i,t)的非零值和種子。

### 縮略更新
縮略(sketched)更新，學習一個完整模型更新，學成之後，再通過有損的量化、隨機旋轉、子採樣將其壓縮後再發給服務器。

量化：將權重做概率量化，用更小的標量類型充當原始權重的unbiased estimator。

隨機旋轉：其實就是用一個隨機正交矩陣乘一下，防止大部分數據都是0的情形，導致量化效果差。

子採樣：原本上傳的是H(i,t)，subsampling之後只上傳一個隨機子集。

## 完全去中心化聯邦學習在各個領域面臨的挑戰
[Advances and Open Problems in Federated Learning](https://arxiv.org/pdf/1912.04977.pdf)中討論了完全去中心化聯邦學習在算法、隱私、安全、工程等個層面的挑戰:
1. 中央服務器可能成爲瓶頸和單點故障風險點。由此產生p2p/完全去中心化的設計思路。
2. 完全去中心化的算法需應對client可用性和網絡穩定性的侷限。
3. 設計一個試圖達到最快收斂速度的模型平均策略是困難的。
4. 去中心化的場景會導致算法易受惡意攻擊、不可靠數據或標註的威脅。
5. client的通信帶寬和電量不足，，將已有的壓縮算法移植到移動端比較困難。
6. 隱私問題：如何防止一個client重建另一個client的隱私數據。
7. 如何實現的問題：區塊鏈作爲分佈式賬簿本質上是一個最終一致的複製狀態機。不過以太坊這樣的區塊鏈上的數據是公開的，如果要適用於聯邦學習，還需要改造。
8. cross-silo場景（多個組織或公司一起訓一個模型，但數據不能直接共享，比如多個銀行一起訓一個fraud detection模型）對數據進行分區，並增加incentive機制。
9. 通信和壓縮瓶頸。
10. 公平性：聯邦學習引入了新的bias來源——設備型號、地理位置、活動模式、本地數據集大小等。
11. 安全性計算問題：如何應對惡意服務器？如何應對外部攻擊？

### Non-IID數據分佈問題
Non-IID(independent & identically distributed) data問題：樣本的統計屬性沒有均勻分佈，對於任何client-partitioned數據集來說都是常見的。

論文中給出了多種non-identical client分佈（考慮對特徵x，標籤y進行有監督學習，(x,y)~Pi(x,y)即爲client i的本地分佈，P(x,y) = P(y|x)P(x) = P(x|y)P(y)）：
- Feature distribution skew(covariate shift)，即不同client的P(y|x)相同，但特徵的邊際分佈P(x)不同。比如筆跡識別中不同用戶寫相同的字，但筆法、書寫習慣還是不同。
- Label distribution skew(prior probability shift)，即不同client的P(x|y)相同，但標籤的邊際分佈P(y)不同。比如澳洲的日常動物識別應用，標籤裏就會頻繁出現袋鼠，其他地區則不會。
- Same label, different features(concept drift)，即不同client的P(y)相同，但條件分佈P(x|y)不同，相同標籤在不同client上對應到了不同的特徵。比如豪宅，在香港和加州尺度是不同的。
- Same features, different label(concept shift)，即不同client的P(x)相同，但條件分佈P(y|x)不同，相同特徵被標註成立不同標籤。比如有人把熊貓標註爲寵物，也有人把熊貓標註爲猛獸。
- Quantity skew or unbalancedness，不同的client的數據量差異大。

違反independence同樣常見，因爲client的分佈很容易收到觸發訓練的約束條件的影響：比如很多訓練是利用夜間睡眠時間跑的，那就導致往往一個經度的地區的clients更容易遇到一起。

應對Non-IID數據，一個可行的方法是用一個不涉及隱私數據的全局共享小數據集做數據增強。此外，還可以限制一個用戶每天能做的貢獻上限，避免量上面的不平衡。此外，某些場景還可以把Non-IID從bug轉化爲feature，就訓一個本地客製化的專用模型出來，提供個性化服務，而不是最終產生一個全局模型。文章後面還介紹了Non-IID數據集上的優化算法和收斂速率。

### 應對隱私問題：Split Learning
Split Learning是模型執行路徑層面的橫切，同時適用於訓練和推理：最簡單場景是讓每個client一直前向算到某個特定的cut layer停下，將cut layer的輸出（smashed data）傳給中央服務器或peer接着算，於是就完成了無需數據共享就實現的前向傳播。類似地，梯度反向傳播是從最後一層到cut layer停下，僅將cut layer的梯度回傳給client。這樣整個過程中其他節點都不會直接訪問本地數據。

考慮到cut layer的權重本身也能一定程度上反映底層的數據現實，Split Learning是否能提供形式化的隱私承諾仍然是個開放問題。

## 應對通信延遲高的問題
還有一個關鍵問題——高通信延遲，由於無線和長距離傳輸的特性而無法迴避，但在之前的論文中沒被很好地address。[Delayed Gradient Averaging: Tolerate the
Communication Latency in Federated Learning](https://dga.hanlab.ai/assets/neurips21_dga.pdf)一文提出了一種延遲進行梯度平均的算法，用16節點是樹莓派集羣模擬現實世界中的移動節點和無線網絡環境做了個實驗，經驗性地證明應用延遲梯度平均可以使聯邦學習過程容忍高網絡延遲，同時還不犧牲準確度。

In-center環境下，同一個機櫃的延遲<1us，同機房則是ms級別。無線環境大概是20ms，跨洋連接則至少100ms。在解決帶寬問題後，延遲就成爲最大瓶頸。這篇論文提出的DGA(Delayed Gradient Aggregation)算法的核心思路是延遲梯度平均到未來的某個迭代，即模型更新時接收過時的平均梯度，從而允許通信和計算流水線化。論文將問題形式化爲：最小化隨機函數的和。

![DGA](https://wujipeng.com/img/DGA.png)

N表示client數量，fi表示client i的stochastic損失函數。隨機變量ζi關聯一個mini-batch樣本。


![DGA2](https://wujipeng.com/img/DGA2.png)

算法的主要思路是允許averaging通信過程中做本地更新（averaging通信和本地更新並行，）。FedAvg中clients在每輪結束髮送參數到彼此，等averaging結束再開啓下一輪。DGA裏把averaging barrier延遲到了後續迭代(iteration，指的是本地更新的迭代)。因此clients可以立刻開啓下一輪(round，指的是最外層循環，即一輪更新)。第一輪下收到外部信息時迭代已經發生了D次，延遲了D個迭代後進行梯度修正。理想情況下不存在通信延遲，D=0時，DGA恢復成最初的FedAvg。

1. clients在第t輪彼此發更新。
2. clients在本地更新後繼續用最新的本地參數繼續本地更新。(averaging通信延遲 > 單次甚至若干次本地更新)
3. 當其他client的第t輪信息到達，則client已進行了D次額外本地更新。
4. 將本地t輪梯度替換爲接受到的平均梯度。

在最寬泛的場景下（延遲極高），延遲梯度可能要幾個輪次之後才能抵達。這就需要將延遲參數D表示爲D = sK + r，其中s>=0, r <=K。DGA仍能保證不同client只在最近D個梯度上是不同的，t-D輪之前的梯度都是一樣的。
