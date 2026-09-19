# The Little Book 書評

> 正如DDIA可被視爲分佈式系統方向的入門教程，LBDL是理想的深度學習101。

---

LLMS index: [llms.txt](/llms.txt)

---

"The Little Book of Deep Learning"([```LBDL```](https://fleuret.org/francois/lbdl.html))是François Fleuret寫的一本適配手機屏的書，精簡扼要地面向stem背景讀者介紹深度學習。正如```DDIA```可被視爲分佈式系統方向的入門教程，```LBDL```是理想的深度學習101。

![tlb](https://wujipeng.com/img/tlb.jpg)

精簡，或者說壓縮，正是深度模型的strength，也是這個信息過載時代的virtue。用A4紙打印這個小冊子，讀起來非常舒適。

---

接下來是知識內化和梳理。

## 【一】概述 

高維信號難以用規則系統分析，而深度網絡則克服了這個困難，用具有大量權重的深層映射擬合出一個足夠好（loss足夠低）的近似函數——這個函數可以是高維信號到連續向量（迴歸）或離散值（分類）的映射，也可以是一種概率密度函數，總之，它能從數據分佈中學習到某種緊湊且有區分能力的表徵。

若數據樣本不足，即使訓練數據上表現良好，也可能在真實應用中效果不佳，這就是過擬合。
若模型能力不足，無法適應多變場景、準確捕捉輸入輸出的關係，訓練時loss就高，則是欠擬合。

機器學習模型可以粗粒度地分爲3類：
1. 迴歸模型：有監督，訓練數據是輸入信號和ground-truth數值的pairs，將高維信號映射到某個向量。
2. 分類模型：有監督，訓練數據是輸入信號和標籤的pairs，將高維信號映射到有限標籤集上。
3. 概率密度函數模型：無監督，訓練數據就是輸入信號本身。

## 【二】訓練 
### 損失函數
所謂訓練，就是降低訓練集上預測函數的損失函數（loss，記作$\mathscr{L}$）的過程。

損失函數如何定義？對連續數值來說，均方差是一個標準選擇。對概率密度來說，則用似然值——可令$\mathscr{L}=-\sum f(x;w)$，其中$f(x;w)$是各個訓練樣本的標準化log概率。對分類任務來說，一般用交叉熵。

何爲交叉熵？分類模型爲N個類輸出N個logits（其實LLM也是這樣，爲vocabulary裏每個token生成對應的logits，表示未標準化的log概率），logits經過softmax，得到後驗概率$P(Y=y|X=x)$，這是裸概率，各個類的概率加起來和爲1。令$\mathscr{L}=-\frac{1}{N} \sum_{n=1}^N logP(Y=y_n|X=x_n)$，這個$\mathscr{L}$即爲交叉熵。交叉熵最小化，則真類別的概率最大化。

在度量學習中，雖然預測的值是連續的，但實際監督形式是分級，因爲度量學習的目標是學習出樣本之間可比較的距離，比如A、B、C三個點，其中A和B是同一個人臉的不同側面，C是另一個人，那就要求AB之間距離小於AC。因此度量學習一般採用contrastive loss或triplet loss。

損失函數通常只是一個代理指標，而非實際性能指標，以分類任務爲例，顯然直接性能指標應該是分類錯誤率，只不過這個指標的梯度沒有攜帶有效指導信息——錯誤率函數和模型權重是完全剝離的，知曉錯誤率的變化不能在訓練中幫助模型減小錯誤率。

損失函數還可以被設計爲依賴於模型權重，從而對模型權重進行某種約束和控制。比如權重衰減（weight decay），一種防止過擬合的正則化技術，給損失函數里增加了一項模型權重的平方和，從而懲罰大的權重數值，偏好小的數值，進而減少訓練數據對模型權重取值範圍的影響。這麼做會使訓練集上性能下降，但有利於在未見過的數據集上更好地泛化。

### 自迴歸模型
自迴歸模型是NLP/CV等領域處理離散序列的關鍵方法。原理是利用條件概率的鏈式法則: 
$$P(A\cap B)=P(A) P(B|A)$$
$$P(A\cap B\cap C)=P(A) P(B|A) P(C|A\cap B)$$

自迴歸模型輸入是已有的T個token（每個token取值範圍是大小爲K的vocabulary集合），輸出是K個候選token的logits。

token詞彙域有限的場景是可計算的，條件概率的鏈式分解又使計算量降低——採樣下一個token時，可以利用上一個token的概率，最終能生成符合聯合概率分佈的token序列。

訓練自迴歸模型可以遍歷各個步驟，把每一個邏輯時間節點上模型預測和真正的下一個token的交叉熵快照加起來，形成交叉熵loss。減小這個loss，即增大每個邏輯時間節點上模型預測token的似然。實際上監控的往往不是交叉熵，而是交叉熵(H)的指數，即困惑度perplexity(PPL)，PPL = 2^H。相比交叉熵，困惑度是歸一化的，並不依賴輸入序列的長度。

訓練時，每個時刻都需要重新計算之前已經算過的，考慮到總體邏輯時間/步驟數目往往相當長，成百上千，甚至上萬，這樣的計算顯然非常低效。解決方案是設計一個一次性預測所有邏輯時間（T）上的logits向量的模型——$f: \{1,...,K\}^T \rightarrow \mathbb{R}^{T\times K}$，並確保t時刻的輸入$x_t$對應的logits $l_t$只依賴於$x_1, x_2, x_3, ... x_{t-1}$。這種模型即因果模型（causal models），其原則是不讓未來影響過去。
 
![causal](https://wujipeng.com/img/causal.png)

因果模型訓練時可以用完整的序列計算output，一次性最大化序列中所有token的概率，最終也等價於最小化per-token交叉熵。

自然語言處理中有一個重要技術細節，即如何進行token的表示，可以是最低粒度的單符號，也可以說整個詞。進行token表示的算法過程叫做tokenizer。一個標準做法是Byte Pair Encoding[Sennrich et al., 2015][^1]。

### 梯度下降
除了線性迴歸這種簡單特例，一般最優權重$w^*$不會有closed-form expression。這種情況下最小化函數的工具是梯度下降：將權重初始化爲隨機的$w_0$，然後反覆迭代，每次迭代都朝梯度方向修改權重使loss逐步降低，即每次迭代都令$w_{n+1} = w_n - \eta \nabla \mathscr{L}_{|w}(W_n). $ 其中$\eta$即學習率，如果設置得太小，訓練可能太慢，而且容易卡在局部最小，如果設置得太大，則容易在最低點附近左右橫跳。

![gd](https://wujipeng.com/img/gradient_descent.png)

對於每個點$w$來說，梯度$\nabla \mathscr{L}_{|w}(w)$就是能最大化$\mathscr{L}$增量的方向。因此梯度下降就可以通過每次迭代中減去學習率*梯度的值，使所有迭代串聯起來可形成一個接近最優的最小化$\mathscr{L}$路線。

實踐中，所有loss均能表示爲多個小樣本甚至單樣本loss的均值：$\mathscr{L} = \frac{1}{N} \sum_{n=1}^N \ell_n(w) $，其中$\ell_n(w)=L(f(x_n;w),y_n)$，因而梯度可表示爲下式：

$$\nabla ℒ_{|w}(w) = \frac{1}{N} \sum_{n=1}^N \nabla \ell_{n|w}(w)$$

全量計算梯度開銷較高，可以用局部求和估計全量求和（要求做好數據shuffling，數據的stochasticity消除估計的偏倚）。爲了讓計算能放進內存，標準的做法是把完整訓練集分成相當多（可以是百萬級）batches，從每個batch得到一個梯度的估計，然後根據這個估計去更新權重，這種做法即mini-batch SGD(stochastic gradient descent)。這種算法有很多變種，比如Adam[Kingma and Ba, 2014][^2]。

### 反向傳播
給定$\ell(w)=L(f(x;w),y)$，怎麼計算$\nabla\ell_{|w}(w)$呢？考慮到$f$和$L$都是標準張量運算的組合，它們與任何數學表達式一樣，基於鏈式法則可以得到其表達式。

![bp](https://wujipeng.com/img/bp.png)

簡單起見，將一個深度爲$D$的模型表示爲$f = f^{(D)} \circ f^{(D-1)} \circ ... \circ f^{(1)}$。則前饋過程即按順序計算$x^{(d-1)} \rightarrow x^{(d)}$，即$x^{(d)} = f^{(d)}(x^{(d-1)};w_d)$，直到最終得到$x^{(D)}$作爲模型輸出。

反向傳播過程則是反過來計算
$\nabla\ell_{|x^{(d-1)}} \leftarrow \nabla\ell_{|x^{(d)}}$
以及
$\nabla\ell_{|w_d} \leftarrow \nabla\ell_{|x^{(d)}}$，
其中
$\nabla\ell_{|x^{(d-1)}}$[^3]
是
$\nabla\ell_{|x^{(d)}}$[^4]
和
$J_{f^{(d)}|x}$[^5]
乘積。

而我們在訓練過程中實際關心的另一個梯度$\nabla\ell_{|w_d}$是$\nabla\ell_{|x^{(d)}}$和$J_{f^{(d)}|w}$[^6]乘積。

深度學習訓練框架主要處理的就是隱藏反向傳播梯度計算的複雜度，即提供自動求導/計算求導能力，該技術在深度學習之前也有廣泛應用，詳見AutoGrad [Baydin et al., 2015][^7]。

顯然反向傳播的矩陣計算量兩倍於前饋推理（每層多了一次權重的反向傳播）。反向傳播的內存需求也遠大於前饋推理，因爲每一層的$x^{(d)}$都要保留在內存中，而非推理則不需要保留，只要留着最新的。解決內存佔用過高的技術包括：reversible layers[Gomez et al., 2017][^8]和checkpointing[Chen et al., 2016][^9]。

深度模型的一個問題是梯度消失（見[Glorot and Bengio, 2010][^11]），即經過很多次反向傳播後，數值變得太大或太小。常規應對做法是gradient norm clipping[Pascanu et al., 2013][^10]。

### 自監督訓練
GPT在大規模無標註訓練集上訓練就足以處理很多任務，比如翻譯（見[Radford et al., 2019][^12]），這就是典型的自監督訓練應用，其最重要的優勢是可以利用超大規模的未標註數據，將訓練數據規模的邊界再向前推進。

## 【三】構件
### 線性層
全連接層是最基礎的線性層構件，可用$D \times D'$的矩陣$W$和一個bias向量$b$表示。它實現了一個能泛化到任意張量形狀的仿射變換，給定任意輸入$X$，其形狀爲$D_1 \times \dots \times D_k \times D$，全連接層計算得到一個輸出$Y$，其形狀爲$D_1 \times \dots \times D_k \times D'$：$\forall d_1,\dots,d_K, Y[d1,\dots,d_K] = WX[d1,\dots,d_K] + b $


全連接層處理高維數據時，參數量太大。此外全連接層假設輸入輸出之間存在複雜非線性關係，忽視了更簡單的結構化規律[^13]。然而高維信號普遍有這種強結構，比如圖片兼具short-term關聯和抵抗變換、縮放、對稱的統計學靜態性。相較而言，卷積層能更好地捕捉信號的空間結構——因爲卷積層權重被輸入信號的不同部分共享，這些權重因而能學習到某種局部空間結構，比如圖片的邊緣、形狀、角。多層卷積是高維信號（圖片、聲音）的常用降維工具。

一維卷積以$D \times T$張量$X$爲輸入，對每個$D\times K$子張量施加仿射變換$\phi(\cdot;w): \mathbb{R}^{D\times K} \rightarrow \mathbb{R}^{D'\times 1}$，將$D'\times 1$結果依次存入$Y$中。 

一維反捲積則是以$D \times T$張量$X$爲輸入，對每個$D\times 1$子張量施加仿射變換$\phi(\cdot;w): \mathbb{R}^{D\times 1} \rightarrow \mathbb{R}^{D'\times K}$，將結果加起來形成的$D'\times K$張量存入$Y$中。

下圖中，$D=3, K=5, D'=4$。

![1dconv](https://wujipeng.com/img/1dconv.png)

![2dconv](https://wujipeng.com/img/2dconv.png)

一維卷積往往用於處理序列數據或時序數據。二維卷積則常用於處理圖片，或其他以2D矩陣爲輸入的任務。轉置卷積/反捲積則主要用於GAN、VAE這樣的生成式模型中，從一個低維特徵膨脹到一個高分辨率圖片。

### 激活函數
如果模型中僅使用線性組件，則整體也是線性操作，因此必須要引入非線性性，這種非線性性通常由激活函數實現。最常用的激活函數是ReLU [Glorot et al., 2011][^14]。ReLU之前則是雙曲正切函數Tanh。

還有一些激活函數思路和ReLU差不多，保證正數不變，壓縮負值，如：Leaky ReLU[Maas et al., 2013][^15]，GELU [Hendrycks and Gimpel, 2016][^16]。

### 池化
池化是降維，減少信號大小的經典策略，將若干鄰近值取max或avg合併出一個值。

### 隨機失活
Dropout[Srivastava et al., 2014][^17]層無可訓練參數，只有一個超參$p$，在訓練時用於以概率$p$隨機關閉一些neuron，避免個別neuron對整體的影響，迫使其他neuron用略微不同的權重代勞。因此dropout是訓練時防止過擬合的正則化工具。推理時dropout是關閉的。

### 歸一化層
歸一化可用於抵抗梯度消失。最主要的歸一化層是Batch Normalization[Ioffe and Szegedy, 2015][^18]，由超參$D$和可訓練參數$\beta_1, \dots,\beta_D$和$\gamma_1, \dots,\gamma_D$組成。給定一批$D$維樣本$x_1, \dots, x_B$，先計算每個維度的均值$m_d = \frac{1}{B} \sum_{b=1}^B x_{b,d}$ 和方差 $v_d  = \frac{1}{B} \sum_{b=1}^B (x_{b,d} - m_d)^2$。

然後再對每個b，計算均值爲0方差爲1的歸一化值$z_{b,d} = \frac{x_{b,d} - m_d}{\sqrt{v_d + \epsilon}} $，再算出最終結果$y_{b,d} = \gamma_d z_{b,d} + \beta_d$，這個最終值的均值爲$\beta_d$，方差爲$\gamma_d$。

![norm](https://wujipeng.com/img/norm.png)

### 殘差連接
跳躍連接（skip connections，見[Long et al., 2014][^19]; [Ronneberger et al., 2015][^20]）同樣可對抗梯度消失。實際上跳躍連接不是一個layer，而是一種讓某些層的輸出跳過一些中間層嫁接到後面的設計。這種設計允許更原始的信號在更後面的層中得到“反思”。

跳躍連接的實用實現是殘差連接（residual connections），直接把兩種信號求和，而且跳躍的層數不算多。這種設計允許信號在穿越某些原本會梯度消失的層時得以倖存。基於殘差連接，何凱明構建了ResNet[He et al., 2015][^21]，Google設計了Transformer[Vaswani et al., 2017][^22]。

### 注意力層
已有的組件缺乏將局部信息和張量中較遠位置的信息結合起來的能力，Attention Layer則專長於此道——通過爲所得張量的每個組件與輸入張量的每個組件計算注意力得分，不受局部性約束地，在整個張量範圍內對特徵進行平均[^22]。

給定$N^Q\times D^{QK}$維queries張量$Q$，$N^{KV}\times D^{QK}$維keys張量$K$，$N^{KV}\times D^V$維values張量$V$，通過```Attention操作```$att(K,Q,V)$計算得到$N^Q\times D^V$維的張量Y：

$$Y = att(K,Q,V) = \underbrace{softargmax(\frac{QK^T}{\frac{1}{\sqrt{D^{QK}}}})}_A V$$

整個過程分兩步，第一步先計算每個query index $q$和每個key index $k$的注意力得分，即queries和keys點乘後的```softargmax```結果：$A_{q,k} = \frac{exp(\frac{1}{\sqrt{D^{QK}}} Q_q \cdot K_k ) }{\sum_l exp(\frac{1}{\sqrt{D^{QK}}} Q_q \cdot K_l)}$，其中$\frac{1}{\sqrt{D^{QK}}}$是一個縮放參數，用於保證取值範圍在$D^{QK}$變大時大體不變。

![att](https://wujipeng.com/img/attention.png)

得到注意力得分$A_{q,k}$後，再進行第二步計算：$Y_q = \sum_k A_{q,k}V_k$。注意力得分即queries和keys之間的匹配程度，匹配程度越高，該權重就越高。如果某個query和某個key匹配度達到極限，注意力得分接近1，則直接拿到這個key對應的value。如果和好幾個key都有中等水準的匹配，則按注意力得分做加權平均。

## 其他話題
```LBDL```还讨论了各种深度学习模型架构和应用，如多层感知机、卷积网络、注意力模型、RNN、Autoencoder、GAN、图神经网络、GPT、Diffusion。


[^1]: R. Sennrich, B. Haddow, and A. Birch. Neural Machine Translation of Rare Words with Subword Units. CoRR, abs/1508.07909, 2015. [[pdf]](https://arxiv.org/pdf/1508.07909).
[^2]: D. Kingma and J. Ba. Adam: A Method for Stochastic Optimization. CoRR, abs/1412.6980, 2014. [[pdf]](https://arxiv.org/pdf/1412.6980).
[^3]: $\nabla\ell_{|x^(d-1)}$即$f^{d-1}$的变量$x^{d-1}$对应的损失函数梯度。
[^4]: $\nabla\ell_{|x^(d-1)}$即$f^{d}$的变量$x^{d}$对应的损失函数梯度。
[^5]: $J_{f^{(d)}|x}$即第d个layer函数$f^{(d)}$相对变量x的Jacobian，雅可比矩阵，即函数的一阶偏导数以一定方式排列成的矩阵。
[^6]: $J_{f^{(d)}|w}$即第d个layer函数$f^{(d)}$相对权重w的Jacobian。
[^7]: A. Baydin, B. Pearlmutter, A. Radul, and J. Siskind. Automatic differentiation in machine learning: a survey. CoRR, abs/1502.05767, 2015. [[pdf]](https://arxiv.org/pdf/1502.05767).
[^8]: A. Gomez, M. Ren, R. Urtasun, and R. Grosse. The Reversible Residual Network: Backpropagation Without Storing Activations. CoRR, abs/1707.04585, 2017. [[pdf]](https://arxiv.org/pdf/1707.04585).
[^9]: T. Chen, B. Xu, C. Zhang, and C. Guestrin. Training Deep Nets with Sublinear Memory Cost. CoRR, abs/1604.06174, 2016. [pdf](https://arxiv.org/pdf/1604.06174).
[^10]: R. Pascanu, T. Mikolov, and Y. Bengio. On the difficulty of training recurrent neural networks. In International Conference on Machine Learning (ICML), 2013. [pdf](https://proceedings.mlr.press/v28/pascanu13.pdf).
[^11]: X. Glorot and Y. Bengio. Understanding the difficulty of training deep feedforward neural networks. In International Conference on Artificial Intelligence and Statistics (AISTATS), 2010. [pdf](https://proceedings.mlr.press/v9/glorot10a/glorot10a.pdf).
[^12]: A. Radford, J. Wu, R. Child, et al. Language Models are Unsupervised Multitask Learners, 2019. [[pdf]](https://d4mucfpksywv.cloudfront.net/better-language-models/language_models_are_unsupervised_multitask_learners.pdf).
[^13]: 这就是所谓的全连接层的```inductive bias```。
[^14]: X. Glorot, A. Bordes, and Y. Bengio. Deep Sparse Rectifier Neural Networks. In International Conference on Artificial Intelligence and Statistics (AISTATS), 2011. [[pdf]](http://proceedings.mlr.press/v15/glorot11a/glorot11a.pdf).
[^15]: A. L. Maas, A. Y. Hannun, and A. Y. Ng. Rectifier nonlinearities improve neural network acoustic models. In proceedings of the ICML Workshop on Deep Learning for Audio, Speech and Language Processing, 2013. [[pdf]](https://ai.stanford.edu/~amaas/papers/relu_hybrid_icml2013_final.pdf).
[^16]: D. Hendrycks and K. Gimpel. Gaussian Error Linear Units (GELUs). CoRR, abs/1606.08415, 2016. [[pdf]](https://arxiv.org/pdf/1606.08415). 
[^17]: N. Srivastava, G. Hinton, A. Krizhevsky, et al. Dropout: A Simple Way to Prevent Neural Networks from Overfitting. Journal of Machine Learning Research (JMLR), 15:1929–1958, 2014. [[pdf]](https://jmlr.org/papers/volume15/srivastava14a/srivastava14a.pdf).
[^18]: S. Ioffe and C. Szegedy. Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift. In International Conference on Machine Learning (ICML), 2015. [[pdf]](http://static.googleusercontent.com/media/research.google.com/en//pubs/archive/43442.pdf). 
[^19]: J. Long, E. Shelhamer, and T. Darrell. Fully Convolutional Networks for Semantic Segmentation. CoRR, abs/1411.4038, 2014. [[pdf]](https://arxiv.org/pdf/1411.4038). 
[^20]: O. Ronneberger, P. Fischer, and T. Brox. U-Net: Convolutional Networks for Biomedical Image Segmentation. In Medical Image Computing and Computer-Assisted Intervention, 2015. [[pdf]](https://arxiv.org/pdf/1505.04597.pdf).
[^21]: K. He, X. Zhang, S. Ren, and J. Sun. Deep Residual Learning for Image Recognition. CoRR, abs/1512.03385, 2015. [[pdf]](https://arxiv.org/pdf/1512.03385).
[^22]: A. Vaswani, N. Shazeer, N. Parmar, et al. Attention Is All You Need. CoRR, abs/1706.03762, 2017. [[pdf]](https://arxiv.org/pdf/1706.03762)
