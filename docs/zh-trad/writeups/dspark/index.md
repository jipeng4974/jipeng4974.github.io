# DSpark

> DSpark = 半自迴歸 draft（重並行骨幹 + 輕量順序頭） + 置信度感知的動態驗證調度

---

LLMS index: [llms.txt](/llms.txt)

---

投機解碼的每個token平均延遲是$L = (T_{draft} + T_{verify}) / \tau$，$\tau$是每輪接受的token數。DSpark的工作[^1]（代碼在DeepSpec[^2]）圍繞公式中的三個變量展開：它修正了並行drafter的依賴缺失（提$\tau$而不漲$T_{draft}$），又把驗證長度從靜態超參變成了一個每步求解的資源分配問題（降有效$T_{verify}$）。後者本質上是把serving系統的負載信息反饋進了算法層。

## 兩難

自迴歸drafter（EAGLE系[^3]）逐token串行起draft，$T_{draft} \propto \gamma$，像一條很短但級數不可省的串行流水線。只能用小$\gamma$，爲補償接受率又引入tree attention驗證，大量候選token白白佔用target的batch容量。

並行drafter（Medusa[^4]/DFlash[^5]系）單次前向輸出全部$\gamma$個位置，$T_{draft}$與塊長無關。但各位置獨立預測、塊內無依賴建模，產生multi-modal collision：context同時允許"of course"和"no problem"時，並行預測可能拼出"of problem"。表現爲接受率沿塊快速衰減（suffix decay），$\gamma$越大浪費越多。

還有一個被忽視的系統維度：固定長度驗證在真實serving下不是最優。最優驗證長度沿兩個軸變化——

- 數據軸：code的接受率天然高於開放式chat；
- 系統軸：輕載時多驗證幾個token近乎免費；重載時驗證註定被拒的token會擠佔其他請求的batch容量。生產上MTP只用1個token，原因正在於此。

## 半自迴歸

DSpark的draft分兩階段。Parallel stage是DFlash的骨架：5層塊並行backbone，輸入`[anchor_token, mask×(γ-1)]`，單次前向、塊內雙向注意力（`is_causal=False`），產出全部$\gamma$個位置的hidden和base logits。關鍵機制是KV injection：prefill時取target多箇中間層hidden states拼接投影，注入每層draft注意力的K/V——context位置的K/V由target特徵算出而非draft自身歷史，相當於把target的中間表徵線性讀出後當作draft的外置記憶。draft與target共享embedding和LM head，均凍結。

Sequential stage是一個低秩Markov head，在base logits上疊加依賴已採樣前綴的轉移bias：

$$p_k(v \mid x_0, x_{<k}) = \mathrm{softmax}\bigl(U_k(v) + B(x_{k-1}, v)\bigr), \qquad B = W_1 W_2,\; r=256$$

推理時從左到右走$\gamma$步廉價循環：每步用上一個採出的token查`W_1[x_{k-1}]`算bias、修正logits、採樣。逐token概率仍是精確softmax——這是rejection sampling無損性的前提，也是它和CRF-NAT（全局配分函數給不出精確per-token概率）、CTC（只能greedy）的分界線。

效果：重型依賴建模只並行做一次，串行依賴被壓成一張$V \times V$低秩bigram表。$T_{draft}$幾乎不變（實測比DFlash僅+0.2%-1.3%整輪延遲），suffix decay顯著緩解。論文還給了一個GRU式RNN head，僅長proposal有邊際收益，部署默認Markov——這個取捨很工程。

## 置信度即准入

Confidence head就是單個線性層，每位置輸出標量$c_k$，建模條件存活概率（前$k-1$個全被接受的條件下第$k$個被接受的概率）。監督信號是解析的：逐步接受率恰好等於$1 - \tfrac{1}{2}\|p_d - p_t\|_1$，直接BCE訓練，無需額外標註。神經置信度系統性過自信，post-hoc用逐位置溫度縮放（STS）校準累積乘積的ECE，從3%-8%降到1%；溫度縮放保序，不破壞token排序。

調度側，問題被形式化爲全局吞吐最大化。$R$個活躍請求，前綴存活概率$a_{r,j} = \prod_{i \le j} c_{r,i}$，驗證batch token數$B = \sum_r (1+\ell_r)$，期望接受數$\tau = \sum_r (1 + \sum_j a_{r,j})$。引擎初始化時profile一次實測SPS(B)（steps/s vs batch size，一張輕量cost table），目標是

$$\max\; \Theta = \tau \cdot \mathrm{SPS}(B)$$

——多驗證一個token的期望收益，和batch變大後SPS下降的邊際成本，放進同一個目標裏權衡。算法：全部候選token按置信度全局降序貪心准入，$\Theta$不再提升時early-stop。

對我這種做系統出身的人來說，這就是admission control：帶收益模型的准入決策，cost function是實測的SPS曲線，每個decode step做一次輕量的全局資源分配。DFlash/EAGLE之爭則是寬而淺的單發射對窄而深的多發射，DSpark用低秩表把串行依賴的代價壓到了接近零。

## 因果屏障

$c_{k+1}$依賴已採樣token $x_k$的具體取值，所以"是否驗證第$k+1$個token"的決策若泄漏了$x_k$的取值就會引入selection bias。附錄A給了具體反例：構造下回顧式調度使輸出分佈從$(0.7, 0.3)$變成$(0.85, 0.15)$，不再lossless。Early-stopping使截斷決策只依賴決策點之前的前綴信息（non-anticipating），恢復嚴格無損。

生產部署（DeepSeek-V4）還有一層異步改造：理論算法假設SPS平滑單峯，真實SPS(B)是離散階梯，且Zero-Overhead Scheduling要求當前step結束前已知下一step的batch size。做法是用兩步之前的置信度估計驗證容量$K$，轉化爲dynamic top-K選擇，去掉break做全局搜索以跨越SPS懸崖。決策只基於兩步前的歷史，天然形成因果屏障，無損性得以保留。變長驗證kernel則把batch內所有token flatten成獨立元素統一處理，序列內依賴通過sparse attention裏的marker tensor傳達，只需改index-attention和compress兩個kernel。

## target不進GPU

訓練側的一個漂亮取捨：target全程凍結且不在GPU上。先離線跑target抓多層hidden states寫target cache（自定義二進制格式，mmap隨機讀，4B模型默認配置約38TB磁盤），訓練只讀cache；target分佈由target最後一層hidden過共享lm_head重建。

每條序列隨機採512個anchor位置，各構造7-token塊，pack成稠密batch；注意力mask用flex_attention的`create_block_mask`表達：塊內雙向、跨塊隔離、可attend各自anchor之前的context。三項loss按位置指數衰減加權$w_k = \exp(-(k-1)/\gamma)$：

- CE（0.1）：teacher-forcing逐位置交叉熵；
- L1蒸餾（0.9）：$\|p_d - p_t\|_1$，直接優化接受率；
- confidence BCE（1.0）：軟標籤是detach的$1 - \tfrac{1}{2}\|p_d - p_t\|_1$。

一個實現上的觀察：DFlash在DeepSpec代碼裏不是獨立模型，而是DSpark的配置特例（`markov_rank=0`、`confidence_head_alpha=0`、純CE loss）。

## 鏈式拒絕採樣

標準speculative sampling，沒有tree：target對draft+1個token一次前向，`accept_prob = clamp(p_t/p_d, 1)`做cumprod得接受前綴，拒絕處從殘差分佈`norm(max(p_t - p_d, 0))`採樣修正token，全接受則採bonus token。KV cache處理：target側crop回滾被拒token；draft側每輪block前向後立即crop掉noise token的K/V，只保留已接受context部分（K/V由target hidden算出，增量追加）。

## 性能提升與侷限

離線（Qwen3-4B/8B/14B、Gemma4-12B，9個benchmark，平均接受長度$\tau$）：相對Eagle3 +27%-31%，相對DFlash +16%-18%；2層DSpark即超5層DFlash；$\gamma$越大優勢越大，$\gamma=15$時領先DFlash 22%-30%。逐位置看，並行骨架給了很高的首位置接受率（chat 0.72 vs Eagle3 0.53——prefix-matching中首token槓桿最大），Markov head壓制了後綴衰減。

在線（DeepSeek-V4-Flash/Pro preview，真實流量，對比生產基線MTP-1）：80 tok/s/user SLA下聚合吞吐+51%；吞吐匹配時單用戶速度+60%-85%。負載自適應符合預期：中低併發時驗證預算從靜態2 token擴到4-6 token，併發飽和時平滑收縮。論文裏+661%，better take it with a grain of salt[^6]。

侷限同樣清楚。Prefix scheduler只減少驗證浪費，並行backbone生成整塊的計算是沉沒成本，對接受率極低的查詢無法回收——論文自己提了difficulty-aware early exit作爲未來方向。Early-stopping貪心的全局最優性依賴$\Theta$單峯，真實SPS鋸齒狀，靠異步兩步前估計繞過。SPS(B)假設忽略context長度對decode延遲的影響，這個假設依賴"平均context遠小於1M + PD分離負載均衡"的前提。

[^1]: DSpark: Confidence-Scheduled Speculative Decoding with Semi-Autoregressive Generation. [[arxiv]](https://arxiv.org/abs/2607.05147)
[^2]: DeepSeek. DeepSpec: draft模型訓練/評測庫，含Eagle3、DFlash、DSpark. [[github]](https://github.com/deepseek-ai/DeepSpec)
[^3]: EAGLE-3. [[arxiv]](https://arxiv.org/abs/2503.01840)
[^4]: Medusa: Simple LLM Inference Acceleration Framework with Multiple Decoding Heads. [[arxiv]](https://arxiv.org/abs/2401.10774)
[^5]: DFlash. [[arxiv]](https://arxiv.org/abs/2602.06036)
[^6]: 該數字出現在120 tok/s/user的嚴格SLA點上，此時MTP-1基線已進入低併發退化區、只能維持很小併發。論文自己也註明這應解讀爲"擴展了可行交互前沿"而非代表性加速比。吞吐匹配時單用戶速度+60%-85%纔是更誠實的數字。
