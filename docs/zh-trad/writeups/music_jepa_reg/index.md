# Music JEPA Regularizers

> 再試Music LeJEPA之抗坍縮正則

---

LLMS index: [llms.txt](/llms.txt)

---

本文接續[Music LeJEPA](https://jipeng4974.github.io/writeups/music_lejepa/)的實踐，梳理VicReg、SigReg、VisReg三種正則項的數學原理、源碼實現，以及實踐中踩到的低秩局部最優陷阱，並基於“防止維度塌陷遠比復活死維度簡單”這一直覺，提出抗坍縮正則的λ調度策略。

## 三種正則

三者都是JEPA式訓練裏的抗坍縮正則項，共用同一個框架：

$$L = \lambda \cdot L_{reg} + (1-\lambda) \cdot L_{inv}$$

proj形狀`(V, B, D)`——V個view、batch size B、embedding維度D。inv項讓同一樣本不同view的投影互相靠近（L2 loss），reg項防止所有樣本坍縮成一個點。三種方法的區別只在reg項：把embedding分佈約束成什麼形狀、用什麼方式約束。

| 方法 | 約束目標 | 約束方式 |
|---|---|---|
| VICReg | 每維方差≥γ，維度間去相關 | 逐維統計量（二階矩） |
| SIGReg | embedding分佈=各向同性標準高斯N(0,I) | 隨機1D投影+特徵函數匹配（Epps-Pulley檢驗） |
| VISReg | 同爲N(0,I)，但分離center/scale/shape | 隨機1D投影+排序分位數匹配（sliced Wasserstein風格） |

## VICReg

VICReg[^1]把`(V, B, D)`的投影flatten成`N = V·B`個樣本，對每個維度施加兩個約束。

方差項，用hinge loss把每維標準差推到閾值γ以上：

$$v(Z) = \frac{1}{D}\sum_{d=1}^{D} \max(0,\ \gamma - \mathrm{Std}(Z_{:,d}))^2$$

協方差項，懲罰協方差矩陣非對角元，讓不同維度去相關：

$$c(Z) = \frac{1}{D}\sum_{i \neq j} \mathrm{Cov}(Z)_{i,j}^2$$

源碼（連同invariance項之外的完整正則）：

```python
import torch.nn as nn
import torch.nn.functional as F


class VICReg(nn.Module):
    def __init__(self, var_weight=25.0, cov_weight=1.0, gamma=1.0):
        super().__init__()
        self.var_weight = var_weight
        self.cov_weight = cov_weight
        self.gamma = gamma

    def forward(self, z):
        V, B, D = z.shape
        z_flat = z.reshape(V * B, D)
        std = z_flat.std(dim=0) + 1e-6
        var_loss = F.relu(self.gamma - std).pow(2).mean()          # hinge(γ - std)
        z_centered = z_flat - z_flat.mean(dim=0, keepdim=True)
        cov = (z_centered.T @ z_centered) / (V * B - 1)            # D×D 协方差矩阵
        off_diag = cov.pow(2).sum() - cov.diagonal().pow(2).sum()  # 非对角元平方和
        cov_loss = off_diag / D
        return self.var_weight * var_loss + self.cov_weight * cov_loss
```

優點是簡單、無隨機性、每個維度都能拿到確定性滿強度的梯度。缺點同樣直白——二階矩約束不緊，方差夠、協方差對角化的分佈不一定是高斯（球面均勻分佈也能矇混過關）；且協方差矩陣是$O(B \cdot D^2)$，高維時昂貴。

## SIGReg

SIGReg出自LeJEPA[^2]，數學上依賴兩個統計學結果。

一是Cramér–Wold定理：D維分佈由它所有1D投影唯一確定。隨機採K個單位方向$a_1..a_K$，讓每個投影$z \cdot a_k$都是N(0,1)，即逼近"整個分佈是各向同性高斯"。

二是Epps–Pulley檢驗[^3]：1983年的經典正態性擬合優度檢驗。依據是分佈由其特徵函數唯一確定（Bochner定理）。標準正態的特徵函數是$\varphi(t) = e^{-t^2/2}$（純實數），於是比較經驗特徵函數$\hat\varphi(t) = \frac{1}{N}\sum_j e^{itx_j}$與它的加權L2距離：

$$T = N \int_{-\infty}^{\infty} \left|\hat\varphi(t) - e^{-t^2/2}\right|^2 e^{-t^2/2}\, dt$$

把模方展開成cos/sin實虛部，在$t \in [0, t_{max}]$上取knots個點做梯形數值積分（利用對稱性，t<0部分權重翻倍），即得可微loss。關鍵在於這個統計量對樣本完全可微——假設檢驗搖身一變成爲正則項，這是LeJEPA的核心observation。

源碼：

```python
import torch
import torch.nn as nn


class SIGReg(nn.Module):
    def __init__(self, *, knots: int = 17, t_max: float = 3.0, num_projections: int = 256):
        super().__init__()
        self.num_projections = int(num_projections)
        t = torch.linspace(0, float(t_max), int(knots), dtype=torch.float32)
        dt = float(t_max) / (int(knots) - 1)
        weights = torch.full((int(knots),), 2 * dt, dtype=torch.float32)
        weights[[0, -1]] = dt                       # 梯形规则：端点半权重
        window = torch.exp(-t.square() / 2.0)       # 权重函数 = 目标特征函数
        self.register_buffer("t", t)
        self.register_buffer("phi", window)
        self.register_buffer("weights", weights * window)

    def forward(self, proj: torch.Tensor) -> torch.Tensor:
        _, B, D = proj.shape

        A = torch.randn(D, self.num_projections, device=proj.device, dtype=proj.dtype)
        A = A.div_(A.norm(p=2, dim=0, keepdim=True) + 1e-12)   # 随机单位投影方向

        t = self.t.to(device=proj.device, dtype=proj.dtype)
        phi = self.phi.to(device=proj.device, dtype=proj.dtype)
        weights = self.weights.to(device=proj.device, dtype=proj.dtype)

        x_t = (proj @ A).unsqueeze(-1) * t                        # (V,B,K,knots)
        err = (x_t.cos().mean(-3) - phi).square() + x_t.sin().mean(-3).square()
        statistic = (err @ weights) * B                           # 每个投影的 EP 统计量
        loss = statistic.mean()
        return loss
```

逐行對應數學：`proj @ A`是Cramér–Wold slicing，把embedding投到K=256個隨機方向；`x_t.cos().mean(-3)`和`x_t.sin().mean(-3)`是經驗特徵函數的實虛部（對batch取均值）；誤差項裏虛部直接平方，因爲標準正態特徵函數是純實數；`err @ weights`是以$e^{-t^2/2}$爲權重的梯形積分，×B是EP統計量的N倍縮放。

SIGReg約束的是完整分佈而非二階矩，理論上嚴格防坍縮，計算量$O(B \cdot D \cdot K)$，通常遠小於VICReg的$O(B \cdot D^2)$。缺點是隨機投影帶來Monte Carlo噪聲，且均值偏移、整體尺度、分佈形狀的懲罰混在一個統計量裏，梯度信噪比一般。

## VisReg

VisReg目標同爲N(0,I)，但把約束確定性地拆成center/scale/shape三部分，替代SIGReg"什麼都靠隨機投影+特徵函數"的做法。

Center，均值歸零：

$$L_{center} = \|\bar{z}\|^2$$

Scale，到均值的RMS半徑爲1：

$$L_{scale} = \left(\frac{\|z - \bar z\|_2}{\sqrt{B}} - 1\right)^2$$

Shape，中心化歸一化後在K個隨機方向上投影，把排序後的樣本值與標準正態的理論分位數對齊。標準正態的B個均勻分位點由逆誤差函數給出：

$$q_i = \sqrt{2}\,\mathrm{erf}^{-1}\!\left(\frac{2i}{B+1} - 1\right),\quad i = 1..B$$

$$L_{shape} = \frac{1}{K}\sum_{k}\frac{1}{B}\sum_{i}\left(\mathrm{sort}(z_{norm} \cdot a_k)_i - q_i\right)^2$$

這本質上是sliced 1D Wasserstein-2距離的平方——1D時W₂的最優傳輸就是排序對齊，方向隨機化對應isotropic Gaussian的sketching。

```python
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class VISReg(nn.Module):
    def __init__(self, num_projections: int = 256, scale_weight: float = 1.0,
                 shape_weight: float = 1.0, center_weight: float = 1.0):
        super().__init__()
        self.K = num_projections
        self._cached_B = -1
        self._cached_target = None
        self.scale_weight = scale_weight
        self.shape_weight = shape_weight
        self.center_weight = center_weight

    def _get_target(self, B: int, device) -> torch.Tensor:
        if self._cached_B != B:                      # 目标分位数只依赖 B，缓存复用
            q = torch.linspace(1, B, B, device=device, dtype=torch.float32) / (B + 1)
            self._cached_target = torch.erfinv(2 * q - 1).mul_(math.sqrt(2))
            self._cached_B = B
        return self._cached_target.to(device=device)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        _, B, D = z.shape

        mu = z.mean(dim=1, keepdim=True)
        center_loss = mu.pow(2).mean()                               # center

        z_centered = z - mu
        std = z_centered.norm(dim=1).div(math.sqrt(B)).clamp_min(1e-6)
        scale_loss = (std - 1.0).pow(2).mean()                       # scale

        z_norm = z_centered / std.detach().unsqueeze(1)              # detach: 形状与尺度解耦
        W = F.normalize(torch.randn(D, self.K, device=z.device, dtype=z.dtype), dim=0)
        p_sorted = (z_norm @ W).sort(dim=1).values                   # K 个随机投影, 每个排序
        target = self._get_target(B, z.device).view(1, B, 1)         # N(0,1) 理论分位数
        shape_loss = (p_sorted - target).pow(2).mean()               # 排序对齐 = 1D W2

        return self.scale_weight * scale_loss + self.shape_weight * shape_loss \
             + self.center_weight * center_loss
```

注意`std.detach()`這一行——尺度學習完全交給scale_loss，shape_loss只負責"形狀像不像正態"，兩者梯度互不干擾。這是VisReg相對SIGReg梯度更乾淨的原因，也是後文陷阱的伏筆。center/scale用確定性閉式約束（無Monte Carlo噪聲），只有shape用隨機投影；t網格、knots這類超參也都省掉了。

## 實踐觀察：erank先崩後升，然後停滯

我的設定：anchor music clip與degraded music clip做invariance對齊，同時施加常規λ正則項，用effective rank（embedding協方差譜的熵有效維度）監控坍縮。隨機初始化時音樂頻譜輸入模型得到的embeds的erank約20~30，訓練曲線呈現三階段：

- 早期迅速崩到~6——可以稱之爲維度塌陷；
- 緩慢回升到~100；
- 增速幾乎歸零，停滯。

SigReg如此；VisReg稍好[^4]；VICReg未試。

如果訓練早期λ不調得比較大（超過自然圖像訓練中的常見設定），inv loss迅速收斂，坍縮直接壓倒正則。

另一個反常實驗：VisReg + λ=0.99維持前3000步，erank早期不升反降，陰跌——而VisReg loss在同步下降。表面上embedding更接近isotropic Gaussian，dead dims卻變多了。實踐中我沒見誰真的用0.99的λ，試了一下出乎意料，但數學上解釋得通。

## 低秩局部最優陷阱

先說結論：VisReg的全局最優確實是N(0,I)，問題不在目標，在優化路徑的梯度結構。loss度量邊際分佈形狀，erank度量協方差譜，兩者在優化路徑上可以反向走。具體四個機制。

**(a) 恢復梯度是稀釋且隨機的。** shape項對高斯雲N(0,Σ)在隨機方向a上度量的領頭項等價於W₂²，即$(a^\top\Sigma a - 1)^2$。對隨機單位方向取期望：

$$\mathbb{E}_a\big[(a^\top\Sigma a - 1)^2\big] \propto \frac{\|\Sigma - I\|_F^2}{D}$$

Σ=I確實是全局最優，但單個瀕死特徵值對期望loss的貢獻只有O(1/D)，而每步只用K=256個方向Monte Carlo估計，單步噪聲~1/√K。相比之下invariance的坍縮壓力是確定性的、每步滿強度的——凡是對degradation敏感的維度都是inv loss的來源，梯度直接掐掉它。確定性坍縮對上隨機稀釋恢復，早期誰贏一目瞭然。

**(b) 自抑制：恢復力∝缺陷大小。** 特徵值λ_j越小，它在sketch裏留下的殘差越小，恢復梯度越弱。且深度坍縮時B個樣本在投影上幾乎重合，排序匹配的相對順序近乎隨機，梯度被排列噪聲主導。erank回升到~100後停滯不是"失去動力"，是恢復力衰減到與inv坍縮壓力平衡的不動點。

**(c) 因子丟失逃逸路徑。** `std.detach()`的白化讓shape項對逐維方差免疫，檢測死維只剩scale_loss的$(std_d - 1)^2$——而它只問"方差是不是1"，不問"方差從哪來"。不妨設想embedding各維度是k個潛在因子的線性混合：網絡丟掉一個因子後，把剩餘k−1個因子重新混合，讓每個座標維仍保持std=1、均值0、邊際鐘形。此時center、scale、shape三項全綠，VisReg loss繼續下降，erank掉一格。λ=0.99實驗中"loss降、dead dims變多"正是這個機制。

**(d) λ=0.99主動走進陷阱。** rank的有用擴張只能由invariance任務驅動——reg只要雲像高斯，不在乎樣本在雲裏的排布（排序匹配在投影內置換不變）。0.99把inv錨定砍到0.01：樣本相對幾何失去維護，判別性因子被weight decay和噪聲梯度慢慢侵蝕；reg主導時，從當前雲出發最近的類高斯配置是"壓縮、重排現有結構"而非"擴張新方向"——擴張需要編碼新的輸入信息，沒有inv信號組織它。正則太純，就是在優化一個與信息量正交的目標，低秩局域解成了最速下降方向。

## λ調度：防優先於救

坍縮的下坡是確定性、滿強度、逐維直接的；復活的上坡是O(1/D)稀釋、1/√K帶噪、且恢復力∝缺陷大小的——維度死得越深越難救，λ_j→0時梯度信噪比趨於零。不對稱性極其懸殊。因此λ調度的首要目標不是"坍縮後把erank拉回來"，而是"根本不讓坍縮發生"。一旦erank崩到個位數，後續訓練大部分時間在還早期的債，且只能還到不動點爲止。

具體策略：

- 不用極端λ。0.99切斷inv錨定，主動走進陷阱；太小則inv秒收斂直接坍縮。取中間偏大（0.5~0.8區間，按erank監控微調）。
- λ從第0步就設在目標值附近（或只經數百步短warmup到位）。等看到erank下跌再調大λ已經晚了——死維恢復又慢又不完全，讓inv從不在無監管狀態下運行。
- 以erank爲調度信號，而不是以loss爲信號。VisReg loss下降不能說明分佈健康（因子丟失路徑），erank纔是譜的直接度量。設一個下限報警線（如初始值的70%），跌破即升λ或降lr。
- erank算在哪個空間要確認。VisReg作用在proj上，若監控的是emb，還有projector hiding的問題——projector可以把一個正在坍縮的emb白化成高斯proj。emb和proj的erank都監控。
- 逐維std直方圖輔助診斷：std全≈1而erank掉，確認是因子丟失（相關性坍縮）；部分維std→0，是scale項被inv壓制，說明λ不夠。

## 監控指標：erank之外還需要什麼

erank只看譜的熵，VisReg loss只看K個隨機sketch上的邊際——各自有盲區。分佈趨近isotropic Gaussian有三個正交的失效模式：各向異性（Σ≠I，二階）、非高斯性（形狀，高階）、維度間依賴（聯合結構），單一標量覆蓋不了，用分層套件。

首先應該增加per-dim std以便診斷死維。std=0是完全死維——該維度在整個batch上是常數，不攜帶任何區分樣本的信息。std=1只是達到scale項的目標值——只說明"非死"，不說明"有用"：方差來源可以是真實信息，可以是噪聲，也可以是其他因子的重混合。

第二步、第三步是引入相關矩陣和kurtosis，用於判斷方差是否"獨立且高斯"。

最終如果爲了追求完美，還可以算Epps-Pulley檢驗的多元閉式版本，於是整體就形成了不同時間尺度上的分層監控：

- 每step 輕量metrics（O(D³)特徵分解）：
    - 逐維std直方圖+erank。
    - log det Σ[^5]。
    - −log det R[^6]。
- 每3k steps 中量metrics（O(B·D²)）：
    - 逐維excess kurtosis與skew的均值/最大值——譜全平但每維尖峯厚尾時，譜指標完全無感，kurtosis立刻暴露。逐維邊際都正態不等於聯合正態（copula反例），所以這一層只查必要條件。
    - Mahalanobis χ²檢驗：若z~N(μ,Σ)，馬氏距離平方d²=(z−μ)ᵀΣ⁻¹(z−μ)服從χ²_D。對batch算d²的經驗分佈，與χ²_D分位數做QQ圖，或用KS距離壓成單標量——同時檢驗徑向輪廓（高斯的殼層結構）和橢球性，對"譜接近I但形狀是球面均勻分佈"這類二階矩漏網之魚很敏感。
- 每10k steps 重量metrics（O(B²·D)）：
    - BHEP統計量——Epps-Pulley檢驗的多元閉式版本。權重取高斯核後積分有解析解，完全不需要數值積分和隨機方向：

$$T_\beta = \frac{1}{B^2}\sum_{j,k} e^{-\frac{\beta^2}{2}\|Y_j - Y_k\|^2} - \frac{2}{B}(1+\beta^2)^{-D/2}\sum_j e^{-\frac{\beta^2\|Y_j\|^2}{2(1+\beta^2)}} + (1+2\beta^2)^{-D/2}$$
    其中Y_j=Σ^{-1/2}(z_j−z̄)是studentized樣本，β≈1即可。特徵函數唯一確定分佈，所以它對任何偏離多元正態的方向都非零——不會被因子重混合騙過，也沒有sketch噪聲。



## 針對低秩盲區和死維復活的改進

1. 可以考慮訓練早期用VICReg的確定性約束迅速把erank撐起來，中後期切換成VisReg，讓embedding分佈收斂到isotropic高斯。
2. 將VIC的協方差項加入VIS中。
3. 對VIS的scale項進行改進，避免過度稀釋，通過更強力的scale項保證下限，硬抬eranks。
4. 考慮curriculum learning，一開始先不要讓invariance的強度太高，把degradation合成做難度分級，先從最簡單的開始（比如輕微的變速變調，輕微加噪），再逐步增加難度。過於困難的對齊，也容易逼迫模型找到正則項漏洞進行hacking——當然這種hacking也不是全然無用，至少有指明更好的正則項設計的作用。

```
# 现状：1/D 平均，死维的求救信号被 512 分母稀释
l_scale = (1.0 - std).square().mean()

# 改法：恒定推力聚焦最差的 k 维，让死维获得 O(1) 级复活力
k = max(1, std.size(-1) // 8)
worst = std.topk(k, largest=False).values          # bottom-k std
l_revive = (1.0 - worst).square().mean()
l_scale = 0.5 * (1.0 - std).square().mean() + 0.5 * l_revive
```

[^1]: Bardes, Ponce, LeCun. VICReg: Variance-Invariance-Covariance Regularization for Self-Supervised Learning. ICLR 2022. [[arxiv]](https://arxiv.org/abs/2105.04906)
[^2]: Balestriero, LeCun. LeJEPA: Provable and Scalable Self-Supervised Learning Without the Heuristics. 2025. [[arxiv]](https://arxiv.org/abs/2511.08544)
[^3]: Epps, Pulley. A test for normality based on the empirical characteristic function. Biometrika 70(3), 1983.
[^4]: 不嚴謹的猜測："稍好"可能只是因爲排序匹配對低方差方向的恢復力強於被$e^{-t^2/2}$窗口鈍化的特徵函數匹配——兩者梯度幾何不同，需要更長的訓練曲線驗證。
[^5]: log det Σ = Σᵢ log λᵢ，λᵢ爲協方差特徵值。幾何上是高維橢球體積的對數——每個維度貢獻一個log λ，死一個維度體積歸零、log det→−∞，因此對瀕死維度比erank敏感得多。scale項把trace鎖定爲D後，它在Σ=I取最大值；它還正比於高斯微分熵h = ½·log det(2πe·Σ)。
[^6]: R是相關矩陣（Σ歸一化到對角線全1）。−log det R度量維度間的整體相關性：R=I時爲0，相關性越強行列式越小。對高斯分佈，−½·log det R恰好等於total correlation，即各維互信息之和——因子丟失/重混合時它確定性升高，不經過任何sketch。
