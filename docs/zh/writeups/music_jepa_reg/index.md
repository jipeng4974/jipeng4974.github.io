# Music JEPA Regularizers

> 再试Music LeJEPA之抗坍缩正则

---

LLMS index: [llms.txt](/llms.txt)

---

本文接续[Music LeJEPA](https://jipeng4974.github.io/writeups/music_lejepa/)的实践，梳理VicReg、SigReg、VisReg三种正则项的数学原理、源码实现，以及实践中踩到的低秩局部最优陷阱，并基于“防止维度塌陷远比复活死维度简单”这一直觉，提出抗坍缩正则的λ调度策略。

## 三种正则

三者都是JEPA式训练里的抗坍缩正则项，共用同一个框架：

$$L = \lambda \cdot L_{reg} + (1-\lambda) \cdot L_{inv}$$

proj形状`(V, B, D)`——V个view、batch size B、embedding维度D。inv项让同一样本不同view的投影互相靠近（L2 loss），reg项防止所有样本坍缩成一个点。三种方法的区别只在reg项：把embedding分布约束成什么形状、用什么方式约束。

| 方法 | 约束目标 | 约束方式 |
|---|---|---|
| VICReg | 每维方差≥γ，维度间去相关 | 逐维统计量（二阶矩） |
| SIGReg | embedding分布=各向同性标准高斯N(0,I) | 随机1D投影+特征函数匹配（Epps-Pulley检验） |
| VISReg | 同为N(0,I)，但分离center/scale/shape | 随机1D投影+排序分位数匹配（sliced Wasserstein风格） |

## VICReg

VICReg[^1]把`(V, B, D)`的投影flatten成`N = V·B`个样本，对每个维度施加两个约束。

方差项，用hinge loss把每维标准差推到阈值γ以上：

$$v(Z) = \frac{1}{D}\sum_{d=1}^{D} \max(0,\ \gamma - \mathrm{Std}(Z_{:,d}))^2$$

协方差项，惩罚协方差矩阵非对角元，让不同维度去相关：

$$c(Z) = \frac{1}{D}\sum_{i \neq j} \mathrm{Cov}(Z)_{i,j}^2$$

源码（连同invariance项之外的完整正则）：

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

优点是简单、无随机性、每个维度都能拿到确定性满强度的梯度。缺点同样直白——二阶矩约束不紧，方差够、协方差对角化的分布不一定是高斯（球面均匀分布也能蒙混过关）；且协方差矩阵是$O(B \cdot D^2)$，高维时昂贵。

## SIGReg

SIGReg出自LeJEPA[^2]，数学上依赖两个统计学结果。

一是Cramér–Wold定理：D维分布由它所有1D投影唯一确定。随机采K个单位方向$a_1..a_K$，让每个投影$z \cdot a_k$都是N(0,1)，即逼近"整个分布是各向同性高斯"。

二是Epps–Pulley检验[^3]：1983年的经典正态性拟合优度检验。依据是分布由其特征函数唯一确定（Bochner定理）。标准正态的特征函数是$\varphi(t) = e^{-t^2/2}$（纯实数），于是比较经验特征函数$\hat\varphi(t) = \frac{1}{N}\sum_j e^{itx_j}$与它的加权L2距离：

$$T = N \int_{-\infty}^{\infty} \left|\hat\varphi(t) - e^{-t^2/2}\right|^2 e^{-t^2/2}\, dt$$

把模方展开成cos/sin实虚部，在$t \in [0, t_{max}]$上取knots个点做梯形数值积分（利用对称性，t<0部分权重翻倍），即得可微loss。关键在于这个统计量对样本完全可微——假设检验摇身一变成为正则项，这是LeJEPA的核心observation。

源码：

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

逐行对应数学：`proj @ A`是Cramér–Wold slicing，把embedding投到K=256个随机方向；`x_t.cos().mean(-3)`和`x_t.sin().mean(-3)`是经验特征函数的实虚部（对batch取均值）；误差项里虚部直接平方，因为标准正态特征函数是纯实数；`err @ weights`是以$e^{-t^2/2}$为权重的梯形积分，×B是EP统计量的N倍缩放。

SIGReg约束的是完整分布而非二阶矩，理论上严格防坍缩，计算量$O(B \cdot D \cdot K)$，通常远小于VICReg的$O(B \cdot D^2)$。缺点是随机投影带来Monte Carlo噪声，且均值偏移、整体尺度、分布形状的惩罚混在一个统计量里，梯度信噪比一般。

## VisReg

VisReg目标同为N(0,I)，但把约束确定性地拆成center/scale/shape三部分，替代SIGReg"什么都靠随机投影+特征函数"的做法。

Center，均值归零：

$$L_{center} = \|\bar{z}\|^2$$

Scale，到均值的RMS半径为1：

$$L_{scale} = \left(\frac{\|z - \bar z\|_2}{\sqrt{B}} - 1\right)^2$$

Shape，中心化归一化后在K个随机方向上投影，把排序后的样本值与标准正态的理论分位数对齐。标准正态的B个均匀分位点由逆误差函数给出：

$$q_i = \sqrt{2}\,\mathrm{erf}^{-1}\!\left(\frac{2i}{B+1} - 1\right),\quad i = 1..B$$

$$L_{shape} = \frac{1}{K}\sum_{k}\frac{1}{B}\sum_{i}\left(\mathrm{sort}(z_{norm} \cdot a_k)_i - q_i\right)^2$$

这本质上是sliced 1D Wasserstein-2距离的平方——1D时W₂的最优传输就是排序对齐，方向随机化对应isotropic Gaussian的sketching。

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

注意`std.detach()`这一行——尺度学习完全交给scale_loss，shape_loss只负责"形状像不像正态"，两者梯度互不干扰。这是VisReg相对SIGReg梯度更干净的原因，也是后文陷阱的伏笔。center/scale用确定性闭式约束（无Monte Carlo噪声），只有shape用随机投影；t网格、knots这类超参也都省掉了。

## 实践观察：erank先崩后升，然后停滞

我的设定：anchor music clip与degraded music clip做invariance对齐，同时施加常规λ正则项，用effective rank（embedding协方差谱的熵有效维度）监控坍缩。随机初始化时音乐频谱输入模型得到的embeds的erank约20~30，训练曲线呈现三阶段：

- 早期迅速崩到~6——可以称之为维度塌陷；
- 缓慢回升到~100；
- 增速几乎归零，停滞。

SigReg如此；VisReg稍好[^4]；VICReg未试。

如果训练早期λ不调得比较大（超过自然图像训练中的常见设定），inv loss迅速收敛，坍缩直接压倒正则。

另一个反常实验：VisReg + λ=0.99维持前3000步，erank早期不升反降，阴跌——而VisReg loss在同步下降。表面上embedding更接近isotropic Gaussian，dead dims却变多了。实践中我没见谁真的用0.99的λ，试了一下出乎意料，但数学上解释得通。

## 低秩局部最优陷阱

先说结论：VisReg的全局最优确实是N(0,I)，问题不在目标，在优化路径的梯度结构。loss度量边际分布形状，erank度量协方差谱，两者在优化路径上可以反向走。具体四个机制。

**(a) 恢复梯度是稀释且随机的。** shape项对高斯云N(0,Σ)在随机方向a上度量的领头项等价于W₂²，即$(a^\top\Sigma a - 1)^2$。对随机单位方向取期望：

$$\mathbb{E}_a\big[(a^\top\Sigma a - 1)^2\big] \propto \frac{\|\Sigma - I\|_F^2}{D}$$

Σ=I确实是全局最优，但单个濒死特征值对期望loss的贡献只有O(1/D)，而每步只用K=256个方向Monte Carlo估计，单步噪声~1/√K。相比之下invariance的坍缩压力是确定性的、每步满强度的——凡是对degradation敏感的维度都是inv loss的来源，梯度直接掐掉它。确定性坍缩对上随机稀释恢复，早期谁赢一目了然。

**(b) 自抑制：恢复力∝缺陷大小。** 特征值λ_j越小，它在sketch里留下的残差越小，恢复梯度越弱。且深度坍缩时B个样本在投影上几乎重合，排序匹配的相对顺序近乎随机，梯度被排列噪声主导。erank回升到~100后停滞不是"失去动力"，是恢复力衰减到与inv坍缩压力平衡的不动点。

**(c) 因子丢失逃逸路径。** `std.detach()`的白化让shape项对逐维方差免疫，检测死维只剩scale_loss的$(std_d - 1)^2$——而它只问"方差是不是1"，不问"方差从哪来"。不妨设想embedding各维度是k个潜在因子的线性混合：网络丢掉一个因子后，把剩余k−1个因子重新混合，让每个坐标维仍保持std=1、均值0、边际钟形。此时center、scale、shape三项全绿，VisReg loss继续下降，erank掉一格。λ=0.99实验中"loss降、dead dims变多"正是这个机制。

**(d) λ=0.99主动走进陷阱。** rank的有用扩张只能由invariance任务驱动——reg只要云像高斯，不在乎样本在云里的排布（排序匹配在投影内置换不变）。0.99把inv锚定砍到0.01：样本相对几何失去维护，判别性因子被weight decay和噪声梯度慢慢侵蚀；reg主导时，从当前云出发最近的类高斯配置是"压缩、重排现有结构"而非"扩张新方向"——扩张需要编码新的输入信息，没有inv信号组织它。正则太纯，就是在优化一个与信息量正交的目标，低秩局域解成了最速下降方向。

## λ调度：防优先于救

坍缩的下坡是确定性、满强度、逐维直接的；复活的上坡是O(1/D)稀释、1/√K带噪、且恢复力∝缺陷大小的——维度死得越深越难救，λ_j→0时梯度信噪比趋于零。不对称性极其悬殊。因此λ调度的首要目标不是"坍缩后把erank拉回来"，而是"根本不让坍缩发生"。一旦erank崩到个位数，后续训练大部分时间在还早期的债，且只能还到不动点为止。

具体策略：

- 不用极端λ。0.99切断inv锚定，主动走进陷阱；太小则inv秒收敛直接坍缩。取中间偏大（0.5~0.8区间，按erank监控微调）。
- λ从第0步就设在目标值附近（或只经数百步短warmup到位）。等看到erank下跌再调大λ已经晚了——死维恢复又慢又不完全，让inv从不在无监管状态下运行。
- 以erank为调度信号，而不是以loss为信号。VisReg loss下降不能说明分布健康（因子丢失路径），erank才是谱的直接度量。设一个下限报警线（如初始值的70%），跌破即升λ或降lr。
- erank算在哪个空间要确认。VisReg作用在proj上，若监控的是emb，还有projector hiding的问题——projector可以把一个正在坍缩的emb白化成高斯proj。emb和proj的erank都监控。
- 逐维std直方图辅助诊断：std全≈1而erank掉，确认是因子丢失（相关性坍缩）；部分维std→0，是scale项被inv压制，说明λ不够。

## 监控指标：erank之外还需要什么

erank只看谱的熵，VisReg loss只看K个随机sketch上的边际——各自有盲区。分布趋近isotropic Gaussian有三个正交的失效模式：各向异性（Σ≠I，二阶）、非高斯性（形状，高阶）、维度间依赖（联合结构），单一标量覆盖不了，用分层套件。

首先应该增加per-dim std以便诊断死维。std=0是完全死维——该维度在整个batch上是常数，不携带任何区分样本的信息。std=1只是达到scale项的目标值——只说明"非死"，不说明"有用"：方差来源可以是真实信息，可以是噪声，也可以是其他因子的重混合。

第二步、第三步是引入相关矩阵和kurtosis，用于判断方差是否"独立且高斯"。

最终如果为了追求完美，还可以算Epps-Pulley检验的多元闭式版本，于是整体就形成了不同时间尺度上的分层监控：

- 每step 轻量metrics（O(D³)特征分解）：
    - 逐维std直方图+erank。
    - log det Σ[^5]。
    - −log det R[^6]。
- 每3k steps 中量metrics（O(B·D²)）：
    - 逐维excess kurtosis与skew的均值/最大值——谱全平但每维尖峰厚尾时，谱指标完全无感，kurtosis立刻暴露。逐维边际都正态不等于联合正态（copula反例），所以这一层只查必要条件。
    - Mahalanobis χ²检验：若z~N(μ,Σ)，马氏距离平方d²=(z−μ)ᵀΣ⁻¹(z−μ)服从χ²_D。对batch算d²的经验分布，与χ²_D分位数做QQ图，或用KS距离压成单标量——同时检验径向轮廓（高斯的壳层结构）和椭球性，对"谱接近I但形状是球面均匀分布"这类二阶矩漏网之鱼很敏感。
- 每10k steps 重量metrics（O(B²·D)）：
    - BHEP统计量——Epps-Pulley检验的多元闭式版本。权重取高斯核后积分有解析解，完全不需要数值积分和随机方向：

$$T_\beta = \frac{1}{B^2}\sum_{j,k} e^{-\frac{\beta^2}{2}\|Y_j - Y_k\|^2} - \frac{2}{B}(1+\beta^2)^{-D/2}\sum_j e^{-\frac{\beta^2\|Y_j\|^2}{2(1+\beta^2)}} + (1+2\beta^2)^{-D/2}$$
    其中Y_j=Σ^{-1/2}(z_j−z̄)是studentized样本，β≈1即可。特征函数唯一确定分布，所以它对任何偏离多元正态的方向都非零——不会被因子重混合骗过，也没有sketch噪声。



## 针对低秩盲区和死维复活的改进

1. 可以考虑训练早期用VICReg的确定性约束迅速把erank撑起来，中后期切换成VisReg，让embedding分布收敛到isotropic高斯。
2. 将VIC的协方差项加入VIS中。
3. 对VIS的scale项进行改进，避免过度稀释，通过更强力的scale项保证下限，硬抬eranks。
4. 考虑curriculum learning，一开始先不要让invariance的强度太高，把degradation合成做难度分级，先从最简单的开始（比如轻微的变速变调，轻微加噪），再逐步增加难度。过于困难的对齐，也容易逼迫模型找到正则项漏洞进行hacking——当然这种hacking也不是全然无用，至少有指明更好的正则项设计的作用。

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
[^4]: 不严谨的猜测："稍好"可能只是因为排序匹配对低方差方向的恢复力强于被$e^{-t^2/2}$窗口钝化的特征函数匹配——两者梯度几何不同，需要更长的训练曲线验证。
[^5]: log det Σ = Σᵢ log λᵢ，λᵢ为协方差特征值。几何上是高维椭球体积的对数——每个维度贡献一个log λ，死一个维度体积归零、log det→−∞，因此对濒死维度比erank敏感得多。scale项把trace锁定为D后，它在Σ=I取最大值；它还正比于高斯微分熵h = ½·log det(2πe·Σ)。
[^6]: R是相关矩阵（Σ归一化到对角线全1）。−log det R度量维度间的整体相关性：R=I时为0，相关性越强行列式越小。对高斯分布，−½·log det R恰好等于total correlation，即各维互信息之和——因子丢失/重混合时它确定性升高，不经过任何sketch。
