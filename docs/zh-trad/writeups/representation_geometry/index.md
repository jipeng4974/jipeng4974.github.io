# 表徵幾何

> 表徵學習的反覆迭代中，應嘗試估計嵌入子流形的內稟維數，在輸入和輸出環節上設計合理的 ambient space。

---

LLMS index: [llms.txt](/llms.txt)

---

考慮以 8s clip的logmel 爲輸入的音頻表徵模型，其 embedding pipeline 是如下降維過程：

$$
\begin{gathered}
w\in\mathbb{R}^{176400}
\;\xrightarrow{\;\Phi\;(\text{logmel})\;}\;
x\in\mathbb{R}^{11008}
\;\xrightarrow{\;h_\theta\;(\text{ViT-L})\;}\;
\mathrm{CLS}\in\mathbb{R}^{1024}
\\[6pt]
\;\xrightarrow{\;g_\theta\;(\text{proj head})\;}\;
z\in\mathbb{R}^{256}
\;\xrightarrow{\;\ell_2\;}\;
\hat z\in S^{255}
\end{gathered}
$$

1. 從點視角看，信號空間的某個單點 $w$，被重參數化爲語義度量空間的點。

2. 從 ambient 視角看，維數從信號空間的 176400 維，收縮到像素空間的 11008 維，經過 ViT-L，映射到 1024 維，再經過 proj head，映射到 256 維，最後經過 $\ell_2$ 歸一化，映射到 255 維的球面。其中可學部分是 $f_\theta=\ell_2\circ g_\theta\circ h_\theta$，即 $\mathbb{R}^{11008}\to S^{255}$ 這一段；$\Phi$ 是前置的固定壓縮。
- 所謂 ambient 即流形的 support，最原始的 ambient 就是 waveform 信號空間，即 $\mathbb{R}^{176400}$。經過 logmel 變換，映射到 11008 維的 logmel像素空間。最終的$S^{255}$通常來說也是輸出子流形的 ambient，因爲實際選擇的輸出維數往往比內稟維數略高。
- Ambient space——如waveform 信號空間、logmel 像素空間、最終向量的球面空間——不僅是直接可見的，還是人爲規定的。需銘記，它們的維度是非自然的，非內稟的。
- 是否有可能讓$S^{D-1}$這個 ambient剛好和其嵌入子流形近似重合？困難，但值得嘗試，一方面要根據對嵌入子流形內稟維度的估算調整超參D，另一方面引入 SIGReg/VISReg正則項。


> **定義（流形假設）。** 設 $\mu_{\text{in}}=\Psi_\#(\mathrm{law}(c,\alpha))$。其**支撐集**
> $$\mathcal{M}_{\text{in}}\;=\;\operatorname{supp}(\mu_{\text{in}})\;\subset\;\mathbb{R}^{P}$$
> 在流形假設下是一個（近似）光滑的 $d_{\text{in}}$-維嵌入子流形，$d_{\text{in}}\ll P$。
> 局部地，任一點 $x$ 附近存在微分同胚 $\varphi:\ U\subset\mathbb{R}^{d_{\text{in}}}\to\mathcal{M}_{\text{in}}$。

3. 從流形視角看，

$$
\mathcal{M}_{\text{proj}} \;=\; f_\theta\big(\mathcal{M}_{\text{in}}\big)
\;=\;\operatorname{supp}\big((f_\theta)_\#\,\mu_{\text{in}}\big)\;\subset\;S^{255}
$$
- 本文討論所謂流形都是各抽象層次上 ambient 的嵌入子流形，其內稟維數即切空間維數。
- Music JEPA 的訓練數據從 N 個音頻基礎上進行變速變調裁切加噪失真變換合成。因此logmel數據實際是原始內容乘以增廣手段所得，輸入流形的內稟維度也包含內容維度$d_{\mathcal C}$和增廣維度$d_{\mathcal A}$：

$$
\underbrace{c\in\mathcal C}_{\text{內容}}
\;\times\;
\underbrace{\alpha\in A}_{\text{增廣}}
\;\xrightarrow{\;\Psi\;}\;
x \in \mathbb{R}^{11008}
$$

- 嵌入子流形內稟維度的降低，更接近表徵學習降維的本質——習得 anchor 和 degraded version之間的invariance。從輸入流形的$d_{\text{in}}\approx d_{\mathcal C}+d_A$ 降低到輸出流形的$d_{\text{proj}}\approx d_{\mathcal C}$。讓模型在檢索任務上只尊重內容，而忽略增廣（在此場景，增廣即退化）。

- 怎麼估計嵌入子流形的內稟維數？已有成熟的 intrinsic dimension estimation方法，如TwoNN、Levina–Bickel MLE。半徑 $r$ 的鄰域內點數按 $\sim r^{d}$ 增長；TwoNN / MLE 就是從最近鄰距離比反解這個 $d$。
