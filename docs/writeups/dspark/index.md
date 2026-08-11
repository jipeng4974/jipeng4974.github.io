# DSpark

> DSpark：低秩bigram赎回块内依赖，置信度head给验证长度做admission control。

---

LLMS index: [llms.txt](/llms.txt)

---

投机解码的每个token平均延迟是$L = (T_{draft} + T_{verify}) / \tau$，$\tau$是每轮接受的token数。DSpark的工作[^1]（代码在DeepSpec[^2]）围绕公式中的三个变量展开：它修正了并行drafter的依赖缺失（提$\tau$而不涨$T_{draft}$），又把验证长度从静态超参变成了一个每步求解的资源分配问题（降有效$T_{verify}$）。后者本质上是把serving系统的负载信息反馈进了算法层。

## 两难

自回归drafter（EAGLE系[^3]）逐token串行起draft，$T_{draft} \propto \gamma$，像一条很短但级数不可省的串行流水线。只能用小$\gamma$，为补偿接受率又引入tree attention验证，大量候选token白白占用target的batch容量。

并行drafter（Medusa[^4]/DFlash[^5]系）单次前向输出全部$\gamma$个位置，$T_{draft}$与块长无关。但各位置独立预测、块内无依赖建模，产生multi-modal collision：context同时允许"of course"和"no problem"时，并行预测可能拼出"of problem"。表现为接受率沿块快速衰减（suffix decay），$\gamma$越大浪费越多。

还有一个被忽视的系统维度：固定长度验证在真实serving下不是最优。最优验证长度沿两个轴变化——

- 数据轴：code的接受率天然高于开放式chat；
- 系统轴：轻载时多验证几个token近乎免费；重载时验证注定被拒的token会挤占其他请求的batch容量。生产上MTP只用1个token，原因正在于此。

## 半自回归

DSpark的draft分两阶段。Parallel stage是DFlash的骨架：5层块并行backbone，输入`[anchor_token, mask×(γ-1)]`，单次前向、块内双向注意力（`is_causal=False`），产出全部$\gamma$个位置的hidden和base logits。关键机制是KV injection：prefill时取target多个中间层hidden states拼接投影，注入每层draft注意力的K/V——context位置的K/V由target特征算出而非draft自身历史，相当于把target的中间表征线性读出后当作draft的外置记忆。draft与target共享embedding和LM head，均冻结。

Sequential stage是一个低秩Markov head，在base logits上叠加依赖已采样前缀的转移bias：

$$p_k(v \mid x_0, x_{<k}) = \mathrm{softmax}\bigl(U_k(v) + B(x_{k-1}, v)\bigr), \qquad B = W_1 W_2,\; r=256$$

推理时从左到右走$\gamma$步廉价循环：每步用上一个采出的token查`W_1[x_{k-1}]`算bias、修正logits、采样。逐token概率仍是精确softmax——这是rejection sampling无损性的前提，也是它和CRF-NAT（全局配分函数给不出精确per-token概率）、CTC（只能greedy）的分界线。

效果：重型依赖建模只并行做一次，串行依赖被压成一张$V \times V$低秩bigram表。$T_{draft}$几乎不变（实测比DFlash仅+0.2%-1.3%整轮延迟），suffix decay显著缓解。论文还给了一个GRU式RNN head，仅长proposal有边际收益，部署默认Markov——这个取舍很工程。

## 置信度即准入

Confidence head就是单个线性层，每位置输出标量$c_k$，建模条件存活概率（前$k-1$个全被接受的条件下第$k$个被接受的概率）。监督信号是解析的：逐步接受率恰好等于$1 - \tfrac{1}{2}\|p_d - p_t\|_1$，直接BCE训练，无需额外标注。神经置信度系统性过自信，post-hoc用逐位置温度缩放（STS）校准累积乘积的ECE，从3%-8%降到1%；温度缩放保序，不破坏token排序。

调度侧，问题被形式化为全局吞吐最大化。$R$个活跃请求，前缀存活概率$a_{r,j} = \prod_{i \le j} c_{r,i}$，验证batch token数$B = \sum_r (1+\ell_r)$，期望接受数$\tau = \sum_r (1 + \sum_j a_{r,j})$。引擎初始化时profile一次实测SPS(B)（steps/s vs batch size，一张轻量cost table），目标是

$$\max\; \Theta = \tau \cdot \mathrm{SPS}(B)$$

——多验证一个token的期望收益，和batch变大后SPS下降的边际成本，放进同一个目标里权衡。算法：全部候选token按置信度全局降序贪心准入，$\Theta$不再提升时early-stop。

对我这种做系统出身的人来说，这就是admission control：带收益模型的准入决策，cost function是实测的SPS曲线，每个decode step做一次轻量的全局资源分配。DFlash/EAGLE之争则是宽而浅的单发射对窄而深的多发射，DSpark用低秩表把串行依赖的代价压到了接近零。

## 因果屏障

$c_{k+1}$依赖已采样token $x_k$的具体取值，所以"是否验证第$k+1$个token"的决策若泄漏了$x_k$的取值就会引入selection bias。附录A给了具体反例：构造下回顾式调度使输出分布从$(0.7, 0.3)$变成$(0.85, 0.15)$，不再lossless。Early-stopping使截断决策只依赖决策点之前的前缀信息（non-anticipating），恢复严格无损。

生产部署（DeepSeek-V4）还有一层异步改造：理论算法假设SPS平滑单峰，真实SPS(B)是离散阶梯，且Zero-Overhead Scheduling要求当前step结束前已知下一step的batch size。做法是用两步之前的置信度估计验证容量$K$，转化为dynamic top-K选择，去掉break做全局搜索以跨越SPS悬崖。决策只基于两步前的历史，天然形成因果屏障，无损性得以保留。变长验证kernel则把batch内所有token flatten成独立元素统一处理，序列内依赖通过sparse attention里的marker tensor传达，只需改index-attention和compress两个kernel。

## target不进GPU

训练侧的一个漂亮取舍：target全程冻结且不在GPU上。先离线跑target抓多层hidden states写target cache（自定义二进制格式，mmap随机读，4B模型默认配置约38TB磁盘），训练只读cache；target分布由target最后一层hidden过共享lm_head重建。

每条序列随机采512个anchor位置，各构造7-token块，pack成稠密batch；注意力mask用flex_attention的`create_block_mask`表达：块内双向、跨块隔离、可attend各自anchor之前的context。三项loss按位置指数衰减加权$w_k = \exp(-(k-1)/\gamma)$：

- CE（0.1）：teacher-forcing逐位置交叉熵；
- L1蒸馏（0.9）：$\|p_d - p_t\|_1$，直接优化接受率；
- confidence BCE（1.0）：软标签是detach的$1 - \tfrac{1}{2}\|p_d - p_t\|_1$。

一个实现上的观察：DFlash在DeepSpec代码里不是独立模型，而是DSpark的配置特例（`markov_rank=0`、`confidence_head_alpha=0`、纯CE loss）。

## 链式拒绝采样

标准speculative sampling，没有tree：target对draft+1个token一次前向，`accept_prob = clamp(p_t/p_d, 1)`做cumprod得接受前缀，拒绝处从残差分布`norm(max(p_t - p_d, 0))`采样修正token，全接受则采bonus token。KV cache处理：target侧crop回滚被拒token；draft侧每轮block前向后立即crop掉noise token的K/V，只保留已接受context部分（K/V由target hidden算出，增量追加）。

## 性能提升与局限

离线（Qwen3-4B/8B/14B、Gemma4-12B，9个benchmark，平均接受长度$\tau$）：相对Eagle3 +27%-31%，相对DFlash +16%-18%；2层DSpark即超5层DFlash；$\gamma$越大优势越大，$\gamma=15$时领先DFlash 22%-30%。逐位置看，并行骨架给了很高的首位置接受率（chat 0.72 vs Eagle3 0.53——prefix-matching中首token杠杆最大），Markov head压制了后缀衰减。

在线（DeepSeek-V4-Flash/Pro preview，真实流量，对比生产基线MTP-1）：80 tok/s/user SLA下聚合吞吐+51%；吞吐匹配时单用户速度+60%-85%。负载自适应符合预期：中低并发时验证预算从静态2 token扩到4-6 token，并发饱和时平滑收缩。论文里+661%，better take it with a grain of salt[^6]。

局限同样清楚。Prefix scheduler只减少验证浪费，并行backbone生成整块的计算是沉没成本，对接受率极低的查询无法回收——论文自己提了difficulty-aware early exit作为未来方向。Early-stopping贪心的全局最优性依赖$\Theta$单峰，真实SPS锯齿状，靠异步两步前估计绕过。SPS(B)假设忽略context长度对decode延迟的影响，这个假设依赖"平均context远小于1M + PD分离负载均衡"的前提。

[^1]: DSpark: Confidence-Scheduled Speculative Decoding with Semi-Autoregressive Generation. [[arxiv]](https://arxiv.org/abs/2607.05147)
[^2]: DeepSeek. DeepSpec: draft模型训练/评测库，含Eagle3、DFlash、DSpark. [[github]](https://github.com/deepseek-ai/DeepSpec)
[^3]: EAGLE-3. [[arxiv]](https://arxiv.org/abs/2503.01840)
[^4]: Medusa: Simple LLM Inference Acceleration Framework with Multiple Decoding Heads. [[arxiv]](https://arxiv.org/abs/2401.10774)
[^5]: DFlash. [[arxiv]](https://arxiv.org/abs/2602.06036)
[^6]: 该数字出现在120 tok/s/user的严格SLA点上，此时MTP-1基线已进入低并发退化区、只能维持很小并发。论文自己也注明这应解读为"扩展了可行交互前沿"而非代表性加速比。吞吐匹配时单用户速度+60%-85%才是更诚实的数字。
