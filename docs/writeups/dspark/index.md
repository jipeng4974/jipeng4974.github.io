# DSpark

> DSpark = semi-autoregressive draft (heavy parallel backbone + lightweight serial head) + confidence-scheduled verification.

---

LLMS index: [llms.txt](/llms.txt)

---

The average per-token latency of speculative decoding is $L = (T_{draft} + T_{verify}) / \tau$, where $\tau$ is the number of tokens accepted per round. The DSpark paper[^1] (code in DeepSpec[^2]) is organized around the three variables in this formula: it fixes the missing dependency modeling of parallel drafters (raising $\tau$ without increasing $T_{draft}$), and turns verification length from a static hyperparameter into a resource-allocation problem solved at every step (lowering the effective $T_{verify}$). The latter essentially feeds load information from the serving system back into the algorithm layer.

## The Dilemma

Autoregressive drafters (the EAGLE family[^3]) draft serially, token by token, so $T_{draft} \propto \gamma$ — like a very short serial pipeline whose stages cannot be skipped. You are forced into a small $\gamma$, and to compensate for the acceptance rate you bring in tree-attention verification, where large numbers of candidate tokens occupy the target's batch capacity for nothing.

Parallel drafters (the Medusa[^4]/DFlash[^5] family) emit all $\gamma$ positions in a single forward pass, so $T_{draft}$ is independent of block length. But each position is predicted independently, with no intra-block dependency modeling, which produces multi-modal collisions: when the context admits both "of course" and "no problem", parallel prediction can stitch together "of problem". The symptom is an acceptance rate that decays rapidly along the block (suffix decay) — the larger $\gamma$, the greater the waste.

There is also a neglected systems dimension: fixed-length verification is not optimal under real serving. The optimal verification length varies along two axes —

- the data axis: code naturally has a higher acceptance rate than open-ended chat;
- the system axis: under light load, verifying a few extra tokens is nearly free; under heavy load, verifying tokens that are doomed to be rejected steals batch capacity from other requests. This is exactly why MTP uses only 1 token in production.

## Semi-Autoregressive

DSpark's draft has two stages. The parallel stage is DFlash's skeleton: a 5-layer block-parallel backbone that takes `[anchor_token, mask×(γ-1)]` as input, runs a single forward pass with bidirectional intra-block attention (`is_causal=False`), and produces the hidden states and base logits for all $\gamma$ positions. The key mechanism is KV injection: at prefill time, hidden states from several intermediate layers of the target are concatenated, projected, and injected into the K/V of every draft attention layer — the K/V at context positions is computed from target features rather than from the draft's own history, which amounts to linearly reading out the target's intermediate representations and using them as external memory for the draft. The draft shares the embedding and LM head with the target; both are frozen.

The sequential stage is a low-rank Markov head that adds a transition bias, conditioned on the already-sampled prefix, on top of the base logits:

$$p_k(v \mid x_0, x_{<k}) = \mathrm{softmax}\bigl(U_k(v) + B(x_{k-1}, v)\bigr), \qquad B = W_1 W_2,\; r=256$$

At inference time it runs a cheap left-to-right loop of $\gamma$ steps: each step looks up `W_1[x_{k-1}]` with the previously sampled token to compute the bias, adjusts the logits, and samples. The per-token probabilities are still exact softmaxes — this is the precondition for the losslessness of rejection sampling, and the dividing line between it and CRF-NAT (whose global partition function cannot yield exact per-token probabilities) and CTC (greedy only).

The effect: heavy dependency modeling is done once, in parallel, and the serial dependency is compressed into a low-rank $V \times V$ bigram table. $T_{draft}$ barely changes (measured at only +0.2%-1.3% whole-round latency over DFlash), and suffix decay is significantly mitigated. The paper also provides a GRU-style RNN head, which brings only marginal gains on long proposals; deployment defaults to Markov — a very engineering-minded trade-off.

## Confidence as Admission

The confidence head is a single linear layer that outputs a scalar $c_k$ per position, modeling the conditional survival probability (the probability that the $k$-th token is accepted given that all of the previous $k-1$ were accepted). The supervision signal is analytic: the per-step acceptance rate is exactly $1 - \tfrac{1}{2}\|p_d - p_t\|_1$, so it is trained directly with BCE, no extra labeling needed. Neural confidence is systematically overconfident, so post-hoc per-position temperature scaling (STS) calibrates the ECE of the cumulative product, from 3%-8% down to 1%; temperature scaling is order-preserving and does not disturb the token ranking.

On the scheduling side, the problem is formalized as global throughput maximization. With $R$ active requests, prefix survival probabilities $a_{r,j} = \prod_{i \le j} c_{r,i}$, verification batch token count $B = \sum_r (1+\ell_r)$, and expected accepted tokens $\tau = \sum_r (1 + \sum_j a_{r,j})$. At engine startup, SPS(B) is profiled once (steps/s vs. batch size — a lightweight cost table), and the objective is

$$\max\; \Theta = \tau \cdot \mathrm{SPS}(B)$$

— the expected benefit of verifying one more token and the marginal cost of SPS declining as the batch grows are weighed inside a single objective. The algorithm: greedily admit all candidate tokens in globally descending order of confidence, and early-stop when $\Theta$ no longer improves.

To someone from a systems background like me, this is admission control: admission decisions with a revenue model, a cost function given by the measured SPS curve, and a lightweight global resource allocation performed once per decode step. The DFlash/EAGLE debate, meanwhile, is wide-and-shallow single-issue versus narrow-and-deep multi-issue, and DSpark uses a low-rank table to push the cost of serial dependency to nearly zero.

## The Causality Barrier

$c_{k+1}$ depends on the actual value of the sampled token $x_k$, so a decision about whether to verify the $(k+1)$-th token that leaks the value of $x_k$ introduces selection bias. Appendix A gives a concrete counterexample: a constructed hindsight-based scheduler shifts the output distribution from $(0.7, 0.3)$ to $(0.85, 0.15)$, no longer lossless. Early-stopping makes the truncation decision depend only on prefix information available before the decision point (non-anticipating), restoring strict losslessness.

The production deployment (DeepSeek-V4) adds an asynchronous adaptation: the theoretical algorithm assumes SPS is smooth and unimodal, but real SPS(B) is a discrete staircase, and Zero-Overhead Scheduling requires the next step's batch size to be known before the current step ends. The approach is to estimate the verification capacity $K$ using confidence from two steps earlier, turning the problem into dynamic top-K selection, and to drop the break in favor of a global search so it can cross SPS cliffs. Decisions are based only on history from two steps back, which naturally forms a causality barrier and preserves losslessness. The variable-length verification kernel flattens all tokens in the batch into independent elements processed uniformly; intra-sequence dependencies are conveyed through a marker tensor in the sparse attention, requiring changes to only two kernels: index-attention and compress.

## The Target Never Touches the GPU

An elegant trade-off on the training side: the target stays frozen throughout and is never on the GPU. The target is first run offline to capture multi-layer hidden states into a target cache (a custom binary format, randomly read via mmap, about 38TB of disk for a 4B model with the default configuration), and training only reads the cache; the target distribution is reconstructed by passing the target's last-layer hidden states through the shared lm_head.

Each sequence randomly samples 512 anchor positions, builds a 7-token block for each, and packs them into a dense batch; the attention mask is expressed with flex_attention's `create_block_mask`: bidirectional within a block, isolated across blocks, and able to attend to the context before its own anchor. The three losses are weighted by a position-wise exponential decay $w_k = \exp(-(k-1)/\gamma)$:

- CE (0.1): teacher-forcing per-position cross-entropy;
- L1 distillation (0.9): $\|p_d - p_t\|_1$, directly optimizing the acceptance rate;
- confidence BCE (1.0): the soft label is the detached $1 - \tfrac{1}{2}\|p_d - p_t\|_1$.

An implementation observation: in the DeepSpec codebase, DFlash is not a separate model but a special configuration of DSpark (`markov_rank=0`, `confidence_head_alpha=0`, pure CE loss).

## Chain Rejection Sampling

Standard speculative sampling, no tree: the target runs a single forward pass over the draft+1 tokens; `accept_prob = clamp(p_t/p_d, 1)` is cumprod-ed to obtain the accepted prefix; at the first rejection, a corrected token is sampled from the residual distribution `norm(max(p_t - p_d, 0))`; if all are accepted, a bonus token is sampled. KV cache handling: on the target side, crop rolls back the rejected tokens; on the draft side, the K/V of noise tokens is cropped immediately after each round's block forward, keeping only the accepted-context portion (K/V computed from target hidden states, appended incrementally).

## Performance Gains and Limitations

Offline (Qwen3-4B/8B/14B, Gemma4-12B, 9 benchmarks, average acceptance length $\tau$): +27%-31% over Eagle3, +16%-18% over DFlash; a 2-layer DSpark already beats a 5-layer DFlash; the larger $\gamma$, the bigger the advantage — at $\gamma=15$ it leads DFlash by 22%-30%. Position by position, the parallel backbone delivers a very high first-position acceptance rate (chat 0.72 vs. Eagle3 0.53 — the first token has the most leverage in prefix matching), while the Markov head suppresses suffix decay.

Online (DeepSeek-V4-Flash/Pro preview, real traffic, compared against the production baseline MTP-1): +51% aggregate throughput under an 80 tok/s/user SLA; +60%-85% single-user speed at matched throughput. The load adaptation behaves as expected: at low-to-medium concurrency the verification budget expands from a static 2 tokens to 4-6 tokens, and it shrinks smoothly as concurrency saturates. As for the +661% in the paper, better take it with a grain of salt[^6].

The limitations are equally clear. The prefix scheduler only reduces verification waste; the parallel backbone's compute for generating the whole block is a sunk cost that cannot be recovered for queries with extremely low acceptance rates — the paper itself lists difficulty-aware early exit as future work. The global optimality of the early-stopping greedy algorithm depends on $\Theta$ being unimodal, while real SPS is sawtooth-shaped, which is worked around with the asynchronous two-steps-ahead estimate. The SPS(B) assumption ignores the effect of context length on decode latency, and this assumption rests on the premise of "average context far below 1M + PD-disaggregated load balancing".

[^1]: DSpark: Confidence-Scheduled Speculative Decoding with Semi-Autoregressive Generation. [[arxiv]](https://arxiv.org/abs/2607.05147)
[^2]: DeepSeek. DeepSpec: a training/evaluation library for draft models, including Eagle3, DFlash, and DSpark. [[github]](https://github.com/deepseek-ai/DeepSpec)
[^3]: EAGLE-3. [[arxiv]](https://arxiv.org/abs/2503.01840)
[^4]: Medusa: Simple LLM Inference Acceleration Framework with Multiple Decoding Heads. [[arxiv]](https://arxiv.org/abs/2401.10774)
[^5]: DFlash. [[arxiv]](https://arxiv.org/abs/2602.06036)
[^6]: This number appears at the strict 120 tok/s/user SLA point, where the MTP-1 baseline has already entered the low-concurrency degradation zone and can only sustain very small concurrency. The paper itself notes that this should be read as "expanding the feasible interaction frontier" rather than a representative speedup. The +60%-85% single-user speed at matched throughput is the more honest number.
