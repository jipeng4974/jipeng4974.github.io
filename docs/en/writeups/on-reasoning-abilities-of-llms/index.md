# Artificial Intuition: Reasoning Abilities of LLMs

> A survey of recent research on LLMs, examining the boundaries of reasoning ability in current large language model architectures.

---

LLMS index: [llms.txt](/llms.txt)

---

## Do LLMs Have System 2?
The distinction between System 1 and System 2 comes from psychologist Daniel Kahneman's theory[^8], which describes two modes of thinking: System 1 is fast, automatic, intuitive, and effortless, while System 2 is slow, deliberate, analytical, and conscious.

Beyond the already-demonstrated "reflexive reasoning ability that maps data distributions onto disjoint low-dimensional subspaces[^7]" (System 1), do current LLMs possess any degree of "slow thinking," multi-step reasoning, and planning ability (System 2)?

When LLMs first burst onto the scene, there were many wild claims, such as "LLM as zero-shot planner" and "LLMs are zero-shot reasoners," but in hindsight these voices look more like bandwagon-jumping and hype.

Now, everyone from ordinary practitioners to renowned scholars such as Yann LeCun, Yoshua Bengio, Yi Ma, and Subbarao Kambhampati has gradually reached a consensus that LLMs do not have System 2. Many studies have corroborated this view with reasoning benchmarks, for example Huang et al., 2023[^3], Jin et al., 2023[^4], Valmeekam, Marquez, 2023[^5], Stechly et al., 2023[^6], and Dziri et al., 2023[^12].

A small number of scholars — Ilya Sutskever, for instance — have said in interviews that scaling alone is enough to produce System 2 capabilities, with no architectural innovation needed. But Ilya doesn't seem to be candid; his motives are unclear[^11]. Some researchers argue that LLMs augmented with Chain-of-Thought prompting do have reasoning ability, such as Saparov & He, 2023[^9] and Feng et al., 2023[^2]. Yet these results ultimately rely on external prompting, and their experimental methods cannot rule out the possibility that approximate information retrieval merely mimics logical reasoning.

Consider LLMs themselves: since the amount of computation a transformer performs each time it generates a response has a fixed upper bound, it is fundamentally impossible for one to devote disproportionate computation to any particular problem. This alone is enough to rationally refute the claim that LLMs possess a human-like System 2. After all, System 2 is by definition a slow-thinking mode that maintains focus over long periods; it ought to have the potential for unbounded thinking time.

## If System 1 Were Strong Enough, Could It Replace System 2?
Theoretically speaking, a transformer with infinite precision doesn't even need infinite weights — it has been proven Turing-complete[^10]. That is to say, as long as it learns in the right way, it is capable of learning any algorithm.

Suppose a future LLM had near-infinite precision and weights, and was trained and run on enormous compute. Then it might indeed be possible to simulate System 2 reasoning with System 1 — to simulate deductive reasoning with simple intuitive mapping. But consider that the LLMs we use for inference today are already down to 8-bit or even 4-bit precision, and even so the cost is becoming unsustainable. There is good reason to regard this line of thinking as impractical: the energy consumption and material substrate this architecture would require could easily exceed the limits of human material civilization by several orders of magnitude.

Could an invincibly strong System 1 conjure "consciousness" into existence in an instant? Consciousness emerging as the input passes through the transformer's layers, then dying out in the softmax over the logits. Under the infinite-precision assumption this doesn't seem impossible either — after all, if you can simulate any algorithm, simulating a kind of System 2 is no surprise. It would amount to manufacturing a virtual machine of consciousness/life within a single execution, even though the physical machine actually executing it is nothing more than a plain transformer — a purely autoregressive model whose only goal is to increase the likelihood of predicting the next token. This is also the theoretical basis for Ilya's belief that scaling is sufficient to produce AGI.

## Artificial Intelligence? More Like Artificial Intuition
In sum, calling LLMs artificial intelligence is still an exaggeration; in principle they are a form of artificial intuition. This intuition might, under the assumptions of infinite compute, infinite precision, and infinite weights, simulate near-human intelligence. But honest datacenter AI practitioners will admit that today's large models are already approaching the hardware limits of advanced computing and advanced interconnects in terms of scale — and compared to algorithmic breakthroughs, the hardware development curve is flat and has a definite upper bound.

[^1]: A Survey on Hallucination in Large Language Models: Principles, Taxonomy, Challenges, and Open Questions [[pdf]](https://arxiv.org/pdf/2311.05232.pdf) 
[^2]: Language Models can be Logical Solvers [[pdf]](https://arxiv.org/pdf/2311.06158.pdf)
[^3]: Large Language Models Cannot Self-Correct Reasoning Yet [[pdf]](https://arxiv.org/pdf/2310.01798.pdf)
[^4]: Can Large Language Models Infer Causation from Correlation?[[pdf]](https://arxiv.org/pdf/2306.05836.pdf)
[^5]: Can Large Language Models Really Improve by Self-critiquing Their Own Plans? [[pdf]](https://arxiv.org/pdf/2310.08118.pdf)
[^6]: GPT-4 Doesn't Know It's Wrong: An Analysis of Iterative Prompting for Reasoning Problems [[pdf]](https://arxiv.org/pdf/2310.12397.pdf)
[^7]: White-Box Transformers via Sparse Rate Reduction [[pdf]](https://arxiv.org/pdf/2306.01129.pdf)
[^8]: Thinking, Fast and Slow
[^9]: Language Models Are Greedy Reasoners: A Systematic Formal Analysis of Chain-of-Thought [[pdf]](https://openreview.net/pdf?id=qFVVBzXxR2V)
[^10]: On the Turing Completeness of Modern Neural Network Architectures [[pdf]](https://arxiv.org/abs/1901.03429)
[^11]: An irresponsible conjecture: given that OpenAI was attempting architectural breakthroughs like Q*, Ilya likely had a motive to deliberately mislead in his interviews, being unwilling to reveal OpenAI's research direction. Moreover, emphasizing scaling helps highlight his own historical contributions, and exaggerating the AGI crisis benefits the superalignment project he was in charge of.
[^12]: Faith and Fate: Limits of Transformers on Compositionality [[pdf]](https://arxiv.org/abs/2305.18654)
