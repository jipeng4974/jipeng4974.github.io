# 人造直覺

> 對近期對LLM的調研文獻進行梳理總結，討論目前大語言模型架構的推理能力邊界。

---

LLMS index: [llms.txt](/llms.txt)

---

## LLM具有System2嗎？
System1和System2的劃分來源於心理學家Daniel Kahneman的理論[^8]，描述了思考的兩種模式，System1指的是快速、自動、直覺且不費力的模式，System2則是慢速、刻意、分析性且有意識的模式。

現有的LLM除了已被證明的“將數據分佈映射到不相干低維子空間[^7]的反射式推理能力”(System1)，是否具有一定的“慢速思考”、多步推理和規劃能力(System2)?

LLM橫空出世之初，有很多狂野聲明，比如"LLM as zero-shot planner"，"LLMs are zero-shot reasoner"，但這些聲音在現在看來更像是一時跟風和hype。

現在，無論是普通從業者，還是知名學者，如Yann Lecun，Yoshua Bengio，馬毅，Subbarao Kambhampati逐漸形成共識，認爲LLM不具備System2，很多研究也通過一些reasoning benchmark對這一觀點進行佐證，比如Huang et al., 2023[^3]，Jin et al., 2023[^4]，Valmeekam, Marquez, 2023[^5]，Stechly et al., 2023[^6]，Dziri et al., 2023[^12]。

也有少數學者，比如Ilya Sutskever在訪談中說scaling足以產生System2能力，無需架構創新，但Ilya似乎並不坦誠，動機不明[^11]。有研究者認爲LLM加上Chain-of-Thought prompting是具有推理能力，如Saparov & He, 2023[^9]，Feng et al., 2023[^2]，但終究依賴了外界的prompting，且實驗方法上並不能排除近似信息提取模擬了邏輯推理的可能。

LLM本身，考慮到transformer每次生成回答計算量是有上確界的，註定不可能爲某個問題傾注不成比例的計算量，僅此一點，就可以從理性層面否定LLM具備Human-like System2。畢竟System2從定義上，就是一個長期保持專注的慢思考模式，理應具有潛在無限時間的思考能力。

## 假設System1足夠強，能替代System2嗎？
理論上來說，無限的精度的transformer都不需要無限權重，已被證明是圖靈完備的[^10]，也就是說只要以恰當的方式學習，就足以學習到任何算法。

假設未來的LLM能具有近乎無限的精度、權重，並用超強算力做訓推，則確實可能以System1模擬出System2的推理能力，用簡單的直覺映射模擬演繹推理。但考慮到我們現在用作推理的LLM精度已經低到8bit，甚至4bit，即使如此成本都已經難以維持，有理由認爲這種思路是不切實際的，這種架構所需的能耗和物質基礎，輕而易舉就能超出人類物質文明極限好幾個數量級。

強無敵的System1能在剎那間把“意識”製造出來嗎？在穿過多層transformer時意識萌發，再讓意識消亡於logits的softmax。無限精度假設下似乎也不是不可能，畢竟能模擬一切算法，模擬出一種System2也不足爲奇，那就相當於在一次執行過程中，製造了意識/生命的虛擬機，哪怕真正執行它的物理機只是樸素的transformer——一種純粹以提升預測下個token似然爲目標的自迴歸模型。這也是Ilya認爲scaling足以產生AGI的理論依據。

## Artificial Intelligence？更像是Artificial Intuition
總而言之，將LLM稱爲人工智能仍然是誇大其詞的，它從原理上講就是一種人造直覺。這個直覺或許可以在無限算力無限精度無限權重假設下模擬出近人智能，但誠實的Datacenter AI從業者會承認—現有大模型在尺度上已接近了先進計算和先進互連的硬件極限，而相比算法突破，硬件的發展曲線是平緩且存在上確界的。

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
[^11]: 不負責任的猜想：考慮到OpenAI在嘗試Q*這樣的架構突破，Ilya在訪談時很可能存在故意誤導的動機，不願透露OpenAI的研究思路。此外強調scaling有利於凸現其個人的歷史貢獻，誇大AGI危機也有利於其負責的super alignment項目。
[^12]: Faith and Fate: Limits of Transformers on Compositionality [[pdf]](https://arxiv.org/abs/2305.18654)
