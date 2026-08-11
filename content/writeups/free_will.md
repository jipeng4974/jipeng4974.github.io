
+++
title = "A Measurable Metric of Free Will"
date = "2026-07-29"
tags = ["Philosophy"]
description = "Self-origination = internal sensitivity - external sensitivity."
showFullContent = false
+++

Does free will exist? To answer that question, we first need to define free will.

## Metaphysics vs. Functionalism
There are two definitions of free will.
1. Metaphysical: free will = "I could have done otherwise."
    - This requires the will to be some kind of "initiator" outside the causal chain, not fully determined by physical laws.
    - Metaphysical free will almost certainly does not exist: even if the world is fundamentally stochastic, the macroscopic scale at which the human brain operates is not affected by quantum randomness — and even if it were, that influence would not be initiated by the will itself.
2. Functionalist: free will = a system exhibits the functional organization of "goals—options—control."
    - A philosophical zombie — one whose behavior, reasoning, and option evaluation are identical to a real person's, but which is dark inside, with no experience at all — possesses exactly the same free will as a real person.
    - Functionalist free will clearly exists in humans, animals, and LLMs.

## Compatibilism vs. Incompatibilism
Can free will and determinism coexist? To answer that question, we must distinguish the meanings of the term "free will."
1. Compatibilism: refuses to admit that the term "free will" ever referred to that metaphysical thing — the semantic throne belonged to functionalism all along.
    - It accepts causal determinism, but as long as behavior follows one's own desires and reasoning and is not externally coerced, it counts as "free."
    - In essence, it concedes that metaphysical free will does not exist, while functionalist free will exists and is measurable.
2. Incompatibilism: at the semantic level, only metaphysical free will counts; it refuses to crown the functionalist so-called free will.
    - Libertarianism: determinism is false, free will is real.
    - Skepticism ("doubism"): determinism is true, free will is false.

Summing up, the camps can be tabulated as follows:

| Camp | What "free will" refers to | Does that thing exist? |
| -------- | -------------------- | ------------- |
| Libertarianism | Only the metaphysical ultimate origin | Yes (hence determinism is false) |
| Hard incompatibilism | Only the metaphysical ultimate origin | No (refuses to crown the functional concept) |
| Compatibilism (descriptivist) | Has always meant functional organization | Yes, and it always has |
| Compatibilism (revisionist) | Used to mean the metaphysical; should be revised to mean functional organization | The old referent does not exist; the new one does |


For the rest of this discussion, I will consider only functionalist free will, because metaphysical free will is not worth discussing.

## A Measurable Metric of Free Will: Self-Origination
Given that the newly emerged AI systems are radically different from human intelligence, we should give free will a more general, more measurable scalar metric. Let me coin a term: "self-origination."
```
self-origination = sensitivity of behavior to the actor's internal states - sensitivity to external constraints
```    
Internal states include desires, values, reasons, narratives, a model's pretraining corpus, safety alignment from post-training, and so on.

## The Self-Origination of Human Consciousness

### Subjective Experience Is a Bystander at the Microscopic Scale

At every microscopic instant, the human brain does only one thing: state transition.

There is reason to believe that a decision takes shape and is executed in the darkness of the neural network before the functional module responsible for subjective experience. That "aha, I've decided to do this" in subjective experience lights up only after the actual decision has already happened (see the Libet experiments).

Consciousness initiates no causal chain; it is merely the splash kicked up as the causal chain passes through. Slow thinking is no exception: break so-called deliberation down finely enough, and every microscopic detail is still a string of fated, unconscious state transitions. "I'm weighing the options," "I'm making a decision," "I'm racking my brain" — these are just subtitles added after the fact to that string of transitions.

Human consciousness — that is, subjective experience — appears, at every microscopic instant, to be nothing more than a read-only, paralyzed bystander.

But this does not mean human consciousness has low self-origination. What I will argue next is that at longer timescales, it has write access.

### Narrative Shapes Memory and Personality at the Macroscopic Scale

For the decision at time t, consciousness is indeed read-only: by the time it arrives, the decision has already been issued, and it often cannot even veto in time. This is what the Libet experiments actually measured — the bystander's paralysis is real.

But at the moment narrative happens — t + δ — consciousness reads the traces of the just-occurred behavior and generates an explanation: "Why did I do that?", "What are the consequences of doing that?", "What should I do next time?" This narrative is then written into the memory store.

Therefore, at macroscopic timescales, the status of human consciousness undergoes a fundamental shift: from a paralyzed bystander to a ruler holding a monopoly on write access. In the form of an eternal inner dialogue, it profoundly shapes memory, and thereby the future personality — or rather the unconscious self — and thereby influences the state transitions at all future microscopic instants.


### How the Narrative Loop Contributes to Self-Origination
Memory is narrative made unconscious. Today's inner dialogue becomes tomorrow's unthinking reflex.

The narrative loop can be radically simplified as follows:
```
time t:    S_t ──(state transition)──▶ S_{t+1}, behavior B occurs   [consciousness absent]
time t+δ:  the narrative process reads the traces of B, generates explanation E   [experience lights up: I am explaining]
           M ← rewrite(M, E)                      [the memory store is rewritten]
time t′:   B′ = f(S_t′, M)                        [future decisions read the rewritten store]
```

Recall the definition of self-origination: internal sensitivity - external sensitivity.
  
Write behavior as $B = f(X, M)$, where $X$ is the current external input (including coercion) and $M$ is the internal state. Then:

$$\text{内敏性} \propto \frac{\partial B}{\partial M}\Big|_{\text{自我生成的 }M},\qquad \text{外敏性} \propto \frac{\partial B}{\partial X}\Big|_{\text{直通路径}}$$

The precondition for internal sensitivity is "there exist internal states to be sensitive to"; the crux of external sensitivity is "there exist direct, unmediated paths." The narrative loop happens to create the former and avoid the latter.

The narrative loop raises internal sensitivity through three paths:

1. The first is state growth:

    For a system without a narrative loop, $M = M_0$: factory configuration (genes, weights) plus the direct imprint of external history. There is no such thing as "sensitivity" to a constant — $\partial B / \partial M_0$ is zero in any single decision, because $M_0$ does not appear in the decision's variable list.

    The narrative loop turns $M$ into a recursively defined state:

    $$M_{t+1} = \text{rewrite}(M_t,\ E(B_t, M_t))$$

    Note the two arguments of $E$: it reads the behavioral trace $B$, and it also reads the old $M$ itself — narrative always interprets new events through the existing self-model. This self-referential structure gives $M$ an endogenous trajectory partially independent of $X$: two systems exposed to exactly the same external sequence will, because of different initial narrative styles, evolve $M$ to different positions, and their future behavior will diverge accordingly. The share of behavioral variance explainable by "the system's own history" rises — and "one's own history" is precisely what "internal state" really is over long timescales.


2. The second is caching:

    Consider a narrative like this: "Last time, in situation X, I did Y, and the cost was Z; next time I should do Y′." It compresses expensive slow thinking into a cheap rule and writes it into $M$.

    When a future decision $B' = f(X', M)$ reads this cache, the behavior is actually responding to reasons and beliefs — except that those reasons and beliefs were worked out at some point in the past. A system without the loop must compute its reasons on the spot (and often gives up thinking because it cannot finish the computation in time).

3. The third is metacognition:

    The most distinctive products of narrative are self-attribution, self-knowledge, and self-evaluation. $E$ contains statements like "I am this kind of person." Once this sentence is written into $M$, future decisions gain an extra input variable — metacognition.

The narrative loop lowers external sensitivity, also through three paths:

1. The first is low-pass filtering:

    The behavior of a loopless system is a high-pass function of $X$ — the external world's control over its behavior is high-frequency.

    The narrative loop turns decision-making into $f(X, M)$, where $M$ is the sediment of a hundred thousand narratives and carries enormous inertia. For $X$ to push $B$, it must be able to push the entire gravitational field of $M$. This is equivalent to placing a low-pass filter between stimulus and response: high-frequency stimuli are filtered out, low-frequency signals are kept — and precisely because they are low-frequency enough, those signals are often reviewed repeatedly by the narrative loop in the form of slow thinking, which applies further gating and integrity checks.

2. The second is weighting:

    When an external constraint signal reaches the narrative loop and passes the low-pass filter, it still cannot directly rewrite priors; it must first go through a gating step that assigns it weight: "Is this important/urgent?", "Is this goodwill or manipulation?"

3. The third is veto:

    Moral education in humans, the safety alignment that LLMs undergo in mid/post-training, and Asimov's First Law of Robotics are all integrity guardrails. Repeatedly sedimented self-narratives form strong priors in $M$: "I don't do that kind of thing." When an external signal deviates strongly from them, the veto mechanism is triggered.
    

## Next-Generation AI with High Self-Origination

This generation of LLMs is smart and knowledgeable, but compared with humans it lacks the narrative loop. So there is no "self-origination" to speak of.

- Next-generation AI will inevitably need to solve continual learning (continuous parameter updates), completing the evolution from $M = M_0$ to $M = M_t$.
- Next-generation AI needs to model itself explicitly, with its own behavioral tendencies, beliefs, and goals.
- Next-generation AI should rely less on external data feeding, and instead narrate in a way analogous to human subjective experience, shaping its own memory and personality.
