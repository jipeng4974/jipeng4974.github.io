+++
title = "Beyond Representation Learning: Prediction-Driven Encoders + Learnable Sensors"
date = "2026-06-29"
tags = ["research", "ai"]
description = "Existing multimodal representation learning is primitive, isolated, and incomplete — it still needs prediction-driven encoders and learnable sensors."
showFullContent = false
+++

One of the core challenges of representation learning for high-dimensional signals is the enormous gap between the dimensionality of the input and that of the output.
- Audio and video computing has long carried a massive computational burden precisely because physical-world audio and video signals are extremely high-dimensional.
- The human brain processes a visual stream with a bandwidth of about 80 Mbps, while the speed of thought is about 800 bps — a difference of 5 orders of magnitude.

The second core challenge of representation learning for high-dimensional signals is that the input signal is necessarily some kind of sampling and profiling of physical reality.
- At different scales, the music you hear is different.
- From different viewpoints, the scene you see is different.
- At different resolutions, the structures worth attending to are different.
- Studying only the "signal -> representation" process is not enough; we must also consider how to let the model learn to design its own strategies for observing signals.

## Filtering Effective Signals by Prediction Error
Our brains are constantly predicting the next frame. With that prediction in place, only a small corrective signal is needed to synthesize a stable, coherent visual experience. That is how a low-power piece of computational wetware like the brain can comfortably handle an extremely high-dimensional visual signal arriving as an ultra-high-bandwidth input.

Strap an action camera without stabilization to your head and the footage shakes so badly it makes you question your life choices, yet our own visual experience while walking is stable and natural. This is direct evidence for the synthesis theory of visual experience.

When we arrive in a foreign country, fresh off the boat, even just walking down an ordinary street brings a sense of novelty, accompanied by the fatigue of sensory overload — this is nervous-system overload caused by the prediction model temporarily failing.

Similarly, in childhood, life moved slowly, time felt long, and the density of memories seems far higher than in adulthood. That is precisely because a child's world model is still immature, and fresh prediction errors come out of the oven every day.

BTW, intolerance to sound and light is also often a case of over-excitability caused by the prediction model failing. Light or signal flicker that is perfectly acceptable to most people can, for Aspies, bypass the hypothalamus and reach straight into the sanctuary of the eternal inner dialogue, causing deep physiological neural pain — often only after an exhaustive, fruitless search for an eye disorder does one learn that this is an overload of the nervous system.

Today's audio and video encoders remain, to a large extent, isolated and passive entities. They lack an **active prediction model** that proactively filters out mundane signals and focuses on the "surprising signals" to assist the encoder in effective dimensionality reduction.

Gemma4 12B makes a radical attempt (or maybe not so radical — from the perspective of following the bitter lesson, it is actually conservative): it removes the encoder entirely, directly reducing audio and video to the LLM's hidden size, thereby eliminating inductive bias at the encoder level. In theory, a Transformer has the capacity to learn a "prediction model + encoding strategy", but with only the standard next-token objective, it is hard to imagine it having the dynamics to learn a complex prediction-driven encoder.


## Returning the Power to Switch Viewpoints to the Model
Viewpoint has an enormous influence on intelligence; even beings as complex and refined as humans cannot resist it. Long-term use of a 14–35mm lens versus an 85mm lens subtly shapes one's state of mind and temperament in different ways. A wide angle gives one the boldness to take in the whole world through the lens, while a medium telephoto makes one quieter, more focused, and gentler.

In the biological world, switching viewpoints is cheap, efficient, and real-time — an essential endowment of any learner.

Even something as primitive as a lizard knows to move its eyeballs when observing the world.

Unfortunately, today's MLLMs simply lack this ability. Fixed patchification and a fixed viewpoint mean that high-dimensional signals, before they even reach the encoder, have already lost the possibility of a comprehensive and contextualized portrayal of the world.

Machine learning today cannot switch viewpoints on its own; everything relies on human design — the "observation strategy" is outsourced to researchers. Only after a long and expensive training run can researchers, through evaluation and reflection, possibly adjust the "observation viewpoint" — an extremely long feedback loop. Adjusting the dataloader/projector/sampler of a 12B model often takes a week and burns millions of kilocalories. By contrast, a lizard moves its eyeballs in 20 ms, consuming 1 microjoule.

As a learner, today's MLLM lacks a cheap, high-frequency, closed-loop viewpoint control system — a real-time embodied zoom lens and a brain that controls that zoom lens.

Out of respect for the bitter lesson, many MLLMs choose to feed more primitive representations to the model (for example, waveforms are generally more primitive than mel spectrograms), letting the model decide on its own how to learn useful knowledge from the raw representation. Does this count as handing viewpoint control over to the model? I don't think so. If you introduce no inductive bias at all, the model at inference time actually has no right to say no — it must swallow the massive audio-video signal shoved down its throat. Lowly creatures know how to refuse; an MLLM cannot refuse. As for training, needless to say, there isn't even a real-time feedback loop at the data-preparation level.
- The current training loop (data pipeline & learning algorithm) is entirely hard-coded. If in the future we could make the simpler part of it — the data pipeline — learnable (becoming part of the model's weights) — we might call it a learnable sensor — letting the model autonomously adjust its strategy for observing the world during training, that would at least close the feedback loop.

```
## In a non-learnable training loop, the data pipeline is obviously the easiest part to make learnable
for batch in dataloader:
    x = augment(batch)
    y = model(x)
    loss = criterion(y)
    loss.backward()
    optimizer.step()
```    

Does adding a learnable sensor run counter to the direction of simplifying/removing the encoder? No. Not only do the two not run counter to each other, they independently converge in echoing the bitter lesson: the learnable sensor is responsible for how to observe the data — a process currently hard-coded in code with zero flexibility, so any loosening of that inflexibility is progress, a reduction of human hard-coding. The encoder, meanwhile, is responsible for feature engineering once the data is in hand — a process where introducing too much inductive bias is genuinely unnecessary (even a CNN is a sin, let alone handcrafted features), so simplifying/removing the encoder is also a reduction of human hard-coding.
