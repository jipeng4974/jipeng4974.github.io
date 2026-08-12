# Knowledge is Embeddings of Reality

> Reality is infinitely vast and infinitely deep; once embedded into our limited cognitive space, it becomes knowledge.

---

LLMS index: [llms.txt](/llms.txt)

---

Reality has infinite dimensions, and behind reality there are latent realities; at the edges of a structure, new and deeper structures emerge.

Human perception, cognition, and understanding of reality are essentially projections of high-dimensional reality onto a low-dimensional representation. In statistics and machine learning this low-dimensional representation is called an embedding. That's right — knowledge is an embedding of reality into the limited cognitive space of humans: compact, shallow, and one-sided.

Take a concrete example: the rainbow observed by the human eye appears in seven colors.

But this is only the first observation of an infinitely vast reality, the most intuitive cognition.

Through the study of human physiology, we discovered that seven-color perception is an emergent structure of a deeper latent reality: the superimposed responses of multiple photoreceptors.

The human color perception system has three types of photoreceptors sensitive to red, green, and blue; the three peaks and four valleys produced by their superimposed responses determine that humans can perceive 7 colors.
![color](https://wujipeng.com/img/color.png)

Birds, because they possess photoreceptors sensitive to ultraviolet light, can perceive 9 colors, and therefore can see a nine-colored rainbow.

Whether it is the human seven-colored rainbow or the birds' nine-colored rainbow, both are merely knowledge (or embeddings), not reality.

Through mathematical tools and the study of electromagnetism, humanity further realized that the spectrum is continuous, including radio waves, microwaves, heat, infrared, visible light, ultraviolet, and so on.

Is this close to reality? No, this is still just a shallow embedding.

Through the study of quantum physics, we discovered that existing electromagnetism is still an emergent model of a deeper latent reality — electromagnetic waves are a manifestation of a stream of quantum particles: photons.

The understanding of the photon reveals an even more complex reality, because a photon simultaneously possesses infinite positions and frequencies, and its position or energy can only be determined once it is observed; it must be described by the Heisenberg uncertainty principle and the Schrödinger equation. And this is still not the truth, not the ultimate — it remains a structure imposed upon reality.

Similarly, a song in reality, seemingly simple and ordinary, actually possesses infinite dimensions. We can understand it from any angle and embed it into any vector space.

Suppose a user's phone sensors capture a piece of audio and upload it to a platform. Deduplication (mask features of the time-frequency diagram), fingerprinting (landmark features of the time-frequency diagram), cover-song recognition (a shallow CNN), humming recognition (the Whisper large model), and multimodal large model applications each capture certain features of this audio, producing knowledge (embeddings) at different levels. The deeper the understanding, the less precise it becomes — and the more flexible the application, the closer to human thinking.

The backend deduplication service or fingerprinting service computes a time-frequency diagram from the sampled signal data representing the audio via FFT, then extracts certain mask features or landmark features from the diagram that directly characterize its energy distribution, and compares them against existing features stored in an audio feature database to find matching audio. This is the process of audio deduplication and fingerprint recognition: applying simple mathematical processing to the signal generates knowledge that is highly specialized, concrete, compact, deterministic, and interpretable. The embeddings representing this kind of knowledge either lack noise robustness (mask features) or have a little of it (landmark features), but in any case they only work for matching the original track — they know nothing of covers, remixes, derivative works, or mashups.

A cover-song recognition service processes the audio signal with a neural network that has learned song melodies, producing an embedding that expresses melodic features to some degree. By storing the cover embeddings of the music library in a vector database, the results recalled via KNN/ANN are often not limited to the original track and can tolerate changes in key and timbre.

A humming recognition service can be built on an audio large model like Whisper. Because this deep network has learned human natural language, when it processes the audio signal it extracts an embedding that expresses lyrical information. Based on such an embedding, the service can still recognize what song the user is humming even when they are severely off-key and there is no accompaniment.

Looking ahead to future multimodal large models: embeddings could even be stored directly in the FFN layers of a transformer, without relying on an external vector database or inverted index, modeling the complex relationships between embeddings inside the model itself, and letting more concepts become associated — recognizing the mood of a song, the emotions of the singer, images of the lyrical content, and even connecting to the context of user interactions to derive more personalized, scene-aware outputs.

Large models in AI, simply built with transformers, possess unexpected logical ability, suggesting that the essence of human consciousness may have been extremely shallow all along: once knowledge is internalized and associated, Bayesian inference performed with our innate neural circuits produces logic. Admittedly, current architectures are far less efficient than the human brain, but the boundary between having it and not having it has already been crossed.

A sufficiently good computational neural network can be regarded as an emergent model of neuroscience.
As it always has been.
Just as neuroscience is an emergence of cell biology.
Just as cell biology is an emergence of molecular biology.
Just as molecular biology is an emergence of physical chemistry.
Just as physical chemistry is an emergence of quantum physics.

![em](https://wujipeng.com/img/em.png)
