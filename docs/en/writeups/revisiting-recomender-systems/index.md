# Revisiting Recommender Systems

> Notes on Professor Aixin Sun's talk, "Understanding the Current State of Recommender Systems Research" — a summary of the main content with links to the papers mentioned. A glimpse into the current state of academic research on recommender systems: incisive, interesting, and startling.

---

LLMS index: [llms.txt](/llms.txt)

---

The recommendation problem can be defined as recommending the right items to the right users at the right time.

Typically, the dataset used for a recommendation task is the user-item interaction matrix. A recommender system infers user interests from this dataset, then puts the results online for testing and evaluation (academia doesn't have this luxury and can only evaluate offline).

## What's Missing in User-Item Interaction Datasets
> We usually understand the recommendation task as predicting missing entries in a static user-item interaction matrix, rather than predicting a user's next interaction in a dynamic environment under specific circumstances.

It's worth noting that in user-item interaction matrix datasets, the time dimension has collapsed, and so have the various constraints of real-world interaction scenarios.

Seventy percent of recommender systems papers use the MovieLens dataset — but can it really reproduce a real recommendation scenario? Not necessarily, because MovieLens has users recall years of movie-watching experience in a single initialization session — with likely omissions and forgotten entries, and it ignores real-world factors like release dates and prices. In reality, a user's interests form gradually, subject to all kinds of real-world constraints, and their viewing decisions are inevitably influenced by release timing, ticket prices, whether their interests have shifted, and many other factors.

## A Worrying Analysis of Current Practice 
> Based on our past five years of work on recommender systems evaluation and dataset analysis, we revisit the problem definition of recommender systems and offer an interpretation for the lack of consensus.

Dacrema et al., 2019[^1] argue that new deep learning models deliver only mediocre results.

Bauer et al., 2024[^2] conducted a synthesis analyzing the datasets used, and found them to be extremely concentrated — dominated by mainstream datasets like MovieLens and Amazon Reviews, which are generally quite old. Most papers propose new models. A few focus on evaluation criteria.

Ivanova et al., 2023[^3] point out that the field has no consensus on which baselines to use. Part of the reason is that roughly 5,000 recommender systems papers are published every year — nobody can read them all, so reviewers never converge on a shared view. Some reviewers consider certain methods outstanding and insist they must be included as baselines, while others hold different opinions. Nearest neighbor, for example, is simple, yet with proper tuning it is a very strong baseline that outperforms complex models in many scenarios — but many papers don't include it as a baseline, because everyone considers it a decades-old method not worth comparing against.

Not only are baselines inconsistent, but even with unified baselines there is still the tuning problem. As Shehzad, Jannach, 2023[^4] note, when you propose your own model, you tune hyperparameters with great care, but when comparing against others' models you don't tune them nearly as finely.

McElfresh et al., 2022[^5] conducted a very large-scale study, comparing 315 metrics across 24 algorithms on 85 datasets, and reached a startling conclusion: these algorithms do not generalize — one that excels on one dataset may fail on the next. Every algorithm can rank first or second somewhere, and even the best can rank near the bottom elsewhere. In the end, the strongest algorithm turned out to be Item-kNN!


## Data Leakage in Train/Test Split
Sun, 2022[^6] surveyed the ``train/test split`` practices of 88 RecSys conference papers from 2020–2022 and found that 34% use ``random split``, 25% use ``leave-one-out`` (the last interaction becomes the test set, all preceding ones are training), 19.5% ``single time point``, 17% ``simulation-based online``, and 4.5% ``sliding window``. In theory, the perfect train/test split strictly follows the timeline: everything before a chosen time point goes into the training set, everything after into the test set, and the time point is gradually pushed forward — each choice of time point generates a corresponding training and test set, and the later the time point, the more training data and the less test data. Unfortunately, this is very hard to do in practice. The ``random split`` and ``leave-one-out`` methods used by most papers suffer from severe information leakage.

Take ``leave-one-out`` as an example: only the last interaction of each user goes into the test set. The problem is that different users' last interactions can occur at wildly different times — suppose some item is extremely popular during a specific period, ranking very high in ``popularity`` (popularity is the simplest baseline in recommender systems: just rank items by their interaction counts), but some user's last interaction happened before that period. Recommending a future hit to that user is clearly unreasonable. Ji et al., 2020[^7] revisited this problem and corrected ``popularity`` to use the "popularity at the time" of each user's last interaction, improving the confidence of ``popularity`` by 70%.

Ji et al., 2020[^8] point out that nearly all ML/DL models — especially recommender system models evaluated offline — suffer from this kind of data leakage: the model inadvertently trains on future data and learns user-item interactions that shouldn't yet exist. ``BPR``, ``NeuMF``, ``LightGCN``, and ``SASRec`` all fail to avoid this leakage by design. The study also demonstrates experimentally that this leakage does significantly affect recommendation accuracy, and that its impact on accuracy is unpredictable.

## Recommendation should be Task-specific & Dynamic
User decisions involve both general preferences and present context factors. The present context is highly task-specific and highly dynamic — and this gives the recommendation task the same character.

To a large extent, existing recommender systems are confined to the level of general preferences. Looking back at the data level: existing datasets — the various user-item interaction matrices — clearly discard context and can only capture general preferences. At the model level, training is based on decision outcomes rather than the decision process itself, which also means models can only learn users' general preferences. And on the evaluation side, naturally, only general preferences can be evaluated.

In industrial practice, a recommender system should be a retrieval problem. In this retrieval problem, the query contains two kinds of dynamically updated information — general preferences and current context; the item collection is also dynamically updated; and ranking aims to improve decision quality.

Given the context factor, recommender systems for different scenarios — food recommendation, movie recommendation, e-commerce recommendation, hotel recommendation — should have very different implementations and be modeled separately. Some scenarios have a fixed set of options, some are well-suited to repetition, and some emphasize exploration. The inputs of different scenarios even differ. For example, food delivery recommendation requires not just a user id but also the delivery address and whether it's breakfast, lunch, or dinner.

## Conclusions and What's Next
A recent paper by Professor Sun, Sun, 2024[^9], rethinks the problem definition of recommender systems and argues that current recommender systems research has oversimplified the recommendation problem to the point that almost none of the solutions proposed by academia suit concrete real-world tasks. It offers several judgments about future research directions:
- There will likely be no winner-takes-all model; in the future, every model will remain invincible in its own paper.
- The recommender systems problem should be subdivided: short video has its own track, e-commerce has its track, news has its track — design and evaluate new models for each track. Stop using MovieLens to evaluate e-commerce recommendation!
- Item-kNN will remain a very strong baseline. It just needs better feature engineering in how "nearest" and "neighbor" are defined.

[^1]: Are we really making much progress? A worrying analysis of recent neural recommendation approaches [[arxiv]](https://arxiv.org/abs/1907.06902)
[^2]: Exploring the Landscape of Recommender Systems Evaluation: Practices and Perspectives [[pdf]](https://arxiv.org/pdf/2311.05232.pdf) 
[^3]: RecBaselines2023: a new dataset for choosing baselines for recommender models [[arxiv]](https://arxiv.org/abs/2306.14292)
[^4]: Everyone’s a Winner! On Hyperparameter Tuning of Recommendation Models [[pdf]](https://dl.acm.org/doi/pdf/10.1145/3604915.3609488)
[^5]: On the Generalizability and Predictability of Recommender Systems[[arxiv]](https://arxiv.org/abs/2206.11886)
[^6]: Take a Fresh Look at Recommender Systems from an Evaluation Standpoint [[arxiv]](https://arxiv.org/abs/2210.04149)
[^7]: A Re-visit of the Popularity Baseline in Recommender Systems [[arxiv]](https://arxiv.org/abs/2005.13829)
[^8]: A Critical Study on Data Leakage in Recommender System Offline Evaluation  [[arxiv]](https://arxiv.org/abs/2010.11060)
[^9]: Beyond Collaborative Filtering: A Relook at Task Formulation in Recommender Systems [[arxiv]](https://arxiv.org/abs/2404.13375)
