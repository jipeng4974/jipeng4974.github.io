+++
title = "Federated Learning"
date = "2023-06-01"
tags = ["Sys"]
description = "Federated Learning is a machine learning paradigm in which many mobile devices collaboratively train a model under the orchestration of a central server. Training data stays decentralized, user data is never collected, and only client model updates are uploaded to the central server, where they are aggregated into a new global model. Compared with in-center distributed training, it has unique advantages and challenges."
showFullContent = false
+++

# Federated Learning
[Federated Learning](https://arxiv.org/pdf/1602.05629.pdf) was proposed by McMahan in 2016. It refers to a machine learning paradigm in which many mobile devices collaboratively train a model under the orchestration of a central server, keeping training data decentralized, avoiding the collection of user data, and uploading only client model updates to the central server, where they are aggregated into a new global model.

A naive implementation of federated learning works as follows:
1. The central server selects a set of clients and has them download the model.
2. Each client computes an update based on its own data.
3. Each client uploads the update (i.e., the new complete model) to the central server.
4. The central server aggregates these models in some way (e.g., averaging) to produce a global model.

## Tackling the High Cost of Uploading Models
Federated learning has been widely discussed in the literature. [Federated Learning: Strategies For Improving Communication Efficiency](https://arxiv.org/pdf/1610.05492.pdf) points out that step 3 of the naive implementation is prone to becoming a communication bottleneck — the first problem to address — and proposes two approaches to reduce uplink communication cost: structured updates and sketched updates.

The federated learning problem can be formally stated as learning the parameters of a model. The parameters of a fully connected layer can be represented by a real-valued matrix (this is just a simplification, so we consider a single matrix) W ∈ R^(d1×d2), with shape (#input × #output), where d1 and d2 denote the output and input dimensions. The kernel of a convolutional layer is a 4d tensor (#input × width × height × #output) and must be reshaped to (#input × width × height) × #output.

Let W(t) denote the model in the current round (t), and W(i,t) the model after the local update; the update itself is H(i,t) = W(i,t) - W(t). The central server can aggregate these to obtain the new model: W(t+1) = W(t) + η(t)H(t), where H(t) = Sum(H(i,t))/N and η(t) is the learning rate.

### Structured Updates
A structured update means learning the update directly from a restricted space parameterized by a small number of variables, rather than learning the update of the entire model.

Structured updates, i.e., imposing structure on updates, come in two forms in the paper: low-rank matrices and random masks.

In the low-rank matrix variant of structured updates, the local model update H(i,t) must be a low-rank matrix with rank < k, where k is a fixed number. H(i,t) is expressed as H(i,t)=A(i,t)B(i,t), where A(i,t)∈ R^(d1×k) and B(i,t)∈ R^(k×d2). In the subsequent computation, an A(i,t) is randomly generated and treated as a constant, and only B(i,t) is optimized. This way the data of A does not need to be uploaded and can be collapsed into a random seed; only B(i,t) needs to be uploaded. This optimization is essentially a compression technique based on dimensionality reduction: a matrix is used to reduce the dimensionality of the original data, and then a reconstruction matrix is used to rebuild the original data from the reduced representation.
A(i,t) is randomly generated for every client in every round. This immediately yields a d1/k reduction in upload cost. Fixing B and training A, or training A and B simultaneously, were both tried, but neither works as well as fixing A and training B. The explanation is that A can be viewed as a reconstruction matrix (reconstructing the original vector from the transformed vector), while B is a projection matrix (projecting a vector onto a subspace of another vector). Fixing A and training B is effectively equivalent to solving the following problem: given a random reconstruction matrix, which projection matrix can recover the most information?

In the random mask variant of structured updates, the local model update H(i,t) must be a sparse matrix generated with some predefined random mask. Again, the mask is regenerated for every client in every round. The sparse mask can be collapsed into a random seed, so only the non-zero values of H(i,t) and the seed need to be uploaded.

### Sketched Updates
In sketched updates, a complete model update is learned first, and only after that is it compressed — with lossy quantization, random rotation, and subsampling — before being sent to the server.

Quantization: weights are probabilistically quantized, using a smaller scalar type as an unbiased estimator of the original weights.

Random rotation: essentially multiplying by a random orthogonal matrix, to guard against the case where most of the data is 0, which would make quantization perform poorly.

Subsampling: instead of uploading H(i,t), only a random subset of it is uploaded after subsampling.

## Challenges Facing Fully Decentralized Federated Learning Across Domains
[Advances and Open Problems in Federated Learning](https://arxiv.org/pdf/1912.04977.pdf) discusses the challenges of fully decentralized federated learning at the algorithmic, privacy, security, and engineering levels:
1. The central server can become a bottleneck and a single point of failure. This motivates p2p/fully decentralized designs.
2. Fully decentralized algorithms must cope with the limitations of client availability and network stability.
3. Designing a model averaging strategy that aims for the fastest convergence speed is difficult.
4. Decentralized scenarios make algorithms vulnerable to malicious attacks and unreliable data or labels.
5. Clients have limited communication bandwidth and battery power, and porting existing compression algorithms to mobile devices is difficult.
6. Privacy: how to prevent one client from reconstructing another client's private data.
7. Implementation: a blockchain, as a distributed ledger, is essentially an eventually consistent replicated state machine. However, data on blockchains like Ethereum is public, so modifications would be needed to make it suitable for federated learning.
8. In cross-silo scenarios (multiple organizations or companies jointly training a model without directly sharing data — e.g., several banks jointly training a fraud detection model), data must be partitioned and incentive mechanisms added.
9. Communication and compression bottlenecks.
10. Fairness: federated learning introduces new sources of bias — device model, geographic location, activity patterns, local dataset size, and so on.
11. Secure computation: how to deal with a malicious server? How to deal with external attacks?

### The Non-IID Data Distribution Problem
The Non-IID (independent & identically distributed) data problem — where the statistical properties of samples are not uniformly distributed — is common for any client-partitioned dataset.

The paper presents several types of non-identical client distributions (considering supervised learning on features x and labels y, where (x,y)~Pi(x,y) is the local distribution of client i, and P(x,y) = P(y|x)P(x) = P(x|y)P(y)):
- Feature distribution skew (covariate shift): different clients share the same P(y|x), but the marginal distribution of features P(x) differs. For example, in handwriting recognition, different users write the same characters, but their strokes and writing habits still differ.
- Label distribution skew (prior probability shift): different clients share the same P(x|y), but the marginal distribution of labels P(y) differs. For example, an everyday animal recognition app in Australia would frequently see kangaroos in its labels, while other regions would not.
- Same label, different features (concept drift): different clients share the same P(y), but the conditional distribution P(x|y) differs — the same label corresponds to different features on different clients. For example, a luxury mansion means something different in scale in Hong Kong than in California.
- Same features, different label (concept shift): different clients share the same P(x), but the conditional distribution P(y|x) differs — the same features are assigned different labels. For example, some people label pandas as pets, while others label them as wild beasts.
- Quantity skew or unbalancedness: the amount of data varies greatly across clients.

Violations of independence are equally common, because the distribution of clients is easily affected by the constraints that trigger training: for example, since much training runs during nighttime sleeping hours, clients in regions along the same longitude are more likely to end up training together.

One feasible way to deal with Non-IID data is data augmentation with a small, globally shared dataset that contains no private data. In addition, one can cap the contribution a single user can make per day to avoid imbalance in quantity. Furthermore, in some scenarios Non-IID can be turned from a bug into a feature: simply train a locally customized, specialized model that provides a personalized service, rather than ultimately producing a global model. The paper goes on to introduce optimization algorithms and convergence rates on Non-IID datasets.

### Tackling Privacy: Split Learning
Split Learning is a cross-cutting split at the level of the model execution path, applicable to both training and inference: in the simplest scenario, each client keeps running the forward pass until it stops at a specific cut layer, then passes the output of the cut layer (the smashed data) to the central server or a peer to continue the computation — thus completing forward propagation without any data sharing. Similarly, gradient backpropagation runs from the last layer down to the cut layer and stops there, and only the gradient of the cut layer is passed back to the client. Throughout the entire process, no other node directly accesses the local data.

Given that the weights of the cut layer themselves can, to some extent, reflect the underlying reality of the data, whether Split Learning can provide a formal privacy guarantee remains an open question.

## Tackling High Communication Latency
There is another key problem — high communication latency — which is unavoidable due to the nature of wireless and long-distance transmission, but was not well addressed in earlier papers. [Delayed Gradient Averaging: Tolerate the
Communication Latency in Federated Learning](https://dga.hanlab.ai/assets/neurips21_dga.pdf) proposes an algorithm that delays gradient averaging. Using a 16-node Raspberry Pi cluster to simulate real-world mobile nodes and a wireless network environment, the authors empirically demonstrate that delayed gradient averaging allows the federated learning process to tolerate high network latency without sacrificing accuracy.

In an in-center environment, latency within the same rack is <1us, and within the same datacenter it is on the order of milliseconds. Wireless environments are around 20ms, and transoceanic links are at least 100ms. Once the bandwidth problem is solved, latency becomes the biggest bottleneck. The core idea of the DGA (Delayed Gradient Aggregation) algorithm proposed in this paper is to delay gradient averaging to some future iteration — that is, the model updates while receiving a stale averaged gradient — thereby allowing communication and computation to be pipelined. The paper formalizes the problem as minimizing a sum of stochastic functions.

![DGA](https://jipeng4974.github.io/img/DGA.png)

N denotes the number of clients, and fi denotes the stochastic loss function of client i. The random variable ζi is associated with a mini-batch sample.


![DGA2](https://jipeng4974.github.io/img/DGA2.png)

The main idea of the algorithm is to allow local updates to proceed while the averaging communication is in progress (averaging communication and local updates run in parallel). In FedAvg, clients send parameters to each other at the end of each round and wait for averaging to finish before starting the next round. DGA delays the averaging barrier to a subsequent iteration (iteration here refers to an iteration of the local update). Clients can therefore start the next round immediately (round here refers to the outermost loop, i.e., one round of updates). When the external information from the first round arrives, D iterations have already taken place, and the gradient correction is applied after a delay of D iterations. Ideally, with no communication latency, D=0, and DGA reduces to the original FedAvg.

1. In round t, clients send updates to each other.
2. After a local update, clients continue performing local updates with the latest local parameters. (Averaging communication latency > a single or even several local updates.)
3. By the time the round-t information from other clients arrives, a client has already performed D additional local updates.
4. The local round-t gradient is replaced with the received averaged gradient.

In the most extreme scenario (extremely high latency), the delayed gradient may take several rounds to arrive. This requires expressing the delay parameter D as D = sK + r, where s>=0, r <=K. DGA still guarantees that different clients differ only in their most recent D gradients; gradients from before round t-D are all identical.
