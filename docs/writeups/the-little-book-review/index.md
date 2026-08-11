# The Little Book Review & Internalization

> Just as DDIA can be regarded as the go-to introductory text for distributed systems, LBDL is the ideal deep learning 101.

---

LLMS index: [llms.txt](/llms.txt)

---

"The Little Book of Deep Learning" ([```LBDL```](https://fleuret.org/francois/lbdl.html)) is a book by François Fleuret formatted for phone screens, offering a concise introduction to deep learning for readers with a STEM background. Just as ```DDIA``` can be regarded as the go-to introductory text for distributed systems, ```LBDL``` is the ideal deep learning 101.

![tlb](https://jipeng4974.github.io/img/tlb.jpg)

Conciseness — or compression — is precisely the strength of deep models, and a virtue in this age of information overload. Printed on A4 paper, this booklet is a very comfortable read.

---

What follows is my internalization and organization of the material.

## 1. Overview

High-dimensional signals are hard to analyze with rule-based systems, and deep networks overcome this difficulty by fitting a sufficiently good approximating function (one with a sufficiently low loss) through a deep mapping with a large number of weights. This function can map high-dimensional signals to continuous vectors (regression) or discrete values (classification), or it can be a probability density function; in any case, it learns a compact, discriminative representation from the data distribution.

If the data samples are insufficient, the model may perform well on the training data yet poorly in real applications — this is overfitting.
If the model lacks the capacity to adapt to varied scenarios and accurately capture the input-output relationship, the training loss stays high — this is underfitting.

Machine learning models can be broadly divided into three categories:
1. Regression models: supervised; the training data consists of pairs of input signals and ground-truth values, mapping high-dimensional signals to some vector.
2. Classification models: supervised; the training data consists of pairs of input signals and labels, mapping high-dimensional signals onto a finite label set.
3. Probability density function models: unsupervised; the training data is the input signal itself.

## 2. Training
### Loss Functions
Training is the process of reducing the loss function (loss, denoted $\mathscr{L}$) of the prediction function on the training set.

How is the loss function defined? For continuous values, the mean squared error is a standard choice. For probability densities, the likelihood is used — one can set $\mathscr{L}=-\sum f(x;w)$, where $f(x;w)$ is the normalized log-probability of each training sample. For classification tasks, cross-entropy is generally used.

What is cross-entropy? A classification model outputs N logits for N classes (LLMs actually do the same, generating a logit for every token in the vocabulary, representing unnormalized log-probabilities). Passing the logits through softmax yields the posterior probability $P(Y=y|X=x)$, a proper probability distribution in which the probabilities of all classes sum to 1. Setting $\mathscr{L}=-\frac{1}{N} \sum_{n=1}^N logP(Y=y_n|X=x_n)$, this $\mathscr{L}$ is the cross-entropy. Minimizing the cross-entropy maximizes the probability of the true class.

In metric learning, although the predicted values are continuous, the actual form of supervision is ranking, because the goal of metric learning is to learn comparable distances between samples. For example, given three points A, B, and C, where A and B are different views of the same face and C is a different person, the distance between A and B is required to be smaller than that between A and C. Metric learning therefore typically uses contrastive loss or triplet loss.

The loss function is usually only a proxy metric, not the actual performance metric. Take classification as an example: the direct performance metric should obviously be the classification error rate, but the gradient of this metric carries no useful guidance — the error-rate function is completely decoupled from the model weights, so knowing how the error rate changes cannot help the model reduce it during training.

The loss function can also be designed to depend on the model weights, thereby imposing some constraint and control over them. Weight decay, for instance — a regularization technique that prevents overfitting — adds a term equal to the sum of squared model weights to the loss function, penalizing large weight values in favor of small ones, and thereby reducing the influence of the training data on the range of weight values. This degrades performance on the training set, but helps the model generalize better to unseen data.

### Autoregressive Models
Autoregressive models are a key method for handling discrete sequences in fields such as NLP and CV. The principle relies on the chain rule of conditional probability:
$$P(A\cap B)=P(A) P(B|A)$$
$$P(A\cap B\cap C)=P(A) P(B|A) P(C|A\cap B)$$

An autoregressive model takes as input the T existing tokens (each token drawn from a vocabulary set of size K) and outputs logits for the K candidate tokens.

Scenarios with a finite token vocabulary are computationally tractable, and the chain decomposition of conditional probability further reduces the computation — when sampling the next token, the probability of the previous one can be reused, so that a token sequence conforming to the joint probability distribution can ultimately be generated.

Training an autoregressive model can iterate over every step, summing the cross-entropy snapshots between the model's prediction and the true next token at each logical time step to form the cross-entropy loss. Reducing this loss increases the likelihood of the model's predicted token at each logical time step. In practice, what is usually monitored is not the cross-entropy itself but the exponential of the cross-entropy (H), i.e. perplexity (PPL), PPL = 2^H. Compared with cross-entropy, perplexity is normalized and does not depend on the length of the input sequence.

During training, everything computed before has to be recomputed at every time step. Given that the total number of logical time steps is often quite long — hundreds, thousands, even tens of thousands — such computation is clearly very inefficient. The solution is to design a model that predicts the logits vectors at all logical time steps (T) in one pass — $f: \{1,...,K\}^T \rightarrow \mathbb{R}^{T\times K}$ — while ensuring that the logits $l_t$ corresponding to the input $x_t$ at time t depend only on $x_1, x_2, x_3, ... x_{t-1}$. Such models are causal models, whose principle is to never let the future influence the past.

![causal](https://jipeng4974.github.io/img/causal.png)

When training a causal model, the output can be computed over the full sequence, maximizing the probability of every token in the sequence in one pass, which is ultimately equivalent to minimizing the per-token cross-entropy.

NLP has an important technical detail: how tokens should be represented — they can be single symbols at the lowest granularity, or entire words. The algorithmic process that performs token representation is called a tokenizer. A standard approach is Byte Pair Encoding [Sennrich et al., 2015][^1].

### Gradient Descent
Except for simple special cases like linear regression, the optimal weights $w^*$ generally have no closed-form expression. In such cases, the tool for minimizing the function is gradient descent: initialize the weights to random $w_0$, then iterate repeatedly, modifying the weights along the gradient direction at each iteration so that the loss gradually decreases — that is, at each iteration set $w_{n+1} = w_n - \eta \nabla \mathscr{L}_{|w}(W_n). $ Here $\eta$ is the learning rate; if it is set too small, training may be too slow and can easily get stuck in a local minimum, while if it is set too large, the weights tend to oscillate back and forth near the minimum.

![gd](https://jipeng4974.github.io/img/gradient_descent.png)

For each point $w$, the gradient $\nabla \mathscr{L}_{|w}(w)$ is the direction that maximizes the increase of $\mathscr{L}$. Gradient descent therefore subtracts the learning rate times the gradient at each iteration, so that all the iterations chained together form a near-optimal route minimizing $\mathscr{L}$.

In practice, every loss can be expressed as the mean of the losses over multiple samples or even a single sample: $\mathscr{L} = \frac{1}{N} \sum_{n=1}^N \ell_n(w) $, where $\ell_n(w)=L(f(x_n;w),y_n)$, so the gradient can be written as:

$$\nabla ℒ_{|w}(w) = \frac{1}{N} \sum_{n=1}^N \nabla \ell_{n|w}(w)$$

Computing the full gradient is expensive, but the full sum can be estimated with a partial sum (this requires proper data shuffling, so that the stochasticity of the data removes the bias of the estimate). To make the computation fit in memory, the standard practice is to split the full training set into quite a few (up to millions of) batches, obtain an estimate of the gradient from each batch, and update the weights based on that estimate — this is mini-batch SGD (stochastic gradient descent). There are many variants of this algorithm, such as Adam [Kingma and Ba, 2014][^2].

### Backpropagation
Given $\ell(w)=L(f(x;w),y)$, how do we compute $\nabla\ell_{|w}(w)$? Since $f$ and $L$ are both compositions of standard tensor operations, their expressions can be derived via the chain rule, just like any mathematical expression.

![bp](https://jipeng4974.github.io/img/bp.png)

For simplicity, denote a model of depth $D$ as $f = f^{(D)} \circ f^{(D-1)} \circ ... \circ f^{(1)}$. The feedforward process then computes $x^{(d-1)} \rightarrow x^{(d)}$ in order, i.e. $x^{(d)} = f^{(d)}(x^{(d-1)};w_d)$, until finally obtaining $x^{(D)}$ as the model output.

Backpropagation instead computes in reverse
$\nabla\ell_{|x^{(d-1)}} \leftarrow \nabla\ell_{|x^{(d)}}$
and
$\nabla\ell_{|w_d} \leftarrow \nabla\ell_{|x^{(d)}}$,
where
$\nabla\ell_{|x^{(d-1)}}$[^3]
is the product of
$\nabla\ell_{|x^{(d)}}$[^4]
and
$J_{f^{(d)}|x}$[^5].

The other gradient we actually care about during training, $\nabla\ell_{|w_d}$, is the product of $\nabla\ell_{|x^{(d)}}$ and $J_{f^{(d)}|w}$[^6].

Deep learning training frameworks mainly deal with hiding the complexity of backpropagation gradient computation, i.e. providing automatic differentiation capabilities. This technique was also widely used before deep learning; see AutoGrad [Baydin et al., 2015][^7].

Clearly, backpropagation involves twice as much matrix computation as feedforward inference (each layer has one extra backpropagation pass for the weights). Backpropagation also has far greater memory requirements than feedforward inference, because every layer's $x^{(d)}$ must be kept in memory, whereas inference need not keep them — only the latest one. Techniques that address the excessive memory footprint include reversible layers [Gomez et al., 2017][^8] and checkpointing [Chen et al., 2016][^9].

One problem with deep models is vanishing gradients (see [Glorot and Bengio, 2010][^11]): after many rounds of backpropagation, the values become too large or too small. The standard countermeasure is gradient norm clipping [Pascanu et al., 2013][^10].

### Self-Supervised Training
GPT trained on a large-scale unlabeled training set is sufficient to handle many tasks, such as translation (see [Radford et al., 2019][^12]). This is a typical application of self-supervised training, whose most important advantage is the ability to exploit ultra-large-scale unlabeled data, pushing the boundary of training data scale even further.

## 3. Building Blocks
### Linear Layers
The fully connected layer is the most basic linear-layer building block, represented by a $D \times D'$ matrix $W$ and a bias vector $b$. It implements an affine transformation that generalizes to arbitrary tensor shapes: given any input $X$ of shape $D_1 \times \dots \times D_k \times D$, the fully connected layer computes an output $Y$ of shape $D_1 \times \dots \times D_k \times D'$: $\forall d_1,\dots,d_K, Y[d1,\dots,d_K] = WX[d1,\dots,d_K] + b $


When processing high-dimensional data, fully connected layers have too many parameters. Moreover, fully connected layers assume a complex nonlinear relationship between inputs and outputs, overlooking simpler structural regularities[^13]. Yet high-dimensional signals generally do have such strong structure — images, for instance, exhibit both short-term correlations and statistical invariance to translation, scaling, and symmetry. By contrast, convolutional layers capture the spatial structure of signals much better — because convolutional weights are shared across different parts of the input signal, they can learn certain local spatial structures, such as the edges, shapes, and corners of images. Stacked convolutions are a common dimensionality-reduction tool for high-dimensional signals (images, sound).

A 1D convolution takes a $D \times T$ tensor $X$ as input, applies an affine transformation $\phi(\cdot;w): \mathbb{R}^{D\times K} \rightarrow \mathbb{R}^{D'\times 1}$ to each $D\times K$ sub-tensor, and stores the resulting $D'\times 1$ tensors into $Y$ in order.

A 1D transposed convolution takes a $D \times T$ tensor $X$ as input, applies an affine transformation $\phi(\cdot;w): \mathbb{R}^{D\times 1} \rightarrow \mathbb{R}^{D'\times K}$ to each $D\times 1$ sub-tensor, and sums the results into a $D'\times K$ tensor stored in $Y$.

In the figure below, $D=3, K=5, D'=4$.

![1dconv](https://jipeng4974.github.io/img/1dconv.png)

![2dconv](https://jipeng4974.github.io/img/2dconv.png)

1D convolutions are often used to process sequence data or time-series data. 2D convolutions are commonly used for images, or other tasks that take 2D matrices as input. Transposed convolutions / deconvolutions are mainly used in generative models such as GANs and VAEs, expanding a low-dimensional feature into a high-resolution image.

### Activation Functions
If a model used only linear components, the whole would still be a linear operation, so nonlinearity must be introduced, and this nonlinearity is usually provided by activation functions. The most commonly used activation function is ReLU [Glorot et al., 2011][^14]. Before ReLU, it was the hyperbolic tangent function Tanh.

Some other activation functions follow a similar idea to ReLU — keeping positive values unchanged while compressing negative ones — such as Leaky ReLU [Maas et al., 2013][^15] and GELU [Hendrycks and Gimpel, 2016][^16].

### Pooling
Pooling is a classic strategy for dimensionality reduction — shrinking the signal — by merging several neighboring values into one via max or average.

### Dropout
The Dropout [Srivastava et al., 2014][^17] layer has no trainable parameters, only a hyperparameter $p$. During training it randomly shuts down some neurons with probability $p$, preventing individual neurons from dominating the whole and forcing other neurons to take over with slightly different weights. Dropout is therefore a regularization tool that prevents overfitting during training. It is turned off during inference.

### Normalization Layers
Normalization can be used to combat vanishing gradients. The most important normalization layer is Batch Normalization [Ioffe and Szegedy, 2015][^18], which consists of a hyperparameter $D$ and trainable parameters $\beta_1, \dots,\beta_D$ and $\gamma_1, \dots,\gamma_D$. Given a batch of $D$-dimensional samples $x_1, \dots, x_B$, first compute the mean $m_d = \frac{1}{B} \sum_{b=1}^B x_{b,d}$ and variance $v_d  = \frac{1}{B} \sum_{b=1}^B (x_{b,d} - m_d)^2$ for each dimension.

Then, for each b, compute the normalized value $z_{b,d} = \frac{x_{b,d} - m_d}{\sqrt{v_d + \epsilon}} $ with mean 0 and variance 1, and then the final result $y_{b,d} = \gamma_d z_{b,d} + \beta_d$, which has mean $\beta_d$ and variance $\gamma_d$.

![norm](https://jipeng4974.github.io/img/norm.png)

### Residual Connections
Skip connections (see [Long et al., 2014][^19]; [Ronneberger et al., 2015][^20]) likewise combat vanishing gradients. A skip connection is not actually a layer, but a design in which the output of some layer skips over a few intermediate layers and is grafted onto later ones. This design allows earlier, rawer signals to be "revisited" in later layers.

The practical implementation of skip connections is residual connections, which directly sum the two signals and skip only a modest number of layers. This design allows signals to survive passage through layers where gradients would otherwise vanish. Building on residual connections, Kaiming He created ResNet [He et al., 2015][^21], and Google designed the Transformer [Vaswani et al., 2017][^22].

### Attention Layers
Existing components lack the ability to combine local information with information at distant positions in a tensor; the attention layer specializes in exactly this — by computing an attention score between every component of the resulting tensor and every component of the input tensor, it averages features across the entire tensor, free from locality constraints[^22].

Given a queries tensor $Q$ of dimension $N^Q\times D^{QK}$, a keys tensor $K$ of dimension $N^{KV}\times D^{QK}$, and a values tensor $V$ of dimension $N^{KV}\times D^V$, the ```attention operation``` $att(K,Q,V)$ computes a tensor Y of dimension $N^Q\times D^V$:

$$Y = att(K,Q,V) = \underbrace{softargmax(\frac{QK^T}{\frac{1}{\sqrt{D^{QK}}}})}_A V$$

The whole process has two steps. The first step computes the attention score between every query index $q$ and every key index $k$, i.e. the ```softargmax``` of the dot products of queries and keys: $A_{q,k} = \frac{exp(\frac{1}{\sqrt{D^{QK}}} Q_q \cdot K_k ) }{\sum_l exp(\frac{1}{\sqrt{D^{QK}}} Q_q \cdot K_l)}$, where $\frac{1}{\sqrt{D^{QK}}}$ is a scaling parameter that keeps the range of values roughly unchanged as $D^{QK}$ grows.

![att](https://jipeng4974.github.io/img/attention.png)

With the attention scores $A_{q,k}$ in hand, the second step computes: $Y_q = \sum_k A_{q,k}V_k$. An attention score is the degree of match between a query and a key — the better the match, the higher the weight. If a query matches one key almost perfectly, the attention score approaches 1 and the value corresponding to that key is taken directly. If it matches several keys to a moderate degree, the result is a weighted average according to the attention scores.

## Other Topics
```LBDL``` also discusses various deep learning model architectures and applications, such as multi-layer perceptrons, convolutional networks, attention models, RNNs, autoencoders, GANs, graph neural networks, GPT, and diffusion.


[^1]: R. Sennrich, B. Haddow, and A. Birch. Neural Machine Translation of Rare Words with Subword Units. CoRR, abs/1508.07909, 2015. [[pdf]](https://arxiv.org/pdf/1508.07909).
[^2]: D. Kingma and J. Ba. Adam: A Method for Stochastic Optimization. CoRR, abs/1412.6980, 2014. [[pdf]](https://arxiv.org/pdf/1412.6980).
[^3]: $\nabla\ell_{|x^(d-1)}$ is the gradient of the loss function with respect to $x^{d-1}$, the variable of $f^{d-1}$.
[^4]: $\nabla\ell_{|x^(d-1)}$ is the gradient of the loss function with respect to $x^{d}$, the variable of $f^{d}$.
[^5]: $J_{f^{(d)}|x}$ is the Jacobian of the d-th layer function $f^{(d)}$ with respect to the variable x — the Jacobian matrix, i.e. the matrix formed by arranging the function's first-order partial derivatives in a certain way.
[^6]: $J_{f^{(d)}|w}$ is the Jacobian of the d-th layer function $f^{(d)}$ with respect to the weights w.
[^7]: A. Baydin, B. Pearlmutter, A. Radul, and J. Siskind. Automatic differentiation in machine learning: a survey. CoRR, abs/1502.05767, 2015. [[pdf]](https://arxiv.org/pdf/1502.05767).
[^8]: A. Gomez, M. Ren, R. Urtasun, and R. Grosse. The Reversible Residual Network: Backpropagation Without Storing Activations. CoRR, abs/1707.04585, 2017. [[pdf]](https://arxiv.org/pdf/1707.04585).
[^9]: T. Chen, B. Xu, C. Zhang, and C. Guestrin. Training Deep Nets with Sublinear Memory Cost. CoRR, abs/1604.06174, 2016. [pdf](https://arxiv.org/pdf/1604.06174).
[^10]: R. Pascanu, T. Mikolov, and Y. Bengio. On the difficulty of training recurrent neural networks. In International Conference on Machine Learning (ICML), 2013. [pdf](https://proceedings.mlr.press/v28/pascanu13.pdf).
[^11]: X. Glorot and Y. Bengio. Understanding the difficulty of training deep feedforward neural networks. In International Conference on Artificial Intelligence and Statistics (AISTATS), 2010. [pdf](https://proceedings.mlr.press/v9/glorot10a/glorot10a.pdf).
[^12]: A. Radford, J. Wu, R. Child, et al. Language Models are Unsupervised Multitask Learners, 2019. [[pdf]](https://d4mucfpksywv.cloudfront.net/better-language-models/language_models_are_unsupervised_multitask_learners.pdf).
[^13]: This is the so-called ```inductive bias``` of fully connected layers.
[^14]: X. Glorot, A. Bordes, and Y. Bengio. Deep Sparse Rectifier Neural Networks. In International Conference on Artificial Intelligence and Statistics (AISTATS), 2011. [[pdf]](http://proceedings.mlr.press/v15/glorot11a/glorot11a.pdf).
[^15]: A. L. Maas, A. Y. Hannun, and A. Y. Ng. Rectifier nonlinearities improve neural network acoustic models. In proceedings of the ICML Workshop on Deep Learning for Audio, Speech and Language Processing, 2013. [[pdf]](https://ai.stanford.edu/~amaas/papers/relu_hybrid_icml2013_final.pdf).
[^16]: D. Hendrycks and K. Gimpel. Gaussian Error Linear Units (GELUs). CoRR, abs/1606.08415, 2016. [[pdf]](https://arxiv.org/pdf/1606.08415). 
[^17]: N. Srivastava, G. Hinton, A. Krizhevsky, et al. Dropout: A Simple Way to Prevent Neural Networks from Overfitting. Journal of Machine Learning Research (JMLR), 15:1929–1958, 2014. [[pdf]](https://jmlr.org/papers/volume15/srivastava14a/srivastava14a.pdf).
[^18]: S. Ioffe and C. Szegedy. Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift. In International Conference on Machine Learning (ICML), 2015. [[pdf]](http://static.googleusercontent.com/media/research.google.com/en//pubs/archive/43442.pdf). 
[^19]: J. Long, E. Shelhamer, and T. Darrell. Fully Convolutional Networks for Semantic Segmentation. CoRR, abs/1411.4038, 2014. [[pdf]](https://arxiv.org/pdf/1411.4038). 
[^20]: O. Ronneberger, P. Fischer, and T. Brox. U-Net: Convolutional Networks for Biomedical Image Segmentation. In Medical Image Computing and Computer-Assisted Intervention, 2015. [[pdf]](https://arxiv.org/pdf/1505.04597.pdf).
[^21]: K. He, X. Zhang, S. Ren, and J. Sun. Deep Residual Learning for Image Recognition. CoRR, abs/1512.03385, 2015. [[pdf]](https://arxiv.org/pdf/1512.03385).
[^22]: A. Vaswani, N. Shazeer, N. Parmar, et al. Attention Is All You Need. CoRR, abs/1706.03762, 2017. [[pdf]](https://arxiv.org/pdf/1706.03762)
