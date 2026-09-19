# Efficient ANNS at Scale

> How to perform efficient vector search over feature stores with billions or tens of billions of vectors

---

LLMS index: [llms.txt](/llms.txt)

---

Vector search falls into two categories: KNN and ANN. KNN brute-forces the similarity between the query vector and every indexed vector and returns the top k. ANN accelerates the computation through approximation, trading accuracy for efficiency.

How do we do efficient vector search? There are two approaches. The first is quantization that improves the encoding rate (which not only reduces memory footprint but also speeds up computation, because it is SIMD-friendly and reduces memory access): IVF vector quantization, PQ quantization, ScaNN anisotropic quantization, and 4-bit quantization. The second is graph search (HNSW), which shrinks the search space.

How do we do vector search over feature stores with billions or tens of billions of vectors? The main challenge is that the index cannot fit into memory, so we need a hybrid memory-disk scheme (SPANN).

This post only covers a small set of effective and orthogonal techniques; combined, they form the current best practice. For other older, classic techniques, see the background surveys in the papers cited here.

# Similarity
First, let's review what **similarity between vectors** means:
- For vectors X and Y, the closer x and y are, the more y is large where x is large and small where x is small, and the larger the inner product becomes. So the inner product can serve as a similarity score, though it is not robust to scale changes.
- To be robust to scale changes, the inner product is often normalized to [-1,1], which gives the cosine similarity cosθ, where θ is the angle between the vectors. Obviously, the smaller θ is, the closer the vectors are.
- After normalizing Euclidean distance to [-1,1], it is equivalent to sqrt(2-2cosθ). So squared Euclidean distance is also proportional to cosine similarity.

![sim_measure](https://wujipeng.com/img/sim_measure.png)

After normalization, Euclidean distance, inner product, and cosθ all share the same origin, so we generally use cosine similarity cosθ to measure the similarity between vectors/embeddings. Unless there is a special need to resist positional shifts, in which case the Pearson correlation coefficient can be used: it is the normalization of covariance (the joint error of two variables), but it can also be viewed as the cosine similarity between centered x and y.

# Vector Quantization
A quantizer is a mapping function q from a D-dimensional vector x ∈ R^D to q(x) ∈ C = {c_0, c_1, ... c_k-1}. Each c_i is a cluster centroid.
The so-called codebook is just a lookup table that uses the centroid index as the low-bit representation of the original vector; the codebook size is k.

Vector quantization is essentially lossy compression. Representation learning is also essentially lossy compression — one is a white box, the other a black box — both trying to reduce the encoding rate of high-dimensional data.

# IVF: Clustering, Inverted Index, Pruning
Inverted indexing (specifically IVF) is an old quantization technique, applied early on in [Video Google](https://www.robots.ox.ac.uk/~vgg/publications/2003/Sivic03/sivic03.pdf). Its core idea is vector quantization based on k-means clustering. K-means clustering originates from signal processing; its goal is to partition n vectors into k clusters, where each vector belongs to the cluster with the nearest centroid.

After clustering, some boundary data points belong to both A and B, so they are placed into the posting lists of both A's and B's centroids. But duplicate placement causes posting list bloat; to deal with this bloat, a pruning strategy can be adopted — see Microsoft's SPTAG project and the [SPANN paper](https://arxiv.org/pdf/2111.08566.pdf).

Faiss's IVFFlat provides retrieval on top of the inverted index formed by clustering, with centroids as terms. nlist is the number of clusters, nprobe is the number of nearby centroids probed, and when nprobe=nlist it degenerates into brute-force search. At query time, IVFFlat brute-force searches all original vectors under the nprobe centroids nearest to the query. Since a centroid's representation can be simplified to a cluster id, IVF is actually also a form of quantization: store all centroid vectors in one array, and the array index itself represents the centroid.

# PQ: Product Quantization
Clustering-based quantization works well for low-dimensional vectors, but as vector dimensionality increases, the error grows too — and this error cannot be fixed by simply increasing the number of clusters. If the number of clusters (codebook size) were raised to 2^64, the cost of training that clustering would be unacceptably high, requiring several times 2^64 samples, and it clearly wouldn't fit in memory either. **Product Quantization (PQ)** is the technique that solves IVF's quantization error through dimensionality reduction; both [Jegou et al., 2011](https://lear.inrialpes.fr/pubs/2011/JDS11/jegou_searching_with_quantization.pdf) and [ScaNN](https://arxiv.org/pdf/1908.10396.pdf) discuss PQ. Its core idea is to quantize the indexed data by splitting the d-dimensional space into M subspaces, approximating the error of a high-dimensional vector as the sum of the errors of its segmented low-dimensional vectors.

In product quantization, what the IVF posting list stores is not the original vector, but an encoding of it:
- Compute the residual between the original d-dimensional vector and its high-dimensional coarse-cluster centroid (the purpose of the residual is centering — making the data distribution more concentrated, which improves precision)
- Split the residual vector into M segments, each of dimension d/M
- Run $k=2^n$ fine clusters on each segment, where n is the code length of the low-dimensional fine-cluster id — 4bit or 8bit
- Quantize each original residual segment with its nearest low-dimensional fine-cluster centroid

Faiss's IVFPQ combines IVF and PQ: IVF coarse clustering is the first-level quantizer, and PQ is the second-level quantizer. On one hand the index shrinks; on the other hand, computation only requires distances between residuals and segment-wise fine-cluster centroids, which can also be SIMD-accelerated — a very significant performance win. However, the IVFPQ distance is ultimately just an approximation: the distance to the coarse-cluster centroid plus the sum of distances between the query residual and the nearby fine-cluster centroids across the M segments. This approximation helps find neighbors to some extent but is unsuitable as an actual distance. You can treat IVFPQ as a coarse ranking stage, then rerun brute-force or high-precision ANN on the results to get more accurate distances.

ScaNN proposed anisotropic quantization, which set things right at the source: it corrected the mathematical error of the previous nearest-centroid selection, and it also demonstrated that 4-bit quantization works very well in practice (in fact, 4-bit quantization contributes more to the gains than anisotropy, but anisotropy is the bigger theoretical contribution). Traditional quantization quantizes a data point to its nearest cluster center, but that is not necessarily the lowest-error choice: the more parallel two vectors are, the larger their inner product; the more orthogonal, the smaller. So assigning a higher error penalty weight to parallel directions works better. This means we no longer necessarily quantize to the nearest centroid — the goal is, while keeping the overall encoding rate unchanged, to make the quantization error smaller in parallel directions and tolerate higher error in orthogonal directions, thereby improving overall ANN performance.

Suppose d/M = 4 with 8-bit quantization, and the original vector is a 256-dimensional f32 vector. Then PQ compresses 256 float32 values into 64 int8 values, reducing memory to 1/16. Product quantization obviously reduces memory and storage cost significantly, but it also speeds up computation, for the following reasons:
- Efficient dot products (also trading accuracy for efficiency): O(dk+mn) is faster than O(nd). For the dot product of a d-dimensional query with n quantized vectors, introducing m codebooks of size k, mn is negligible and k is much smaller than n.
- Memory bandwidth: modern processors need "high compute per memory read" to fully exploit their computational performance. Once data points are compressed by quantization, memory bandwidth usage drops as well, making the workload more compute-intensive.
- The low-bit representations introduced by vector quantization — especially 4-bit quantization — enjoy the same SIMD and AMX performance dividends as low-bit floating-point quantization.

# Best Practice: Correctly Combining Orthogonal Techniques for Your Scenario
HNSW, IVF, PQ, anisotropic score-aware quantization loss, and brute-force are in fact all orthogonal techniques that can be combined.

For example, data points that have undergone PQ quantization can have an HNSW built on top of them; using a graph method for queries on ultra-large-scale datasets is clearly more efficient than IVFPQ. Anisotropic quantization, in turn, is a correction to the earlier PQ quantization. And given the distance distortion introduced by quantization, you can also use brute-force to recompute the coarse-ranked results from ivf+pq+hnsw to get the most accurate distances. Since the candidates have already gone through one round of coarse ranking — shrinking from billions down to tens of thousands — the brute-force cost is entirely acceptable.
