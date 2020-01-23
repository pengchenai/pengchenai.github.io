---
title: 'Review on Hierarchical Graph Representation Learning with Differentiable Pooling'
data: 2020-01-21
permalink: /posts/2020/01/blog-post-3/
tags:
  - Graph Neural Network
  - Graph Classification
---

**Summary**: This paper proposed a graph neural network (GNN) model, which can learn hierarchical representations of graphs, for graph classification. As far as I know, it's the hitherto only GNN which has a differentiable graph pooling strategy that generalize across graphs with different number of nodes. This pooling module is called DiffPool.

## Methodology

Each layer of the proposed model contains two parts: one part is a GNN for the node embedding and the other part, i.e. DiffPool, for the hierarchical pooling. 

### Node Embedding

The node embedding learned by a GNN at layer $l$ is denoted as $Z^{(l)}$:

$$Z^{(l)}=GNN_{l, embed}(A^{(l)}, X^{l})$$ 

### Pooling

The basic idea of pooling in graph is to build coarsened graph containing fewer nodes, which is realised by nodes clustering. Nodes in the same cluster are aggregated into one node, and all these nodes form a coarsened graph, which are then passed to next layer. 

In order to utilize backpropagation, the clustering need to be differentiable. Therefore, another graph neural network is used to learn cluster assignments for each node. The input of DiffPool at layer $l$ is the node features from the layer l-1. The output of DiffPool at layer $l$ is an assignment matrix $S^{(l)}\in \mathbb{R}^{n_l\times n_{l+1}}$:

$$S^{(l)} = softmax(GNN_{l, pool}(A^{(l)}, X^{l}))$$

Note that all nodes at the final layer L are assigned to a single cluster.

The node features and adjacency matrix in the coarsened graph is updated accordingly by following equations:

$$X^{(l+1)}={S^{(l)}}^T Z^{(l)}   \in \mathbb{R}^{n_{l+1}\times d}$$

$$A^{(l+1)} = {S^{(l)}}^TA^{(l)}S^{(l)}\in \mathbb{R}^{n_{l+1}\times n_{l+1}}$$

## Learning

It's mentioned in the paper the model is hard to train with gradient signal alone since it's a non-convex optimization problem. Therefore, another two regularization are added:

### Link prediction objective

We want nearby nodes be pooled together. The loss is formalized in the paper as:

$$L_{LP}=||A^{(l)}, S^{(l)}{S^{(l)}}^T||_F$$

In the code provided by the author it is:
$$L_{LP}=H(A, S^{(l)}{S^{(l)}}^T) = -E_{p}[logq]$$

### Entropy loss

We want the assignment for each node be close to a one-hot vector. The corresponding loss is:

$$L_E=\frac{1}{n}\sum_{i=1}^nH(S_i)$$

Besides the author also found that adding an $l_2$ normalization to the node embeddings at each layer made the training more stable.









