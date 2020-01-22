---
title: 'Review on Hierarchical Graph Representation Learning with Differentiable Pooling'
data: 2020-01-21
permalink: /posts/2020/01/blog-post-3/
tags:
  - Graph Neural Network
  - Graph Classification
---

**Summary**: This paper proposed a graph neural network (GNN) model, which can learn hierarchical representations of graphs, for graph classification. As far as I know, it's the hitherto only GNN which has a differentiable graph pooling strategy that generalize across graphs with different number of nodes. This pooling module is called DiffPool.

Each layer of the proposed model contains two parts: one part is a GNN for the node embedding and the other part, i.e. DiffPool, for the hierarchical pooling. The node embedding learned at layer $l$ is denoted as $Z^{(l)}$. Pooling is realized by clustering nodes, which in the same cluster are aggregated into one node. All these nodes form a coarsened graph, which are then passed to next layer. 

In order to utilize backpropagation, the clustering need to be differentiable. Therefore, another graph neural network is used to learn cluster assignments for each node. The input of DiffPool at layer $l$ is the node features from the layer l-1. The output of DiffPool at layer $l$ is an assignment matrix $S^{(l)}\in \mathbb{R}^{n_l\times n_{l+1}}$. Then the graph with learned node embedding $Z^{(l))$ is coarsened by following equations:

$$X^{(l+1)}=S^{(l)}^TZ^{(l)} \in \mathbb{R}^{n_{l+1}\times d},$$

$$A^{(l+1)} = S^{(l)}^TA^{(l)}S^{(l)}\in \mathbb{R}^{n_{l+1}\times n_{l+1}}$$






