---
title: 'Review on Transferring Robustness for Graph Neural Network Against Poisoning Attacks'
data: 2020-01-21
permalink: /posts/2020/01/blog-post-1/
tags:
  - Adversarial Machine Learning
  - Graph Neural Network
---


**Summary**: The author proposed a method to train a robust graph neural network from clean graphs, which are similar to the target poisoned graph, with intentially added perturbation. *Penalized Aggregation Mechanism* is introduced to distinguish the adversarial edges from the benign edges. And *Meta-Optimization* is used to transfer this learned ability to the model applied on target poisoned graph.

**Intuition**: In order to train a model which can reduce the negative effects of adversarial edges, one idea is to restrict the message-passing through perturbed edges. But in a poisoned graph, we have no knowledge of which edge is adversarial. To handle this problem one can learn from similar clear graphs, on which adversarial edges are added. The model is learned to assign low self-attention coefficients for the adversarial edges.

**Methodology**: 

- clean graphs are attacked by adding adversarial edges (metattack), which serve as known perturbations
- a penalized aggregation mechanism is then designed to learn the ability of alleviating negative influences from perturbations
- transfer this ability to the target poisoned graph with a special meta-optimization approach

## Penalized Aggregation Mechanism
Restrict the message-passing through perturbed edges according to the self-attension coefficient, which is calculated from node features and learnable parameters.

$$a_{ij}^l = LeakyReLU((a^l)^T[W^lh_i^l\bigoplus W^lh_j^l])$$

where the $\bigoplus$ indicates the concatenation of vectors. The self-attension coefficient is learned by maximize the difference between attention coefficients received from normal edges and adversarial edges.

$$\mathcal{L}_{dist} = -min(\eta, \underset{e_{ij}\in\mathcal{E}\setminus\mathcal{P}}{\mathbb{E}}a_{ij}^l - \underset{e_{ij}\in\mathcal{P}}{\mathbb{E}}a_{ij}^l)$$

## Transfer with Meta-Optimization
The goal of meta-learning is to learn a good initialization of parameters, so that a small amount or even no supervision knowledge algorithm is needed.

The learning data are M clean graphs, the loss function is defined by above section. The labeled nodes are split into support sets and query sets. For M learning tasks on M clean graphs, M models are initialized separately by $\theta$, and their parameters are updated separately by support sets, 

$$\theta_i'=\theta-\alpha\nabla_{\theta}\mathcal{L}_{\mathcal{T}_i}(\theta)$$

then we get ${\theta_0', \theta_1', ..., \theta_{M-1}'}$. Although only one step update is shown in above equation but multiple gradient updates can also be applied. The model parameters $\theta$ are then updated cross tasks. Following is the objective function for the meta-optimization:

$$\min_{\theta} \sum_{i=1}^{M}\mathcal{L}_{\mathcal{T}_i}(\theta_i')=\min_{\theta}\sum_{i=1}^{M}\mathcal{L}_{\mathcal{T}_i}(\theta-\alpha\nabla_{\theta}\mathcal{L}_{\mathcal{T}_i}(\theta))$$

The objective function is optimized with gradient descent:

$$\theta \leftarrow \theta - \beta\nabla_{\theta}\sum_{i=1}^{M}\mathcal{L}_{\mathcal{T}_i}(\theta_i')$$

