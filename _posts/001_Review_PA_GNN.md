---
title: 'Review on Transferring Robustness for Graph Neural Network Against Poisoning Attacks'
data: 2020-01-21
permalink: /posts/2012/08/blog-post-1/
tags:
  - Adversarial Machine Learning
  - Graph Neural Network
---


Goal: design a robust graph neural network against poisoning attack. The learning process of this model utilizes poisoned graph and some similar clean graphs, so that the it can discriminate adversarial edges from the normal ones.

Methodology: The authors propose the PA-GNN, which relies on a penalized aggregation mechanism that directly restrict the negative impact of adversarial edges by assigning them lower attention coefficients. And this ability is transfered when apply the model on the target poisoned graph via meta-optimization.

- clean graphs are attacked by adding adversarial edges, which serve as known perturbations
- a penalized aggregation mechanism is then designed to learn the ability of alleviating negative influences from perturbations
- transfer this ability to the target poisoned graph with a special meta-optimization approach

## Penalized Aggregation Mechanism
restrict the message-passing through perturbed edge according to the self-attension coefficient, which is calculated according to the node features, the self-attension coefficient is learned by maximize the difference between attention coefficients received from normal edges and adversarial edges.

Since we have no knowledge of which edge is adversarial in poisoned graph, we need to learn from similar clear graphs, on which adversarial edges are added.

## Transfer with Meta-Optimization
The goal of meta-learning is to learn a good initialization of parameters, so that a small amount or even no supervision knowledge algorithm is needed.

The learning data are M clean graphs, the loss function is defined by above section. The labeled nodes are split into support sets and query sets. For M learning tasks on M clean graphs, M models are initialized separately by $\theta$, and their parameters are updated separately by support sets, then we get ${\theta_0, \theta_1, ..., \theta_{M-1}}$. Then the $\theta$ is updated cross tasks, i.e.
\begin{equation}
\min_{\theta} \sum_{i=1}^{M}\mathcal{L}_{\mathcal{T}_i}(\theta_i')=\min_{\theta}\sum_{i=1}^{M}\mathcal{L}_{\mathcal{T}_i}(\theta-\alpha\nabla_{\theta}\mathcal{L}_{\mathcal{T}_i}(\theta))
\end{equation} 

clean graphs are perturbated by metattack