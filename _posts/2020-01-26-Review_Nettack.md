---
title: 'Review on Adversarial Attacks on Neural Networks for Graph Data'
data: 2020-01-26
permalink: /posts/2020/01/blog-post-4/
tags:
  - Adversarial Machine Learning
  - Graph Neural Network
---

**Summary**: This paper is one of the first works on the field of the adversarial attack on graph neural network. It targets at semi-supervised task of node classification in the training phase and can add structure perturbations and feature perturbations on the attributed graph. The threat model proposed in this paper, i.e. local budget and global budget, is widely accepted by other works in this field. A major challenge for adversarial attack on graph lies on the fact that the graph structure is intrinsincly discrete and thus gradient based approaches, which are widely used in attacks on images, are not suited.


### Problem formulation

The adversarial attack on graph can be formally represented as:

$$\underset{(A',X')\in \mathcal{\hat{P}}^{G_0}_{\Delta, \mathcal{A}}}{argmax} \underset{c\neq c_{old}}{max}\, ln Z^*_{v_0,c}-ln Z^*_{v_0, c_{old}}$$

$$\text{subject to }Z^* = f_{\theta^*}(A',X'), \text{with}\, \theta^*=\underset{\theta}{argmin}\,L(\theta;A',X')$$

where $\mathcal{\hat{P}}^{G_0}_{\Delta, \mathcal{A}}$ is the set of perturbed graphs under the global budget and perserving additionally the degree distribution and feature co-occurence. More detail on next part.

### Threat Model
TBD

### Surrogate model

A surrogate model, which is a two-layer-GCN without activation, is firstly attacked. 

$$Z'=\text{softmax}(\hat{A}\hat{A}XW^{(1)}W{(2)})=\text{softmax}(\hat{A}^2XW)$$

There are three reasons for using surrogate model:

- This work focuses on the poisoning attack, which means perturbations are added before the graph neural network is trained.
- Using surrogate model can reduce the time needed to calculate candidates for structure attack. (refer to section 5 in paper for more details)
- It brings in the wanted transferability of adversarial examples.


