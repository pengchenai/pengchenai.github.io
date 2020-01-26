---
title: 'Review^2 on Relational Machine Learning for Knowledge Graphs'
data: 2020-01-22
permalink: /posts/2020/01/blog-post-2/
tags:
  - Knowledge Graph
---


a topic of relational machine learning studies: train statistical models on large knowledge graphs, and then used to predict new facts.

"two" models:
- latent feature models
	- tensor factorization
	- multiway neural networks
- mining observable patterns in the graph
- combined

extract text-based information from the web and construct knowledge graph

Statistical Relational Learning (SRL):

representation of an object can obtain its relationships to other objects. The data is in the form of a graph. 
Main topics of SRL:
- prediction of missing edges
- prediction of properties of nodes
- clustering nodes based on their connectivity patterns

scalable SRL techniques: take time that is (at moost) linear in the size of the graph

# Introduction of Knowledge Graph

A collection of facts
type hierarchies
type constraints

paradigms for the interpretation of non-existing triples:
- closed world assumption
- open world assumption

Main tasks in knowledge graph construction and curation:
- link prediction
- entity resolution: identifying which objects in relational data refer to the same underlying entities.
- link-based clustering

deterministic rules:
- type constraints
- transitivity

"softer" statistical patterns:
- homophily: tendency of entities to be related to other entities with similar characteristics
- block structure: entities can be divided into distinct groups, such that all the member of a group have similar relationships to members of a group have similar relationships to members of other groups
- global and long-range statistical dependencies


# SRL and its application on Knowledge Graph
The correlation of the presence of triples are modeled by:
- M1: All $y_{ijk}$ are conditionally independent given latent features associated with subject, object and relation type and additional parameters. (latent feature models)
- M2: All $y_{ijk}$ are conditionally independent given observed graph features and additional parameters. (graph feature models)
- M3: all $y_{ijk}$ have local interactions (Markov Random Fields)

The model classes M1 and M2 predict the existence of a triple $x_{ijk}$ via a score function $f(x_{ijk};\Theta)$. The conditional independence assumptions of M1 and M2 allow the probability model to be written as:

$$P(\underline{\textbf{Y}}|\mathcal{D}, \Theta)=\prod_{i=1}^{N_e}\prod_{j=1}^{N_e}\prod_{k=1}^{N_r}Ber(y_{ijk}|\sigma(f(x_{ijk};\Theta)))$$

## M1: latent feature models
Every entity is mathematically denoted as a vector of latent featrues. To illustrate this idea, let's have a look at a triple *(Alex Guinness, receive, Academy Award)*, indicating that Alex Guinness received the Academy Award. The Alex Guinness and Academy Award can be represented as 
$$\textbf{e}_{Guinness}=[0.9,0.2]^T, \textbf{e}_{AcademyAward}=[0.2,0.8]^T$$
, where the component $e_{i1}$ correspond to the latent feature Good Actor and $e_{i2}$ correspond to Prestigious Award. Note that the latent features are usually hard to interpret.

The latent features itself is not enough, we also need the interaction between them. The interaction can be modeled in many possible ways:

### RESCAL
The interaction is mathamatically formulated as
$$f_{ijk}^{RESCAL}:=\textbf{e}_i^T\textbf{W}_k\textbf{e}_j$$

The representation of entities are shared no matter they are subjects or objects in a relationship, it captures the "similarity of entities in the relational domain", which means entities connected to similar entities via similar relations. Therefore, similar entities have similar representation. This properties can be further used for entities resolution and as a preprocess for non-relational machine learning algorithms. 

## M2: Correlttion between the nodes/edges using observable properties
## Combination of these two approaches
## M3: Markov Random Fields

# Training models on Knowledge Graphs
# Relational learning using Markov Random Fields
# automated knowledge base construction