# Graph Learning

Generally sorted by date.

***

- **Modeling Relational Data with Graph Convolutional Networks**
  - Author: Michael Schlichtkrull, Thomas N. Kipf, Peter Bloem, Rianne van den Berg, Ivan Titov, Max Welling
  - Publication: ESWC 2018
  - Date: 17 Mar 2017
  - Link: <https://arxiv.org/abs/1703.06103>
  - Abstract: Knowledge graphs enable a wide variety of applications, including question answering and information retrieval. Despite the great effort invested in their creation and maintenance, even the largest (e.g., Yago, DBPedia or Wikidata) remain incomplete. We introduce Relational Graph Convolutional Networks (R-GCNs) and apply them to two standard knowledge base completion tasks: Link prediction (recovery of missing facts, i.e. subject-predicate-object triples) and entity classification (recovery of missing entity attributes). R-GCNs are related to a recent class of neural networks operating on graphs, and are developed specifically to deal with the highly multi-relational data characteristic of realistic knowledge bases. We demonstrate the effectiveness of R-GCNs as a stand-alone model for entity classification. We further show that factorization models for link prediction such as DistMult can be significantly improved by enriching them with an encoder model to accumulate evidence over multiple inference steps in the relational graph, demonstrating a large improvement of 29.8% on FB15k-237 over a decoder-only baseline.
  - [Paper With Code](https://paperswithcode.com/paper/modeling-relational-data-with-graph)
    - [Graph Convolutional Networks for relational graphs](https://github.com/masakicktashiro/rgcn_pytorch_implementation)
    - [Torch-RGCN](https://github.com/thiviyanT/torch-rgcn)
    - [Relational Graph Convolutional Networks (RGCN) Pytorch implementation](https://github.com/berlincho/RGCN-pytorch)
  - Note
    - Heterogeneous graph neural network
    - R-GCN

- **Heterogeneous graph attention network**
  - Author: Xiao Wang, Houye Ji, Chuan Shi, Bai Wang, Peng Cui, P. Yu, Yanfang Ye
  - Publication: WWW 2019
  - Date: 18 Mar 2019
  - Link: <https://arxiv.org/abs/1903.07293>
  - Abstract: Graph neural network, as a powerful graph representation technique based on deep learning, has shown superior performance and attracted considerable research interest. However, it has not been fully considered in graph neural network for heterogeneous graph which contains different types of nodes and links. The heterogeneity and rich semantic information bring great challenges for designing a graph neural network for heterogeneous graph. Recently, one of the most exciting advancements in deep learning is the attention mechanism, whose great potential has been well demonstrated in various areas. In this paper, we first propose a novel heterogeneous graph neural network based on the hierarchical attention, including node-level and semantic-level attentions. Specifically, the node-level attention aims to learn the importance between a node and its metapath based neighbors, while the semantic-level attention is able to learn the importance of different meta-paths. With the learned importance from both node-level and semantic-level attention, the importance of node and meta-path can be fully considered. Then the proposed model can generate node embedding by aggregating features from meta-path based neighbors in a hierarchical manner. Extensive experimental results on three real-world heterogeneous graphs not only show the superior performance of our proposed model over the state-of-the-arts, but also demonstrate its potentially good interpretability for graph analysis.
  - [HAN详解（Heterogeneous graph attention network）](https://zhuanlan.zhihu.com/p/346658317)

- **Heterogeneous Graph Neural Network**
  - Author: Zhang, Chuxu and Song, Dongjin and Huang, Chao and Swami, Ananthram and Chawla, Nitesh V.
  - Publication: SIGKDD 2019
  - Date: 25 July 2019
  - Link: <https://dl.acm.org/doi/abs/10.1145/3292500.3330961>
  - Abstract: Representation learning in heterogeneous graphs aims to pursue a meaningful vector representation for each node so as to facilitate downstream applications such as link prediction, personalized recommendation, node classification, etc. This task, however, is challenging not only because of the demand to incorporate heterogeneous structural (graph) information consisting of multiple types of nodes and edges, but also due to the need for considering heterogeneous attributes or contents (e.g., text or image) associated with each node. Despite a substantial amount of effort has been made to homogeneous (or heterogeneous) graph embedding, attributed graph embedding as well as graph neural networks, few of them can jointly consider heterogeneous structural (graph) information as well as heterogeneous contents information of each node effectively. In this paper, we propose HetGNN, a heterogeneous graph neural network model, to resolve this issue. Specifically, we first introduce a random walk with restart strategy to sample a fixed size of strongly correlated heterogeneous neighbors for each node and group them based upon node types. Next, we design a neural network architecture with two modules to aggregate feature information of those sampled neighboring nodes. The first module encodes "deep" feature interactions of heterogeneous contents and generates content embedding for each node. The second module aggregates content (attribute) embeddings of different neighboring groups (types) and further combines them by considering the impacts of different groups to obtain the ultimate node embedding. Finally, we leverage a graph context loss and a mini-batch gradient descent procedure to train the model in an end-to-end manner. Extensive experiments on several datasets demonstrate that HetGNN can outperform state-of-the-art baselines in various graph mining tasks, i.e., link prediction, recommendation, node classification & clustering and inductive node classification & clustering.

- **Graph transformer networks**
  - Author: Seongjun Yun, Minbyul Jeong, Raehyun Kim, Jaewoo Kang, and Hyunwoo J Kim
  - Publication: NeurIPS’19
  - Date: 6 Nov 2019
  - Link: <https://arxiv.org/abs/1911.06455>
  - Abstract: Graph neural networks (GNNs) have been widely used in representation learning on graphs and achieved state-of-the-art performance in tasks such as node classification and link prediction. However, most existing GNNs are designed to learn node representations on the fixed and homogeneous graphs. The limitations especially become problematic when learning representations on a misspecified graph or a heterogeneous graph that consists of various types of nodes and edges. In this paper, we propose Graph Transformer Networks (GTNs) that are capable of generating new graph structures, which involve identifying useful connections between unconnected nodes on the original graph, while learning effective node representation on the new graphs in an end-to-end fashion. Graph Transformer layer, a core layer of GTNs, learns a soft selection of edge types and composite relations for generating useful multi-hop connections so-called meta-paths. Our experiments show that GTNs learn new graph structures, based on data and tasks without domain knowledge, and yield powerful node representation via convolution on the new graphs. Without domain-specific graph preprocessing, GTNs achieved the best performance in all three benchmark node classification tasks against the state-of-the-art methods that require pre-defined meta-paths from domain knowledge.
  - [Graph Transformer——合理灌水](https://zhuanlan.zhihu.com/p/365129455)
  - [Papers With Code](https://paperswithcode.com/paper/graph-transformer-networks-1)

- **Relation structure-aware heterogeneous graph neural network**
  - Author: Shichao Zhu, Chuan Zhou, [Shirui Pan](https://shiruipan.github.io/), Xingquan Zhu, Bin Wang
  - Publication: ICDM 2019
  - Date: 8 Nov 2019
  - Link: <https://ieeexplore.ieee.org/document/8970828>
  - Abstract: Heterogeneous graphs with different types of nodes and edges are ubiquitous and have immense value in many applications. Existing works on modeling heterogeneous graphs usually follow the idea of splitting a heterogeneous graph into multiple homogeneous subgraphs. This is ineffective in exploiting hidden rich semantic associations between different types of edges for large-scale multi-relational graphs. In this paper, we propose Relation Structure-Aware Heterogeneous Graph Neural Network (RSHN), a unified model that integrates graph and its coarsened line graph to embed both nodes and edges in heterogeneous graphs without requiring any prior knowledge such as metapath. To tackle the heterogeneity of edge connections, RSHN first creates a Coarsened Line Graph Neural Network (CL-GNN) to excavate edge-centric relation structural features that respect the latent associations of different types of edges based on coarsened line graph. After that, a Heterogeneous Graph Neural Network (H-GNN) is used to leverage implicit messages from neighbor nodes and edges propagating among nodes in heterogeneous graphs. As a result, different types of nodes and edges can enhance their embedding through mutual integration and promotion. Experiments and comparisons, based on semi-supervised classification tasks on large scale heterogeneous networks with over a hundred types of edges, show that RSHN significantly outperforms state-of-the-arts.
  - [Code](https://github.com/CheriseZhu/RSHN)
  - [Paper Page](https://shiruipan.github.io/publication/icdm-19-zhu/)

- **MAGNN: Metapath Aggregated Graph Neural Network for Heterogeneous Graph Embedding**
  - Author: Xinyu Fu, Jiani Zhang, Ziqiao Meng, Irwin King
  - Publication: WWW 2020
  - Date: 5 Feb 2020
  - Link: <https://arxiv.org/abs/2002.01680>
  - Abstract: A large number of real-world graphs or networks are inherently heterogeneous, involving a diversity of node types and relation types. Heterogeneous graph embedding is to embed rich structural and semantic information of a heterogeneous graph into low-dimensional node representations. Existing models usually define multiple metapaths in a heterogeneous graph to capture the composite relations and guide neighbor selection. However, these models either omit node content features, discard intermediate nodes along the metapath, or only consider one metapath. To address these three limitations, we propose a new model named Metapath Aggregated Graph Neural Network (MAGNN) to boost the final performance. Specifically, MAGNN employs three major components, i.e., the node content transformation to encapsulate input node attributes, the intra-metapath aggregation to incorporate intermediate semantic nodes, and the inter-metapath aggregation to combine messages from multiple metapaths. Extensive experiments on three real-world heterogeneous graph datasets for node classification, node clustering, and link prediction show that MAGNN achieves more accurate prediction results than state-of-the-art baselines.
  - Code [MAGNN](https://github.com/cynricfu/MAGNN)
  
- **Heterogeneous Graph Transformer**
  - Author: Ziniu Hu, Yuxiao Dong, Kuansan Wang, Yizhou Sun
  - Publication: WWW 2020
  - Date: 3 Mar 2020
  - Link: <https://arxiv.org/abs/2003.01332>
  - Abstract: Recent years have witnessed the emerging success of graph neural networks (GNNs) for modeling structured data. However, most GNNs are designed for homogeneous graphs, in which all nodes and edges belong to the same types, making them infeasible to represent heterogeneous structures. In this paper, we present the Heterogeneous Graph Transformer (HGT) architecture for modeling Web-scale heterogeneous graphs. To model heterogeneity, we design node- and edge-type dependent parameters to characterize the heterogeneous attention over each edge, empowering HGT to maintain dedicated representations for different types of nodes and edges. To handle dynamic heterogeneous graphs, we introduce the relative temporal encoding technique into HGT, which is able to capture the dynamic structural dependency with arbitrary durations. To handle Web-scale graph data, we design the heterogeneous mini-batch graph sampling algorithm---HGSampling---for efficient and scalable training. Extensive experiments on the Open Academic Graph of 179 million nodes and 2 billion edges show that the proposed HGT model consistently outperforms all the state-of-the-art GNN baselines by 9%--21% on various downstream tasks.
  - Code [Heterogeneous Graph Transformer (HGT)](https://github.com/acbull/pyHGT)

- **Subgraph Neural Networks**
  - Author: Emily Alsentzer, Samuel G. Finlayson, Michelle M. Li, Marinka Zitnik
  - Publication: NeurIPS 2020
  - Date: 18 Jun 2020
  - Link: <https://arxiv.org/abs/2006.10538>
  - Abstract: Deep learning methods for graphs achieve remarkable performance on many node-level and graph-level prediction tasks. However, despite the proliferation of the methods and their success, prevailing Graph Neural Networks (GNNs) neglect subgraphs, rendering subgraph prediction tasks challenging to tackle in many impactful applications. Further, subgraph prediction tasks present several unique challenges: subgraphs can have non-trivial internal topology, but also carry a notion of position and external connectivity information relative to the underlying graph in which they exist. Here, we introduce SubGNN, a subgraph neural network to learn disentangled subgraph representations. We propose a novel subgraph routing mechanism that propagates neural messages between the subgraph's components and randomly sampled anchor patches from the underlying graph, yielding highly accurate subgraph representations. SubGNN specifies three channels, each designed to capture a distinct aspect of subgraph topology, and we provide empirical evidence that the channels encode their intended properties. We design a series of new synthetic and real-world subgraph datasets. Empirical results for subgraph classification on eight datasets show that SubGNN achieves considerable performance gains, outperforming strong baseline methods, including node-level and graph-level GNNs, by 19.8% over the strongest baseline. SubGNN performs exceptionally well on challenging biomedical datasets where subgraphs have complex topology and even comprise multiple disconnected components.
  - Note
    - [Subgraph Neural Networks 论文阅读](https://zhuanlan.zhihu.com/p/337203750)
  
- **Are we really making much progress?: Revisiting, benchmarking and refining heterogeneous graph neural networks**
  - Author: Qingsong Lv, Ming Ding, Qiang Liu, Yuxiang Chen, Wenzheng Feng, Siming
He, Chang Zhou, Jianguo Jiang, Yuxiao Dong, Jie Tang. 2
  - Publication: KDD 2021
  - Date: 14 August 2021
  - Link: <https://dl.acm.org/doi/abs/10.1145/3447548.3467350>
  - Abstract: Heterogeneous graph neural networks (HGNNs) have been blossoming in recent years, but the unique data processing and evaluation setups used by each work obstruct a full understanding of their advancements. In this work, we present a systematical reproduction of 12 recent HGNNs by using their official codes, datasets, settings, and hyperparameters, revealing surprising findings about the progress of HGNNs. We find that the simple homogeneous GNNs, e.g., GCN and GAT, are largely underestimated due to improper settings. GAT with proper inputs can generally match or outperform all existing HGNNs across various scenarios. To facilitate robust and reproducible HGNN research, we construct the Heterogeneous Graph Benchmark (HGB) , consisting of 11 diverse datasets with three tasks. HGB standardizes the process of heterogeneous graph data splits, feature processing, and performance evaluation. Finally, we introduce a simple but very strong baseline Simple-HGN-which significantly outperforms all previous models on HGB-to accelerate the advancement of HGNNs in the future.
  - [Slides](https://keg.cs.tsinghua.edu.cn/jietang/publications/KDD21-slides-Lv-et-al-HeterGNN.pdf)
  - [Code & Data Heterogeneous Graph Benchmark](https://github.com/THUDM/HGB)
  - CSDN Blog [【KDD2021】Are we really making much progress? Revisiting, benchmarking, and refining HGNNs](https://blog.csdn.net/qq_36291847/article/details/120712639)
  - 异构图神经蓬勃发展但网络数据处理和评估设置存在差异。本文发现简单的同构GNNs被低估，具有适当输入的GAT已经强于HGNNs。提出异构图基准，标准化数据分割、特征处理、性能评估。
  - HGT、simple-HGNN

<!--
- ****
  - Author: 
  - Publication: 
  - Date: 
  - Link: <>
  - Abstract:
-->

<!-- 
Shirui Pan's Lab
- [ICDM 2019 | Relation Structure-Aware Heterogeneous Graph Neural Network](https://shiruipan.github.io/publication/icdm-19-zhu/)
- [IEEE Transactions on Neural Networks and Learning Systems | A comprehensive survey on graph neural networks](https://shiruipan.github.io/publication/wu-2019-comprehensive/)
- [IJCAI-20 | Reasoning Like Human: Hierarchical Reinforcement Learning for Knowledge Graph Reasoning](https://shiruipan.github.io/publication/ijcai-2020-wan/)
- [JCDL-20 | Multivariate Relations Aggregation Learning in Social Networks](https://shiruipan.github.io/publication/jcdl-2020-xu/)
- [PAKDD-21 | Heterogeneous Graph Attention Network for Small and Medium-Sized Enterprises Bankruptcy Prediction](https://shiruipan.github.io/publication/pakdd-21-zheng/)
- [A survey on knowledge graphs: representation, acquisition, and applications](https://shiruipan.github.io/publication/tnnls-21-ji/)
- [Graph Learning: A Survey](https://shiruipan.github.io/publication/tai-21-xia/)
- [A Survey of Community Detection Approaches: From Statistical Modeling to Deep Learning](https://shiruipan.github.io/publication/tkde-jin-21/)
- [Graph self-supervised learning: A survey](https://shiruipan.github.io/publication/liu-21-survey/) 
- -->
