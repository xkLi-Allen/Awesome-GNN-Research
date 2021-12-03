# Federated Graph Learning Paper List

Generally sorted by category and date.

***

## FL & GNN

- **SGNN: A Graph Neural Network Based Federated Learning Approach by Hiding Structure**
  - Author: Guangxu Mei ; Ziyu Guo; Shijun Liu; Li Pan. (SDU)
  - Publication: Big Data 2019
  - Date: 24 February 2020
  - Link: <https://ieeexplore.ieee.org/abstract/document/9005983/>
  - Abstract: Networks are general tools for modeling numerous information with features and complex relations. Network Embedding aims to learn low-dimension representations for vertexes in the network with rich information including content information and structural information. In recent years, many models based on neural network have been proposed to map the network representations into embedding space whose dimension is much lower than that in original space. However, most of existing methods have the following limitations: 1) they are based on content of nodes in network, failing to measure the structure similarity of nodes; 2) they cannot do well in protecting the privacy of users including the original content information and the structural information. In this paper, we propose a similarity-based graph neural network model, SGNN, which captures the structure information of nodes precisely in node classification tasks. It also takes advantage of the thought of federated learning to hide the original information from different data sources to protect users' privacy. We use deep graph neural network with convolutional layers and dense layers to classify the nodes based on their structures and features. The node classification experiment results on public data sets including Aminer coauthor network, Brazil and Europe flight networks indicate that our proposed model outperforms state-of-the-art models with a higher accuracy.
  -
- **Vertically Federated Graph Neural Network for Privacy-Preserving Node Classification**
  - Author: Jun Zhou, Chaochao Chen, Longfei Zheng, Huiwen Wu, Jia Wu, Xiaolin Zheng, Bingzhe Wu, Ziqi Liu, Li Wang
  - Publication: Preprint
  - Date: 25 May 2020
  - Link: <https://arxiv.org/pdf/2005.11903>
  - Abstract: Recently, Graph Neural Network (GNN) has achieved remarkable progresses in various real-world tasks on graph data, consisting of node features and the adjacent information between different nodes. High-performance GNN models always depend on both rich features and complete edge information in graph. However, such information could possibly be isolated by different data holders in practice, which is the so-called data isolation problem. To solve this problem, in this paper, we propose VFGNN, a federated GNN learning paradigm for privacy-preserving node classification task under data vertically partitioned setting, which can be generalized to existing GNN models. Specifically, we split the computation graph into two parts. We leave the private data (i.e., features, edges, and labels) related computations on data holders, and delegate the rest of computations to a semi-honest server. We also propose to apply differential privacy to prevent potential information leakage from the server. We conduct experiments on three benchmarks and the results demonstrate the effectiveness of VFGNN.
  - Used Name: Privacy-Preserving Graph Neural Network for Node Classification
  - Note: 数据孤岛问题阻碍了图神经网络利用丰富的节点特征和图信息。 VFGNN：**纵向**联邦图学习范式。将计算分为客户端的隐私数据计算和服务器端计算，并利用差分隐私防止数据泄密。
  - [论文笔记：Arxiv 2021 Vertically Federated Graph Neural Network for Privacy-Preserving Node Classification](https://zhuanlan.zhihu.com/p/428406637)

- **Distributed Training of Graph Convolutional Networks**
  - Author: Simone Scardapane, Indro Spinelli, Paolo Di Lorenzo
  - Publication: TSIPN 2021
  - Date: 13 Jul 2020
  - Link: <https://arxiv.org/abs/2007.06281>
  - Abstract: The aim of this work is to develop a fully-distributed algorithmic framework for training graph convolutional networks (GCNs). The proposed method is able to exploit the meaningful relational structure of the input data, which are collected by a set of agents that communicate over a sparse network topology. After formulating the centralized GCN training problem, we first show how to make inference in a distributed scenario where the underlying data graph is split among different agents. Then, we propose a distributed gradient descent procedure to solve the GCN training problem. The resulting model distributes computation along three lines: during inference, during back-propagation, and during optimization. Convergence to stationary solutions of the GCN training problem is also established under mild conditions. Finally, we propose an optimization criterion to design the communication topology between agents in order to match with the graph describing data relationships. A wide set of numerical results validate our proposal. To the best of our knowledge, this is the first work combining graph convolutional neural networks with distributed optimization.
  - Useful Link
    - [论文笔记：TSIPN 2021 Distributed Training of Graph Convolutional Networks](https://zhuanlan.zhihu.com/p/433329525)
    - Note：去中心化的分布式图学习。

- **Federated Dynamic GNN with Secure Aggregation.**
  - Author: Meng Jiang, Taeho Jung, Ryan Karl, Tong Zhao
  - Publication: Preprint
  - Date: 15 Sep 2020
  - Link: <https://arxiv.org/pdf/2009.07351>
  - Abstract: Given video data from multiple personal devices or street cameras, can we exploit the structural and dynamic information to learn dynamic representation of objects for applications such as distributed surveillance, without storing data at a central server that leads to a violation of user privacy? In this work, we introduce Federated Dynamic Graph Neural Network (Feddy), a distributed and secured framework to learn the object representations from multi-user graph sequences: i) It aggregates structural information from nearby objects in the current graph as well as dynamic information from those in the previous graph. It uses a self-supervised loss of predicting the trajectories of objects. ii) It is trained in a federated learning manner. The centrally located server sends the model to user devices. Local models on the respective user devices learn and periodically send their learning to the central server without ever exposing the user's data to server. iii) Studies showed that the aggregated parameters could be inspected though decrypted when broadcast to clients for model synchronizing, after the server performed a weighted average. We design an appropriate aggregation mechanism of secure aggregation primitives that can protect the security and privacy in federated learning with scalability. Experiments on four video camera datasets (in four different scenes) as well as simulation demonstrate that Feddy achieves great effectiveness and security.
  - Note：Feddy：学习动态表示，利用自监督loss进行轨迹预测。
- **ASFGNN: Automated Separated-Federated Graph Neural Network**
  - Author: Longfei Zheng, Jun Zhou, Chaochao Chen, Bingzhe Wu, Li Wang, Benyu Zhang
  - Publication: PPNA 2021
  - Date: 6 Nov 2020
  - Link: <https://arxiv.org/pdf/2011.03248>
  - Abstract: Graph Neural Networks (GNNs) have achieved remarkable performance by taking advantage of graph data. The success of GNN models always depends on rich features and adjacent relationships. However, in practice, such data are usually isolated by different data owners (clients) and thus are likely to be Non-Independent and Identically Distributed (Non-IID). Meanwhile, considering the limited network status of data owners, hyper-parameters optimization for collaborative learning approaches is time-consuming in data isolation scenarios. To address these problems, we propose an Automated Separated-Federated Graph Neural Network (ASFGNN) learning paradigm. ASFGNN consists of two main components, i.e., the training of GNN and the tuning of hyper-parameters. Specifically, to solve the data Non-IID problem, we first propose a separated-federated GNN learning model, which decouples the training of GNN into two parts: the message passing part that is done by clients separately, and the loss computing part that is learnt by clients federally. To handle the time-consuming parameter tuning problem, we leverage Bayesian optimization technique to automatically tune the hyper-parameters of all the clients. We conduct experiments on benchmark datasets and the results demonstrate that ASFGNN significantly outperforms the naive federated GNN, in terms of both accuracy and parameter-tuning efficiency.

  - Note: 本文关注联邦图学习的两个问题：数据Non-IID和超参数调优耗时。提出ASFGNN：对于Non-IID问题，采用解耦的GNN训练，本地进行信息传递，联邦进行loss计算。对于超参数调优，采用贝叶斯优化。
    - 横向联邦学习
    - 解耦的GNN训练：客户端在本地进行MP，得到Node Embedding并输入到判别模型计算loss，然后首次更新消息传递模型和判别模型。服务器端联邦聚合判别模型，并将全局判别模型广播给客户端计算JS散度，再次更新判别模型。
  
  - Useful Link
    - [论文笔记：PPNA 2021 ASFGNN: Automated Separated-Federated Graph Neural Network](https://zhuanlan.zhihu.com/p/431283541)

- **GraphFL: A Federated Learning Framework for Semi-Supervised Node Classification on Graphs**
  - Author: Binghui Wang, Ang Li, Hai Li, Yiran Chen
  - Publication: Preprint
  - Date: 8 Dec 2020
  - Link: <https://arxiv.org/pdf/2012.04187>
  - Abstract: Graph-based semi-supervised node classification (GraphSSC) has wide applications, ranging from networking and security to data mining and machine learning, etc. However, existing centralized GraphSSC methods are impractical to solve many real-world graph-based problems, as collecting the entire graph and labeling a reasonable number of labels is time-consuming and costly, and data privacy may be also violated. Federated learning (FL) is an emerging learning paradigm that enables collaborative learning among multiple clients, which can mitigate the issue of label scarcity and protect data privacy as well. Therefore, performing GraphSSC under the FL setting is a promising solution to solve real-world graph-based problems. However, existing FL methods 1) perform poorly when data across clients are non-IID, 2) cannot handle data with new label domains, and 3) cannot leverage unlabeled data, while all these issues naturally happen in real-world graph-based problems. To address the above issues, we propose the first FL framework, namely GraphFL, for semi-supervised node classification on graphs. Our framework is motivated by meta-learning methods. Specifically, we propose two GraphFL methods to respectively address the non-IID issue in graph data and handle the tasks with new label domains. Furthermore, we design a self-training method to leverage unlabeled graph data. We adopt representative graph neural networks as GraphSSC methods and evaluate GraphFL on multiple graph datasets. Experimental results demonstrate that GraphFL significantly outperforms the compared FL baseline and GraphFL with self-training can obtain better performance.
  - GraphFL
  - Note：三个问题：Non-IID；new label domains； unlabeled data。GraphFL：motivated by meta-learning，two GraphFL methods to address Non-IID and new label domains；self-training to leverage unlabeled data
    - Setting：
    - model-agnostic meta-learning (MAML) 不是学习一个在所有任务上都表现良好的模型，而是学习一个任务无关的初始化参数。
    - 为了解决NON-IID问题，把每个客户端的训练视为一个任务，初始化服务器全局模型。
  - Useful Link
    - [论文笔记：Arxiv 2020 GraphFL: A Federated Learning Framework for Semi-Supervised Node Classification](https://zhuanlan.zhihu.com/p/431479904)

- **Device Sampling for Heterogeneous Federated Learning: Theory, Algorithms, and Implementation**
  - Author: Su Wang, Mengyuan Lee, Seyyedali Hosseinalipour, Roberto Morabito, Mung Chiang, Christopher G. Brinton
  - Publication: IEEE International Conference on Computer Communications (INFOCOM) 2021  (CCF-A 计算机网络)
  - Date: 4 Jan 2021
  - Link: <https://arxiv.org/abs/2101.00787>
  - Abstract: The conventional federated learning (FedL) architecture distributes machine learning (ML) across worker devices by having them train local models that are periodically aggregated by a server. FedL ignores two important characteristics of contemporary wireless networks, however: (i) the network may contain heterogeneous communication/computation resources, while (ii) there may be significant overlaps in devices' local data distributions. In this work, we develop a novel optimization methodology that jointly accounts for these factors via intelligent device sampling complemented by device-to-device (D2D) offloading. Our optimization aims to select the best combination of sampled nodes and data offloading configuration to maximize FedL training accuracy subject to realistic constraints on the network topology and device capabilities. Theoretical analysis of the D2D offloading subproblem leads to new FedL convergence bounds and an efficient sequential convex optimizer. Using this result, we develop a sampling methodology based on graph convolutional networks (GCNs) which learns the relationship between network attributes, sampled nodes, and resulting offloading that maximizes FedL accuracy. Through evaluation on real-world datasets and network measurements from our IoT testbed, we find that our methodology while sampling less than 5% of all devices outperforms conventional FedL substantially both in terms of trained model accuracy and required resource utilization.
  - 提升联邦学习通信效率：减少参数传输、优化全局聚合、优化设备采样。
  - 本文属于第三类。采样时考虑了设备资源的异质性和本地数据分布的重叠。
  
- **FedGNN: Federated Graph Neural Network for Privacy-Preserving Recommendation**
  - Author: Chuhan Wu, Fangzhao Wu, Yang Cao, Yongfeng Huang, Xing Xie
  - Publication: FL-ICML Workshop 2021
  - Date: 9 Feb 2021
  - Link: <https://arxiv.org/abs/2102.04925>  
  - Abstract: Graph neural network (GNN) is widely used for recommendation to model high-order interactions between users and items. Existing GNN-based recommendation methods rely on centralized storage of user-item graphs and centralized model learning. However, user data is privacy-sensitive, and the centralized storage of user-item graphs may arouse privacy concerns and risk. In this paper, we propose a federated framework for privacy-preserving GNN-based recommendation, which can collectively train GNN models from decentralized user data and meanwhile exploit high-order user-item interaction information with privacy well protected. In our method, we locally train GNN model in each user client based on the user-item graph inferred from the local user-item interaction data. Each client uploads the local gradients of GNN to a server for aggregation, which are further sent to user clients for updating local GNN models. Since local gradients may contain private information, we apply local differential privacy techniques to the local gradients to protect user privacy. In addition, in order to protect the items that users have interactions with, we propose to incorporate randomly sampled items as pseudo interacted items for anonymity. To incorporate high-order user-item interactions, we propose a user-item graph expansion method that can find neighboring users with co-interacted items and exchange their embeddings for expanding the local user-item graphs in a privacy-preserving way. Extensive experiments on six benchmark datasets validate that our approach can achieve competitive results with existing centralized GNN-based recommendation methods and meanwhile effectively protect user privacy.
  - Note：
    - Recommendation，user-item 二分图
    - 客户端将本地 GNN 梯度上传汇总，服务器聚合梯度（GNN梯度，embedding梯度）发送给客户端以更新本地 GNN 模型。
    - 为保护梯度中的隐私信息，对本地梯度应用差分隐私。
    - 隐私保护问题
      - embedding梯度：有交互才有梯度，有梯度就有交互。
      - GNN梯度：现有的FedMF利用同态加密（HE），但需要存储并传输item表。
      - pseudo interacted item sampling：item噪声
      - local differential privacy：梯度噪声
  - [论文笔记：ICML 2021 FedGNN: Federated Graph Neural Network for Privacy-Preserving Recommendation](https://zhuanlan.zhihu.com/p/428783383)

- **Personalized Federated Learning using Hypernetworks**
  - Author: Aviv Shamsian, Aviv Navon, Ethan Fetaya, Gal Chechik
  - Publication: ICML 2021
  - Date: 8 Mar 2021
  - Link: <https://arxiv.org/abs/2103.04628>
  - Abstract: Personalized federated learning is tasked with training machine learning models for multiple clients, each with its own data distribution. The goal is to train personalized models in a collaborative way while accounting for data disparities across clients and reducing communication costs. We propose a novel approach to this problem using hypernetworks, termed pFedHN for personalized Federated HyperNetworks. In this approach, a central hypernetwork model is trained to generate a set of models, one model for each client. This architecture provides effective parameter sharing across clients, while maintaining the capacity to generate unique and diverse personal models. Furthermore, since hypernetwork parameters are never transmitted, this approach decouples the communication cost from the trainable model size. We test pFedHN empirically in several personalized federated learning challenges and find that it outperforms previous methods. Finally, since hypernetworks share information across clients we show that pFedHN can generalize better to new clients whose distributions differ from any client observed during training.
  - [Official Code](https://github.com/AvivSham/pFedHN)
  - Note：个性化联邦学习
  - Useful Link
    - [「ICML2021」| 联邦学习系列 — Personalized Federated Learning using Hypernetworks](https://zhuanlan.zhihu.com/p/387491634)
- **FL-AGCNS: Federated Learning Framework for Automatic Graph Convolutional Network Search**
  - Author: Chunnan Wang, Bozhou Chen, Geng Li, Hongzhi Wang
  - Publication: preprint
  - Date: 9 Apr 2021
  - Link: <https://arxiv.org/pdf/2104.04141>
  - Abstract: Recently, some Neural Architecture Search (NAS) techniques are proposed for the automatic design of Graph Convolutional Network (GCN) architectures. They bring great convenience to the use of GCN, but could hardly apply to the Federated Learning (FL) scenarios with distributed and private datasets, which limit their applications. Moreover, they need to train many candidate GCN models from scratch, which is inefficient for FL. To address these challenges, we propose FL-AGCNS, an efficient GCN NAS algorithm suitable for FL scenarios. FL-AGCNS designs a federated evolutionary optimization strategy to enable distributed agents to cooperatively design powerful GCN models while keeping personal information on local devices. Besides, it applies the GCN SuperNet and a weight sharing strategy to speed up the evaluation of GCN models. Experimental results show that FL-AGCNS can find better GCN models in short time under the FL framework, surpassing the state-of-the-arts NAS methods and GCN models.
- **FedGraphNN: A Federated Learning System and Benchmark for Graph Neural Networks**
  - Author: Chaoyang He, Keshav Balasubramanian, Emir Ceyani, Carl Yang, Han Xie, Lichao Sun, Lifang He, Liangwei Yang, Philip S. Yu, Yu Rong, Peilin Zhao, Junzhou Huang, Murali Annavaram, Salman Avestimehr
  - Publication: ICLR-DPML 2021 & MLSys21-GNNSys 2021
  - Date: 14 Apr 2021
  - Link: <https://arxiv.org/pdf/2104.07145>
  - Abstract: Graph Neural Network (GNN) research is rapidly growing thanks to the capacity of GNNs in learning distributed representations from graph-structured data. However, centralizing a massive amount of real-world graph data for GNN training is prohibitive due to privacy concerns, regulation restrictions, and commercial competitions. Federated learning (FL), a trending distributed learning paradigm, provides possibilities to solve this challenge while preserving data privacy. Despite recent advances in vision and language domains, there is no suitable platform for the FL of GNNs. To this end, we introduce FedGraphNN, an open FL benchmark system that can facilitate research on federated GNNs. FedGraphNN is built on a unified formulation of graph FL and contains a wide range of datasets from different domains, popular GNN models, and FL algorithms, with secure and efficient system support. Particularly for the datasets, we collect, preprocess, and partition 36 datasets from 7 domains, including both publicly available ones and specifically obtained ones such as hERG and Tencent. Our empirical analysis showcases the utility of our benchmark system, while exposing significant challenges in graph FL: federated GNNs perform worse in most datasets with a non-IID split than centralized GNNs; the GNN model that attains the best result in the centralized setting may not maintain its advantage in the FL setting. These results imply that more research efforts are needed to unravel the mystery behind federated GNNs. Moreover, our system performance analysis demonstrates that the FedGraphNN system is computationally efficient and secure to large-scale graphs datasets. We maintain the source code at this https URL.
  - *Github* [FedGraphNN](https://github.com/FedML-AI/FedGraphNN)
  - <https://github.com/FedML-AI/FedGraphNN/issues/7>
  
  - Useful Link
    - [[ICLR 2021 Workshop Video Presentation](https://studio.slideslive.com/web_recorder/share/35243?s=c6631fed-5f29-42ef-8cf0-6e2b8c0b2359)]
    - [[MLSys 2021 Workshop Camera Ready Version](https://gnnsys.github.io/papers/GNNSys21_paper_3.pdf)]
    - [[MLSys 2021 Poster](https://gnnsys.github.io/posters/GNNSys21_poster_3.pdf)]
    - [[Code](https://github.com/FedML-AI/FedGraphNN)]
  - [论文笔记：ICLR 2021 FedGraphNN: A Federated Learning Benchmark System for Graph Neural Networks](https://zhuanlan.zhihu.com/p/429220636)

- **A Graph Federated Architecture with Privacy Preserving Learning**
  - Author: Elsa Rizk, Ali H. Sayed
  - Publication:
  - Date: 26 Apr 2021
  - Link: <https://arxiv.org/abs/2104.13215>
  - Abstract: Federated learning involves a central processor that works with multiple agents to find a global model. The process consists of repeatedly exchanging estimates, which results in the diffusion of information pertaining to the local private data. Such a scheme can be inconvenient when dealing with sensitive data, and therefore, there is a need for the privatization of the algorithms. Furthermore, the current architecture of a server connected to multiple clients is highly sensitive to communication failures and computational overloads at the server. Thus in this work, we develop a private multi-server federated learning scheme, which we call graph federated learning. We use cryptographic and differential privacy concepts to privatize the federated learning algorithm that we extend to the graph structure. We study the effect of privatization on the performance of the learning algorithm for general private schemes that can be modeled as additive noise. We show under convexity and Lipschitz conditions, that the privatized process matches the performance of the non-private algorithm, even when we increase the noise variance.

- **Cluster-driven Graph Federated Learning over Multiple Domains**
  - Author: Debora Caldarola, Massimiliano Mancini, Fabio Galasso, Marco Ciccone, Emanuele Rodolà, Barbara Caputo
  - Publication: CVPR21 Workshop Learning from Limited or Imperfect Data (L^2ID)
  - Date: 29 Apr 2021
  - Link: <https://arxiv.org/pdf/2104.14628>
  - Abstract: Federated Learning (FL) deals with learning a central model (i.e. the server) in privacy-constrained scenarios, where data are stored on multiple devices (i.e. the clients). The central model has no direct access to the data, but only to the updates of the parameters computed locally by each client. This raises a problem, known as statistical heterogeneity, because the clients may have different data distributions (i.e. domains). This is only partly alleviated by clustering the clients. Clustering may reduce heterogeneity by identifying the domains, but it deprives each cluster model of the data and supervision of others. Here we propose a novel Cluster-driven Graph Federated Learning (FedCG). In FedCG, clustering serves to address statistical heterogeneity, while Graph Convolutional Networks (GCNs) enable sharing knowledge across them. FedCG: i) identifies the domains via an FL-compliant clustering and instantiates domain-specific modules (residual branches) for each domain; ii) connects the domain-specific modules through a GCN at training to learn the interactions among domains and share knowledge; and iii) learns to cluster unsupervised via teacher-student classifier-training iterations and to address novel unseen test domains via their domain soft-assignment scores. Thanks to the unique interplay of GCN over clusters, FedCG achieves the state-of-the-art on multiple FL benchmarks.
- **FedGL: Federated Graph Learning Framework with Global Self-Supervision**
  - Author: Chuan Chen, Weibo Hu, Ziyue Xu, Zibin Zheng
  - Publication: Preprint
  - Date: 7 May 2021
  - Link: <https://arxiv.org/pdf/2105.03170>
  - Abstract: Graph data are ubiquitous in the real world. Graph learning (GL) tries to mine and analyze graph data so that valuable information can be discovered. Existing GL methods are designed for centralized scenarios. However, in practical scenarios, graph data are usually distributed in different organizations, i.e., the curse of isolated data islands. To address this problem, we incorporate federated learning into GL and propose a general Federated Graph Learning framework FedGL, which is capable of obtaining a high-quality global graph model while protecting data privacy by discovering the global self-supervision information during the federated training. Concretely, we propose to upload the prediction results and node embeddings to the server for discovering the global pseudo label and global pseudo graph, which are distributed to each client to enrich the training labels and complement the graph structure respectively, thereby improving the quality of each local model. Moreover, the global self-supervision enables the information of each client to flow and share in a privacy-preserving manner, thus alleviating the heterogeneity and utilizing the complementarity of graph data among different clients. Finally, experimental results show that FedGL significantly outperforms baselines on four widely used graph datasets.
  - [论文笔记：Arxiv 2021 FedGL: Federated Graph Learning Framework with Global Self-Supervision](https://zhuanlan.zhihu.com/p/431049080)
- **Federated Graph Learning -- A Position Paper**
  - Author: Huanding Zhang, Tao Shen, Fei Wu, Mingyang Yin, Hongxia Yang, Chao Wu
  - Publication: Preprint
  - Date: 24 May 2021
  - Link: <https://arxiv.org/pdf/2105.11099>
  - Abstract: Graph neural networks (GNN) have been successful in many fields, and derived various researches and applications in real industries. However, in some privacy sensitive scenarios (like finance, healthcare), training a GNN model centrally faces challenges due to the distributed data silos. Federated learning (FL) is a an emerging technique that can collaboratively train a shared model while keeping the data decentralized, which is a rational solution for distributed GNN training. We term it as federated graph learning (FGL). Although FGL has received increasing attention recently, the definition and challenges of FGL is still up in the air. In this position paper, we present a categorization to clarify it. Considering how graph data are distributed among clients, we propose four types of FGL: inter-graph FL, intra-graph FL and graph-structured FL, where intra-graph is further divided into horizontal and vertical FGL. For each type of FGL, we make a detailed discussion about the formulation and applications, and propose some potential challenges.
  - Note
    - A categorization of federated graph learning
      - Inter-graph federated learning
      - Intra-graph federated learning
      - Graph-structured federated learning
    - Challenges
      - NON-IID graph structure
        - **图结构**的NON-IID也可能影响其学习过程
      - Isolated graph in horizontal intra-graph FL
        - 发现客户端子图之间的潜在结构联系
        - FedGNN 基于同态加密扩展本地子图
      - Entities matching and secure data sharing in vertical intra-graph FL
        - 纵向联邦 实体匹配 安全数据共享
        - FedE 通过服务器中的匹配表进行知识图谱嵌入，一定程度上违反了隐私保护
      - Dataset of intra-graph FL
        - 没有合适的图数据集
        - 欧式数据容易模拟数据分布 图数据的结构信息困难
      - Communication and memory consumption
        - 通信和内存消耗
        - 图压缩技术：量化、剪枝、蒸馏
  - Useful Link
    - [论文笔记：Arxiv 2021 Federated Graph Learning - A Position Paper](https://zhuanlan.zhihu.com/p/431934452)
- **SpreadGNN: Serverless Multi-task Federated Learning for Graph Neural Networks**
  - Author: Chaoyang He, Emir Ceyani, Keshav Balasubramanian, Murali Annavaram, Salman Avestimehr
  - Publication: ICML 2021
  - Date: 4 Jun 2021
  - Link: <https://arxiv.org/pdf/2106.02743>
  - Abstract: Graph Neural Networks (GNNs) are the first choice methods for graph machine learning problems thanks to their ability to learn state-of-the-art level representations from graph-structured data. However, centralizing a massive amount of real-world graph data for GNN training is prohibitive due to user-side privacy concerns, regulation restrictions, and commercial competition. Federated Learning is the de-facto standard for collaborative training of machine learning models over many distributed edge devices without the need for centralization. Nevertheless, training graph neural networks in a federated setting is vaguely defined and brings statistical and systems challenges. This work proposes SpreadGNN, a novel multi-task federated training framework capable of operating in the presence of partial labels and absence of a central server for the first time in the literature. SpreadGNN extends federated multi-task learning to realistic serverless settings for GNNs, and utilizes a novel optimization algorithm with a convergence guarantee, Decentralized Periodic Averaging SGD (DPA-SGD), to solve decentralized multi-task learning problems. We empirically demonstrate the efficacy of our framework on a variety of non-I.I.D. distributed graph-level molecular property prediction datasets with partial labels. Our results show that SpreadGNN outperforms GNN models trained over a central server-dependent federated learning system, even in constrained topologies. The source code is publicly available at [this https URL](https://github.com/FedML-AI/SpreadGNN)
  - Useful link
    - [论文笔记：ICML 2021 SpreadGNN: Serverless Multi task Federated Learning for Graph Neural Networks](https://zhuanlan.zhihu.com/p/429720860)
- **Cross-Node Federated Graph Neural Network for Spatio-Temporal Data Modeling**
  - Author: Chuizheng Meng, Sirisha Rambhatla, Yan Liu
  - Publication: KDD 2021
  - Date: 9 Jun 2021
  - Link: <https://arxiv.org/pdf/2106.05223>
  - Abstract: Vast amount of data generated from networks of sensors, wearables, and the Internet of Things (IoT) devices underscores the need for advanced modeling techniques that leverage the spatio-temporal structure of decentralized data due to the need for edge computation and licensing (data access) issues. While federated learning (FL) has emerged as a framework for model training without requiring direct data sharing and exchange, effectively modeling the complex spatio-temporal dependencies to improve forecasting capabilities still remains an open problem. On the other hand, state-of-the-art spatio-temporal forecasting models assume unfettered access to the data, neglecting constraints on data sharing. To bridge this gap, we propose a federated spatio-temporal model -- Cross-Node Federated Graph Neural Network (CNFGNN) -- which explicitly encodes the underlying graph structure using graph neural network (GNN)-based architecture under the constraint of cross-node federated learning, which requires that data in a network of nodes is generated locally on each node and remains decentralized. CNFGNN operates by disentangling the temporal dynamics modeling on devices and spatial dynamics on the server, utilizing alternating optimization to reduce the communication cost, facilitating computations on the edge devices. Experiments on the traffic flow forecasting task show that CNFGNN achieves the best forecasting performance in both transductive and inductive learning settings with no extra computation cost on edge devices, while incurring modest communication cost.
- **A Vertical Federated Learning Framework for Graph Convolutional Network**
  - Author: Xiang Ni, Xiaolong Xu, Lingjuan Lyu, Changhua Meng, Weiqiang Wang
  - Publication: Preprint
  - Date: 22 Jun 2021
  - Link: <https://arxiv.org/pdf/2106.11593>
  - Abstract: Recently, Graph Neural Network (GNN) has achieved remarkable success in various real-world problems on graph data. However in most industries, data exists in the form of isolated islands and the data privacy and security is also an important issue. In this paper, we propose FedVGCN, a federated GCN learning paradigm for privacy-preserving node classification task under data vertically partitioned setting, which can be generalized to existing GCN models. Specifically, we split the computation graph data into two parts. For each iteration of the training process, the two parties transfer intermediate results to each other under homomorphic encryption. We conduct experiments on benchmark data and the results demonstrate the effectiveness of FedVGCN in the case of GraphSage.
  - Note
    - GraphSage
  - Useful Link
    - [论文笔记：Arxiv 2021 A Vertical Federated Learning Framework for Graph Convolutional Network](https://zhuanlan.zhihu.com/p/431900470)
- **Federated Graph Classification over Non-IID Graphs**
  - Author: Han Xie, Jing Ma, Li Xiong, Carl Yang
  - Publication: NeurIPS 2021
  - Date: 25 Jun 2021
  - Link: <https://arxiv.org/abs/2106.13423>
  - Abstract: Federated learning has emerged as an important paradigm for training machine learning models in different domains. For graph-level tasks such as graph classification, graphs can also be regarded as a special type of data samples, which can be collected and stored in separate local systems. Similar to other domains, multiple local systems, each holding a small set of graphs, may benefit from collaboratively training a powerful graph mining model, such as the popular graph neural networks (GNNs). To provide more motivation towards such endeavors, we analyze real-world graphs from different domains to confirm that they indeed share certain graph properties that are statistically significant compared with random graphs. However, we also find that different sets of graphs, even from the same domain or same dataset, are non-IID regarding both graph structures and node features. To handle this, we propose a graph clustered federated learning (GCFL) framework that dynamically finds clusters of local systems based on the gradients of GNNs, and theoretically justify that such clusters can reduce the structure and feature heterogeneity among graphs owned by the local systems. Moreover, we observe the gradients of GNNs to be rather fluctuating in GCFL which impedes high-quality clustering, and design a gradient sequence-based clustering mechanism based on dynamic time warping (GCFL+). Extensive experimental results and in-depth analysis demonstrate the effectiveness of our proposed frameworks.
  - Useful Link
    - [论文笔记：NIPS 2021 Federated Graph Classification over Non-IID Graphs (GCFL)](https://zhuanlan.zhihu.com/p/430623053)
- **Subgraph Federated Learning with Missing Neighbor Generation**
  - Author: Ke Zhang, Carl Yang, Xiaoxiao Li, Lichao Sun, Siu Ming Yiu
  - Publication: NeurIPS 2021
  - Date: 25 Jun 2021
  - Link: <https://arxiv.org/abs/2106.13430>  
  - Abstract:
    Graphs have been widely used in data mining and machine learning due to their unique representation of real-world objects and their interactions. As graphs are getting bigger and bigger nowadays, it is common to see their subgraphs separately collected and stored in multiple local systems. Therefore, it is natural to consider the subgraph federated learning setting, where each local system holds a small subgraph that may be biased from the distribution of the whole graph. Hence, the subgraph federated learning aims to collaboratively train a powerful and generalizable graph mining model without directly sharing their graph data. In this work, towards the novel yet realistic setting of subgraph federated learning, we propose two major techniques: (1) FedSage, which trains a GraphSage model based on FedAvg to integrate node features, link structures, and task labels on multiple local subgraphs; (2) FedSage+, which trains a missing neighbor generator along FedSage to deal with missing links across local subgraphs. Empirical results on four real-world graph datasets with synthesized subgraph federated learning settings demonstrate the effectiveness and efficiency of our proposed techniques. At the same time, consistent theoretical implications are made towards their generalization ability on the global graphs.
  - Note
    - FedSage : FedAvg + GraphSage
    - FedSage+ : FedSage + Missing neighbor generator
    - [Meeting_20211028_FedSage](https://docs.google.com/presentation/d/1tC69KdqVy2yEOD6Ev-hBd0LiloJPvUemqAl5hFIvz5s/edit?usp=sharing)
  - Useful Link
    - [论文笔记：NIPS 2021 Subgraph Federated Learning with Missing Neighbor Generation](https://zhuanlan.zhihu.com/p/430789355)

- **Decentralized Federated Graph Neural Networks**
  - Author: Yang Pei, Renxin Mao, Yang Liu, Chaoran Chen, Shifeng Xu, Feng Qiang and Peng Zhang.
  - Publication: FTL-IJCAI Workshop 2021
  - Date:  21 Aug 2021
  - Link: <https://federated-learning.org/fl-ijcai-2021/FTL-IJCAI21_paper_20.pdf>  
  - Abstract: Due to the increasing importance of user-side privacy, federated graph neural networks are proposed recently as a reliable solution for privacypreserving data analysis by enabling the joint training of a graph neural network from graphstructured data located at different clients. However, existing federated graph neural networks are based on a centralized server to orchestrate the training process, which is unacceptable in many real-world applications such as building financial risk control models across competitive banks. In this paper, we propose a new Decentralized Federated Graph Neural Network (D-FedGNN for short) which allows multiple participants to train a graph neural network model without a centralized server. Specifically, D-FedGNN uses a decentralized parallel stochastic gradient descent algorithm DPSGD to train the graph neural network model in a peer-to-peer network structure. To protect privacy during model aggregation, D-FedGNN introduces the Diffie-Hellman key exchange method to achieve secure model aggregation between clients. Both theoretical and empirical studies show that the proposed D-FedGNN model is capable of achieving competitive results compared with traditional centralized federated graph neural networks in the tasks of classification and regression, as well as significantly reducing time cost of the communication of model parameters.
- Useful Link
  - [Poster](https://federated-learning.org/fl-ijcai-2021/P1058--poster.pdf)
  - [论文笔记：IJCAI 2021 Decentralized Federated Graph Neural Networks](https://zhuanlan.zhihu.com/p/430508567)

- **Federated Learning of Molecular Properties in a Heterogeneous Setting**
  - Author: Wei Zhu, Andrew White, Jiebo Luo
  - Publication:
  - Date:
  - Link: <https://arxiv.org/abs/2109.07258>
  - Abstract:
  - [Wei Zhu](https://zwvews.github.io/)

- **A Federated Multigraph Integration Approach for Connectional Brain Template Learning**
  - Author: Hızır Can BayramIslem Rekik
  - Publication: ML-CDS 2021
  - Date: 20 October 2021
  - Link: <https://link.springer.com/chapter/10.1007/978-3-030-89847-2_4>
  - Abstract: The connectional brain template (CBT) is a compact representation (i.e., a single connectivity matrix) multi-view brain networks of a given population. CBTs are especially very powerful tools in brain dysconnectivity diagnosis as well as holistic brain mapping if they are learned properly – i.e., occupy the center of the given population. Even though accessing large-scale datasets is much easier nowadays, it is still challenging to upload all these clinical datasets in a server altogether due to the data privacy and sensitivity. Federated learning, on the other hand, has opened a new era for machine learning algorithms where different computers are trained together via a distributed system. Each computer (i.e., a client) connected to a server, trains a model with its local dataset and sends its learnt model weights back to the server. Then, the server aggregates these weights thereby outputting global model weights encapsulating information drawn from different datasets in a privacy-preserving manner. Such a pipeline endows the global model with a generalizability power as it implicitly benefits from the diversity of the local datasets. In this work, we propose the first federated connectional brain template learning (Fed-CBT) framework to learn how to integrate multi-view brain connectomic datasets collected by different hospitals into a single representative connectivity map. First, we choose a random fraction of hospitals to train our global model. Next, all hospitals send their model weights to the server to aggregate them. We also introduce a weighting method for aggregating model weights to take full benefit from all hospitals. Our model to the best of our knowledge is the first and only federated pipeline to estimate connectional brain templates using graph neural networks. Our Fed-CBT code is available at <https://github.com/basiralab/Fed-CBT>.

- **FedGraph: Federated Graph Learning with Intelligent Sampling**
  - Author:
  - Publication: IEEE TPDS
  - Date: 2 Nov 2021
  - Link: <https://arxiv.org/abs/2111.01370>
  - Abstract:

- **Federated Knowledge Graph Embeddings with Heterogeneous Data**
  - Author:
  - Publication:
  - Date:
  - Link: <https://link.springer.com/chapter/10.1007/978-981-16-6471-7_2>
  - Abstract:

<!--
- ****
  - Author: 
  - Publication: 
  - Date: 
  - Link: <>
  - Abstract:
-->

***

## FL & Knowledge Graph

- **FedE: Embedding Knowledge Graphs in Federated Setting.**
  - Author: Mingyang Chen, Wen Zhang, Zonggang Yuan, Yantao Jia, Huajun Chen
  - Publication: Preprint
  - Date: 24 Oct 2020
  - Link: <https://arxiv.org/pdf/2010.12882>
  - Abstract: Knowledge graphs (KGs) consisting of triples are always incomplete, so it's important to do Knowledge Graph Completion (KGC) by predicting missing triples. Multi-Source KG is a common situation in real KG applications which can be viewed as a set of related individual KGs where different KGs contains relations of different aspects of entities. It's intuitive that, for each individual KG, its completion could be greatly contributed by the triples defined and labeled in other ones. However, because of the data privacy and sensitivity, a set of relevant knowledge graphs cannot complement each other's KGC by just collecting data from different knowledge graphs together. Therefore, in this paper, we introduce federated setting to keep their privacy without triple transferring between KGs and apply it in embedding knowledge graph, a typical method which have proven effective for KGC in the past decade. We propose a Federated Knowledge Graph Embedding framework FedE, focusing on learning knowledge graph embeddings by aggregating locally-computed updates. Finally, we conduct extensive experiments on datasets derived from KGE benchmark datasets and results show the effectiveness of our proposed FedE.
  - [Official Code](https://github.com/AnselCmy/FedE)

- **Improving Federated Relational Data Modeling via Basis Alignment and Weight Penalty.**
  - Author: Yilun Lin, Chaochao Chen, Cen Chen, Li Wang
  - Publication: Preprint
  - Date: 23 Nov 2020
  - Link: <https://arxiv.org/pdf/2011.11369>
  - Abstract: Federated learning (FL) has attracted increasing attention in recent years. As a privacy-preserving collaborative learning paradigm, it enables a broader range of applications, especially for computer vision and natural language processing tasks. However, to date, there is limited research of federated learning on relational data, namely Knowledge Graph (KG). In this work, we present a modified version of the graph neural network algorithm that performs federated modeling over KGs across different participants. Specifically, to tackle the inherent data heterogeneity issue and inefficiency in algorithm convergence, we propose a novel optimization algorithm, named FedAlign, with 1) optimal transportation (OT) for on-client personalization and 2) weight constraint to speed up the convergence. Extensive experiments have been conducted on several widely used datasets. Empirical results show that our proposed method outperforms the state-of-the-art FL methods, such as FedAVG and FedProx, with better convergence.
- **Differentially Private Federated Knowledge Graphs Embedding**
  - Author: Hao Peng, Haoran Li, Yangqiu Song, Vincent Zheng, Jianxin Li
  - Publication: CIKM 2021
  - Date: 17 May 2021
  - Link: <https://arxiv.org/pdf/2105.07615>
  - Abstract: Knowledge graph embedding plays an important role in knowledge representation, reasoning, and data mining applications. However, for multiple cross-domain knowledge graphs, state-of-the-art embedding models cannot make full use of the data from different knowledge domains while preserving the privacy of exchanged data. In addition, the centralized embedding model may not scale to the extensive real-world knowledge graphs. Therefore, we propose a novel decentralized scalable learning framework, \emph{Federated Knowledge Graphs Embedding} (FKGE), where embeddings from different knowledge graphs can be learnt in an asynchronous and peer-to-peer manner while being privacy-preserving. FKGE exploits adversarial generation between pairs of knowledge graphs to translate identical entities and relations of different domains into near embedding spaces. In order to protect the privacy of the training data, FKGE further implements a privacy-preserving neural network structure to guarantee no raw data leakage. We conduct extensive experiments to evaluate FKGE on 11 knowledge graphs, demonstrating a significant and consistent improvement in model quality with at most 17.85\% and 7.90\% increases in performance on triple classification and link prediction tasks.
  - Used name: Federated Knowledge Graphs Embedding
  - Note
    - Knowledge Graph
    - [Official Code](https://github.com/HKUST-KnowComp/FKGE)
    - [[CIKM 2021 | FKGE：差分隐私的联邦知识图谱嵌入]](https://mp.weixin.qq.com/s/PT4bq-xTT6eg4lmPllEOFQ)
- **Leveraging a Federation of Knowledge Graphs to Improve Faceted Search in Digital Libraries.**
  - Author: Golsa Heidari, Ahmad Ramadan, Markus Stocker, Sören Auer
  - Publication: TPDL 2021
  - Date: 5 Jul 2021
  - Link: <https://arxiv.org/pdf/2107.05447>
  - Abstract: Scientists always look for the most accurate and relevant answers to their queries in the literature. Traditional scholarly digital libraries list documents in search results, and therefore are unable to provide precise answers to search queries. In other words, search in digital libraries is metadata search and, if available, full-text search. We present a methodology for improving a faceted search system on structured content by leveraging a federation of scholarly knowledge graphs. We implemented the methodology on top of a scholarly knowledge graph. This search system can leverage content from third-party knowledge graphs to improve the exploration of scholarly content. A novelty of our approach is that we use dynamic facets on diverse data types, meaning that facets can change according to the user query. The user can also adjust the granularity of dynamic facets. An additional novelty is that we leverage third-party knowledge graphs to improve exploring scholarly knowledge.

<!--
- ****
  - Author: 
  - Publication: 
  - Date: 
  - Link: <>
  - Abstract:
-->
