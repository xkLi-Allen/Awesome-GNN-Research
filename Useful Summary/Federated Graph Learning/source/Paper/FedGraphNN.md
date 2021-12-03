<!--Title-->
# FedGraphNN: A Federated Learning System and Benchmark for Graph Neural Networks

## Basic Message

- Author: Chaoyang He, Keshav Balasubramanian, Emir Ceyani, Carl Yang, Han Xie, Lichao Sun, Lifang He, Liangwei Yang, Philip S. Yu, Yu Rong, Peilin Zhao, Junzhou Huang, Murali Annavaram, Salman Avestimehr
- Publication: ICLR-DPML 2021 & MLSys21-GNNSys 2021
- Date: 14 Apr 2021
- Link: <https://arxiv.org/pdf/2104.07145>

## Take Home Message

<!-- 
take home message 总结文章的核心思想
写完笔记之后最后填，概述文章的内容，也是查阅笔记的时候先看的一段。
写文章summary切记需要通过自己的思考，用自己的语言描述。 
-->
  
## Abstract

Graph Neural Network (GNN) research is rapidly growing thanks to the capacity of GNNs in learning distributed representations from graph-structured data. However, centralizing a massive amount of real-world graph data for GNN training is prohibitive due to privacy concerns, regulation restrictions, and commercial competitions. Federated learning (FL), a trending distributed learning paradigm, provides possibilities to solve this challenge while preserving data privacy. Despite recent advances in vision and language domains, there is no suitable platform for the FL of GNNs. To this end, we introduce FedGraphNN, an open FL benchmark system that can facilitate research on federated GNNs. FedGraphNN is built on a unified formulation of graph FL and contains a wide range of datasets from different domains, popular GNN models, and FL algorithms, with secure and efficient system support. Particularly for the datasets, we collect, preprocess, and partition 36 datasets from 7 domains, including both publicly available ones and specifically obtained ones such as hERG and Tencent. Our empirical analysis showcases the utility of our benchmark system, while exposing significant challenges in graph FL: federated GNNs perform worse in most datasets with a non-IID split than centralized GNNs; the GNN model that attains the best result in the centralized setting may not maintain its advantage in the FL setting. These results imply that more research efforts are needed to unravel the mystery behind federated GNNs. Moreover, our system performance analysis demonstrates that the FedGraphNN system is computationally efficient and secure to large-scale graphs datasets. We maintain the source code at this https URL.

<!-- 
背景

问题
现状
缺陷
方法
结果 -->

- Isolated Data Island: centralizing a massive amount of real-world graph data for GNN training is prohibitive due to privacy concerns
  - 个人隐私保护、政府法规限制、商业利益冲突
- there is no suitable platform for the FL of GNNs
- challenges in graph FL
  - federated GNNs perform worse in most datasets with a **non-IID** split than centralized GNNs;
  - the GNN model that attains the best result in the centralized setting may not **maintain its advantage** in the FL setting

## Introduction
<!-- 背景知识 -->

### Related works

<!-- 哪些文章的哪些结论，与本文联系 -->

### Problem Statement
<!-- 问题陈述 -->

<!-- - 需要解决的问题是什么？
- 扩充知识面
  - 重建别人的想法，通过读Introduction思考别人是如何想出来的
- 假设
  - 有什么基本假设、是否正确、假设是否可以系统化验证
  - 假设很有可能是错的，还可以用哪些其他方法来验证
- 应用场景 -->

## Methods
<!-- 文章设计的方法 -->

<!--
解决问题的方法/算法是什么
	主要理论、主要公式、主要创意
	创意的好处、成立条件
	为什么要用这种方法
	是否基于前人的方法？
有什么缺点、空缺、漏洞、局限
	效果不够好
	考虑不顾全面
	在应用上有哪些坏处，怎么引起的
还可以用什么方法？
方法可以还用在哪？有什么可以借鉴的地方？ 
-->
  
## Evaluation & Experiments
<!-- 实验评估 -->

<!-- 
- 作者如何评估自己的方法
- 实验的setup
   § 数据集
    □ 名称、基本参数、异同，为什么选择（Baseline）
    □ 如何处理数据以便于实验
   § 模型
   § baseline
   § 与什么方法比较
- 实验证明了哪些结论
- 实验有什么可借鉴的
- 实验有什么不足 
-->

## Discuss & Conclusion

<!-- 
作者给了哪些结论
- 哪些是strong conclusions, 哪些又是weak的conclusions?
- 文章的讨论、结论部分，
   § 结尾的地方往往有些启发性的讨论 
-->

## Reference
<!-- 列出相关性高的参考文献-->
  
## Useful link

<!-- 
论文笔记、讲解
Code Slides Web Review
Author Page 
-->

- [[ICLR 2021 Workshop Video Presentation](https://studio.slideslive.com/web_recorder/share/35243?s=c6631fed-5f29-42ef-8cf0-6e2b8c0b2359)]
- [[MLSys 2021 Workshop Camera Ready Version](https://gnnsys.github.io/papers/GNNSys21_paper_3.pdf)]
- [[MLSys 2021 Poster](https://gnnsys.github.io/posters/GNNSys21_poster_3.pdf)]
- [[Code](https://github.com/FedML-AI/FedGraphNN)]
- [论文笔记：ICLR 2021 FedGraphNN: A Federated Learning Benchmark System for Graph Neural Networks](https://zhuanlan.zhihu.com/p/429220636)
- [ICLR'21 | GNN联邦学习的新基准](https://zhuanlan.zhihu.com/p/434480171)

## Notes

<!-- - 不符合此框架，但需要额外记录的笔记。
- 英语单词、好的句子 -->

 <!-- 
读文章步骤：
  迭代式读法
  先读标题、摘要 图表 再读介绍 读讨论 读结果 读实验
  通读全文，能不查字典最好先不查字典
  边读边总结，总结主要含义，注重逻辑推理
  
 摘要
  多数文章看摘要，少数文章看全文

 实验
  结合图表

 理论：
  有什么样的假设 是否合理 其他设定
  推导是否完善 用了什么数学工具
  
 idea来源：
  突出理论还是实践
   理论：数学
   实践：跑通code，调参过程中改进，找到work的方案后思考成因
  针对特定缺点，设计方案 
-->
