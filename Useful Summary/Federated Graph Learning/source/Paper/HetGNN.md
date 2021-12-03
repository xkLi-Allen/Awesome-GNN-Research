<!--Title-->
# Heterogeneous Graph Neural Network

## Basic Message

- Author: Zhang, Chuxu and Song, Dongjin and Huang, Chao and Swami, Ananthram and Chawla, Nitesh V.
- Publication: SIGKDD 2019
- Date: 25 July 2019
- Link: <https://dl.acm.org/doi/abs/10.1145/3292500.3330961>

## Take Home Message

<!-- 
take home message 总结文章的核心思想
写完笔记之后最后填，概述文章的内容，也是查阅笔记的时候先看的一段。
写文章summary切记需要通过自己的思考，用自己的语言描述。 
-->
  
## Abstract

Representation learning in heterogeneous graphs aims to pursue a meaningful vector representation for each node so as to facilitate downstream applications such as link prediction, personalized recommendation, node classification, etc. This task, however, is challenging not only because of the demand to incorporate heterogeneous structural (graph) information consisting of multiple types of nodes and edges, but also due to the need for considering heterogeneous attributes or contents (e.g., text or image) associated with each node. Despite a substantial amount of effort has been made to homogeneous (or heterogeneous) graph embedding, attributed graph embedding as well as graph neural networks, few of them can jointly consider heterogeneous structural (graph) information as well as heterogeneous contents information of each node effectively. In this paper, we propose HetGNN, a heterogeneous graph neural network model, to resolve this issue. Specifically, we first introduce a random walk with restart strategy to sample a fixed size of strongly correlated heterogeneous neighbors for each node and group them based upon node types. Next, we design a neural network architecture with two modules to aggregate feature information of those sampled neighboring nodes. The first module encodes "deep" feature interactions of heterogeneous contents and generates content embedding for each node. The second module aggregates content (attribute) embeddings of different neighboring groups (types) and further combines them by considering the impacts of different groups to obtain the ultimate node embedding. Finally, we leverage a graph context loss and a mini-batch gradient descent procedure to train the model in an end-to-end manner. Extensive experiments on several datasets demonstrate that HetGNN can outperform state-of-the-art baselines in various graph mining tasks, i.e., link prediction, recommendation, node classification & clustering and inductive node classification & clustering.

<!-- 
	背景
	问题
	现状
	缺陷
	方法
	结果 
-->

- 表示图学习，学习图表示。
- 利用异构信息学习图表示
  - 异构信息：图结构信息，节点元信息
- 之前的工作不能有效联合考虑两部分信息
- HetGNN：
  - 利用带有重启策略的随机游走进行邻居采样
  - 利用分为两个模块的神经网络对采样节点进行聚合
    - 节点内部进行深度特征交互，生成节点编码
    - 根据节点类别进行聚合，更新节点编码

## Introduction
<!-- 背景知识 -->

### Related works

<!-- 哪些文章的哪些结论，与本文联系 -->

### Problem Statement
<!-- 问题陈述 -->

<!-- 
- 需要解决的问题是什么？
- 扩充知识面
  - 重建别人的想法，通过读Introduction思考别人是如何想出来的
- 假设
  - 有什么基本假设、是否正确、假设是否可以系统化验证
  - 假设很有可能是错的，还可以用哪些其他方法来验证
- 应用场景 
- -->

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
