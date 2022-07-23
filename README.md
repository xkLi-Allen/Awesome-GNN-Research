<p align="center">
 <img width="100px" src="https://res.cloudinary.com/anuraghazra/image/upload/v1594908242/logo_ccswme.svg" align="center" alt="Awesome GNN Research" />
 <h2 align="center">Awesome GNN Research</h2>
</p>

  <div align=center>  
      <img src="https://github-readme-stats.vercel.app/api?username=XunkaiLi&show_icons=true&theme=gruvbox_light"/>
  </div>

:fire: **Top AI&ML&DM Conferences or Journals and arXiv Prep.**

CVPR, ICCV, ECCV, ACL, MM, etc

AAAI, IJCAI, SIGIR, ICDE, SIGMOD, etc

KDD, WWW, ICLR, ICML, NeurIPS, TKDE, etc

:star: **Research Keywords**

Scalable Graph Neural Networks, Federated Graph Learning, Recommender system Based on GNN

# Wizardship

- [Scalable Graph Neural Networks](#Scalable-Graph-Neural-Networks)
  - [Graph Embedding](#Graph-Embedding)
  - [Linear-model based Graph Neural Networks](#Linear-model-based-Graph-Neural-Networks)
  - [Sampling based Graph Neural Networks](#Sampling-based-Graph-Neural-Networks)
  - [Model Compression and Quantification](#Model-Compression-and-Quantification)
  - [Efficient Architecture and Paradigm](#Efficient-Architecture-and-Paradigm)
  - [Graph Data Augmentation](#Graph-Data-Augmentation)
  - [Imbalance Graph Neural Networks](#Imbalance-Graph-Neural-Networks)
- [Federated Graph Learning](#Federated-Graph-Learning)
  - [Personalized and Heterogeneous Federated Learning in CV or NLP](#Personalized-and-Heterogeneous-Federated-Learning-in-CV-or-NLP)
  - [Theoretical Analysis of Federated Learning in CV or NLP](#Theoretical-Analysis-of-Federated-Learning-in-CV-or-NLP)
  - [CV or NLP Model Compression and Quantification in Federated Learning ](#CV-or-NLP-Model-Compression-and-Quantification-in-Federated-Learning )
  - [Transfer Federated Graph Learning and Graph Structure Federated Learning](#Transfer-Federated-Graph-Learning-and-Graph-Structure-Federated-Learning)
  - [Intra-Graph Horizontal Federated Learning](#Intra-Graph-Horizontal-Federated-Learning)
  - [Inter-Graph Horizontal Federated Learning](#Inter-Graph-Horizontal-Federated-Learning)
  - [Vertical Federal Learning](#Vertical-Federal-Learning)
- [Privacy Graph Neural Networks](#Privacy-Graph-Neural-Networks)
- [Survey and Framework Toolkits](#Survey-and-Framework-Toolkits)

# Scalable Graph Neural Networks

## Graph Embedding

- KDD'14 DeepWalk: Online Learning of Social Representations [[Paper](https://arxiv.org/abs/1403.6652)] [[Code](https://github.com/phanein/deepwalk)] [[Link](https://zhuanlan.zhihu.com/p/412713441)]
- WWW'15 LINE: Large-scale Information Network Embedding [[Paper](https://arxiv.org/abs/1503.03578)] [[Code](https://github.com/snowkylin/line)] [[Link](https://zhuanlan.zhihu.com/p/412787557)]
- KDD'16 node2vec: Scalable Feature Learning for Networks [[Paper](https://arxiv.org/abs/1607.00653)] [[Code](https://github.com/eliorc/node2vec)] [[Link](https://zhuanlan.zhihu.com/p/413046898)]
- NeurIPS'13  Distributed Representations of Words and Phrases and their Compositionality [[Paper](https://arxiv.org/abs/1310.4546)] [[Code](https://github.com/brijml/mikolov_word2vec)] [[Link](https://zhuanlan.zhihu.com/p/413169135)]
- KDD'16 Structural Deep Network Embedding [[Paper](http://www.kdd.org/kdd2016/papers/files/rfp0191-wangAemb.pdf)] [[Code](https://github.com/suanrong/SDNE)] [[Link](https://zhuanlan.zhihu.com/p/413468532)]

## Linear-model based Graph Neural Networks

- ICML'19 Simplifying Graph Convolutional Networks [[Paper](https://arxiv.org/abs/1902.07153v1)] [[Code](https://github.com/Tiiiger/SGC)] [[Link](https://zhuanlan.zhihu.com/p/411236675)]
- ICLR'19 Predict Then Propagate: Graph Neural Networks Meet Personalized PageRank [[Paper](https://arxiv.org/abs/1810.05997v5)] [[Code](https://github.com/benedekrozemberczki/APPNP)] [[Link](https://zhuanlan.zhihu.com/p/419843669)]
- arXiv'20 Scalable Graph Neural Networks for Heterogeneous Graphs [[Paper](https://arxiv.org/abs/2011.09679)] [[Code](https://github.com/facebookresearch/NARS)] [[Link](https://zhuanlan.zhihu.com/p/490723967)]
- arXiv'20 Unifying Graph Convolutional Neural Networks and Label Propagation [[Paper](https://arxiv.org/abs/2002.06755)] [[Code](https://github.com/liu6zijian/GCN-LPA-PyTorch)] [[Link](https://zhuanlan.zhihu.com/p/501120937)]
- NeurIPS‘21 Node Dependent Local Smoothing for Scalable Graph Learning [[Paper](https://arxiv.org/abs/2110.14377)] [[Code](https://github.com/zwt233/NDLS)] [[Link](https://zhuanlan.zhihu.com/p/445986238)]
- arXiv'21 Scalable and Adaptive Graph Neural Networks with Self-Label-Enhanced Training [[Paper](https://arxiv.org/pdf/2104.09376.pdf)] [[Code](https://github.com/skepsun/SAGN_with_SLE)] [[Link](https://zhuanlan.zhihu.com/p/543120470)]
- ICLR'21 Combining Label Propagation and Simple Models Out-performs Graph Neural Networks [[Paper](https://arxiv.org/pdf/2010.13993.pdf)] [[Code](https://github.com/CUAI/CorrectAndSmooth)] [[Link](https://zhuanlan.zhihu.com/p/543120470)]
- ICLR'22 Node Feature Extraction by Self-Supervised Multi-scale Neighborhood Prediction [[Paper](https://arxiv.org/pdf/2111.00064.pdf)] [[Code](https://github.com/elichienxD/SAGN_with_SLE)] [[Link](https://zhuanlan.zhihu.com/p/543120470)]
- arXiv'22 SCR: Training Graph Neural Networks with Consistency Regularization [[Paper](https://arxiv.org/pdf/2112.04319.pdf)] [[Code](https://github.com/THUDM/SCR)] [[Link](https://zhuanlan.zhihu.com/p/543120470)]
- KDD'22 Graph Attention MLP with Reliable Label Utilization [[Paper](https://arxiv.org/pdf/2108.10097.pdf)] [[Code](https://github.com/PKU-DAIR/GAMLP)] [[Link](https://zhuanlan.zhihu.com/p/543120470)]

## Sampling based Graph Neural Networks

- NeurIPS'17 Inductive Representation Learning on Large Graphs [[Paper](https://arxiv.org/abs/1706.02216v2)] [[Code](https://github.com/williamleif/GraphSAGE)] [[Link](https://zhuanlan.zhihu.com/p/411612848)]
- ICLR'18 FASTGCN: Fast Learning With Graph Convolutional Networks Via Importance Sampling [[Paper](https://arxiv.org/abs/1801.10247)] [[Code](https://github.com/matenure/FastGCN)] [[Link](https://zhuanlan.zhihu.com/p/412020874)]

## Model Compression and Quantification

- NeurIPS'14 Distilling the Knowledge in a Neural Network [[Paper](https://arxiv.org/pdf/1503.02531.pdf)] [[Code](https://github.com/JoonyoungYi/KD-pytorch)] [[Link](https://zhuanlan.zhihu.com/p/542550895)]
- ICLR'15 FitNets: Hints for Thin Deep Nets [[Paper](https://arxiv.org/pdf/1412.6550.pdf)] [No Code] [[Link](https://zhuanlan.zhihu.com/p/542550895)]
- ICLR'17 Paying More Attention to Attention: Improving The Performance of Convolutional Neural Networks via Attention Transfer [[Paper](https://arxiv.org/pdf/1612.03928.pdf)] [[Code](https://github.com/szagoruyko/attention-transfer)] [[Link](https://zhuanlan.zhihu.com/p/542550895)]
- ICCV'19 Similarity-Preserving Knowledge Distillation [[Paper](https://arxiv.org/pdf/1907.09682.pdf)] [No Code] [[Link](https://zhuanlan.zhihu.com/p/542398176)]
- ICCV'19  Be Your Own Teacher: Improve the Performance of Convolutional Neural Networks via Self Distillation [[Paper](https://openaccess.thecvf.com/content_ICCV_2019/papers/Zhang_Be_Your_Own_Teacher_Improve_the_Performance_of_Convolutional_Neural_ICCV_2019_paper.pdf)] [[Code](https://github.com/luanyunteng/pytorch-be-your-own-teacher)] [[Link](https://zhuanlan.zhihu.com/p/545136524)]
- CVPR'21 Distilling Knowledge via Knowledge Review  [[Paper](https://arxiv.org/pdf/2104.09044.pdf)] [[Code](https://github.com/dvlab-research/ReviewKD)] [[Link](https://zhuanlan.zhihu.com/p/540706096)]
- CVPR'21 Distill on the Go: Online knowledge distillation in self-supervised learning [[Paper](https://openaccess.thecvf.com/content/CVPR2021W/LLID/papers/Bhat_Distill_on_the_Go_Online_Knowledge_Distillation_in_Self-Supervised_Learning_CVPRW_2021_paper.pdf)] [[Code](https://github.com/NeurAI-Lab/DoGo)] [[Link](https://zhuanlan.zhihu.com/p/545141466)]
- CVPR'22 Decoupled Knowledge Distillation [[Paper](https://arxiv.org/pdf/2203.08679.pdf)] [[Code](https://github.com/megvii-research/mdistiller)] [[Link](https://zhuanlan.zhihu.com/p/540345474)]

------



- CVPR'20 Distilling Knowledge from Graph Convolutional Networks [[Paper](https://arxiv.org/abs/2003.10477)] [[Code](https://github.com/ihollywhy/DistillGCN.PyTorch)] [[Link](https://zhuanlan.zhihu.com/p/522926912)]
- SIGMOD'20 Reliable Data Distillation on Graph Convolutional Network [[Paper](http://olivier.ruas.free.fr/papers/SIGMOD20.pdf)] [No Code] [[Link](https://zhuanlan.zhihu.com/p/524706697)]
- KDD'20 TinyGNN: Learning Efficient Graph Neural Networks [[Paper](https://www.kdd.org/kdd2020/accepted-papers/view/tinygnn-learning-efficient-graph-neural-networks)] [No Code] [[Link](https://zhuanlan.zhihu.com/p/534364243)] 
- arXiv’21 Distilling Self-Knowledge From Contrastive Links to Classify Graph Nodes Without Passing Messages [[Paper](https://arxiv.org/pdf/2106.08541.pdf)] [[Code](https://github.com/cf020031308/LinkDist)] [[Link](https://zhuanlan.zhihu.com/p/528128661)]
- MICCAI'21 GKD: Semi-supervised Graph Knowledge Distillation for Graph-Independent Inference [[Paper](https://arxiv.org/pdf/2104.03597.pdf)] [[Code](https://github.com/mahsa91/GKD)] [[Link](https://zhuanlan.zhihu.com/p/529988634)]
- CVPR'21 Bi-GCN: Binary Graph Convolutional Network [[Paper](https://arxiv.org/abs/2010.07565)] [[Code](https://github.com/bywmm/Bi-GCN)] [[Link](https://zhuanlan.zhihu.com/p/520825504)]
- IJCAI'21 Graph-Free Knowledge Distillation for Graph Neural Networks [[Paper](https://www.ijcai.org/proceedings/2021/0320.pdf)] [[Code](https://github.com/Xiang-Deng-DL/GFKD)] [[Link](https://zhuanlan.zhihu.com/p/526258612)]
- IJCAI'21 On Self-Distilling Graph Neural Network [[Paper](https://arxiv.org/abs/2011.02255)] [No Code] [[Link](https://zhuanlan.zhihu.com/p/522345993)]
- ICLR'21 On Graph Neural Networks versus Graph-Augmented MLPs [[Paperr](https://arxiv.org/pdf/2010.15116.pdf)] [[Code](https://github.com/leichen2018/GNN_vs_GAMLP)] [[Link](https://zhuanlan.zhihu.com/p/536284429)]
- WWW'21  Extract the Knowledge of Graph Neural Networks and Go Beyond it: An Effective Knowledge Distillation Framework  [[Paper](https://arxiv.org/abs/2103.02885)] [[Code](https://github.com/BUPT-GAMMA/CPF)] [[Link](https://zhuanlan.zhihu.com/p/523829567)]
- KDD'21 ROD: Reception-aware Online Distillation for Sparse Graphs [[Paper](https://arxiv.org/pdf/2107.11789.pdf)] [[Code](https://github.com/zwt233/ROD)] [[Link](https://zhuanlan.zhihu.com/p/524839956)]
- arXiv'22 On Representation Knowledge Distillation for Graph Neural Networks [[Paper](https://arxiv.org/pdf/2111.04964.pdf)] [No Code] [[Link](https://zhuanlan.zhihu.com/p/529728564)]
- AAAI'22 Workshop  Scalable Consistency Training for Graph Neural Networks via Self-Ensemble Self-Distillation [[Paper](https://arxiv.org/pdf/2110.06290.pdf)] [No Code] [[Link](https://zhuanlan.zhihu.com/p/529094761)]
- ICLR'22 Graph-less Neural Networks: Teaching Old MLPs New Tricks via Distillation [[Paper](https://arxiv.org/abs/2110.08727)] [[Code](https://github.com/snap-research/graphless-neural-networks)] [[Link](https://zhuanlan.zhihu.com/p/523383588)]
- KDD'22 Compressing Deep Graph Neural Networks via Adversarial Knowledge Distillation [[Paper](https://arxiv.org/pdf/2205.11678.pdf)] [[Code](https://github.com/MIRALab-USTC/GraphAKD)] [[Link](https://zhuanlan.zhihu.com/p/525223627)]
- ICLR'22 Cold Brew Distilling Graph Node Representations with Incomplete or Missing Neighborhoods [[Paper](https://arxiv.org/abs/2111.04840)] [[Code](https://github.com/amazon-research/gnn-tail-generalization)] [[Link](https://zhuanlan.zhihu.com/p/511394613)]
- WSDM'22 Deep Graph-level Anomaly Detection by Glocal Knowledge Distillation [[Paper](https://arxiv.org/pdf/2112.10063.pdf)] [[Code](https://github.com/RongrongMa/GLocalKD)] [[Link](https://zhuanlan.zhihu.com/p/542836644)]

## Efficient Architecture and Paradigm

- ICLR'19 How Powerful are Graph Neural Networks? [[Paper](https://arxiv.org/abs/1810.00826)] [[Code](https://github.com/weihua916/powerful-gnns)] [[Link](https://zhuanlan.zhihu.com/p/464818620)] 
- NeurIPS'20 Graph Random Neural Networks for Semi-Supervised Learning on Graphs [[Paper](https://arxiv.org/pdf/2005.11079.pdf)] [[Code](https://github.com/THUDM/GRAND)] [[Link](https://zhuanlan.zhihu.com/p/536767558)]
- arXiv'21 Graph Learning with 1D Convolutions on Random Walks [[Paper](https://arxiv.org/abs/2102.08786)] [[Code](https://github.com/toenshoff/CRaWl)] [[Link](https://zhuanlan.zhihu.com/p/434173732)]
- WSDM'21 Node Similarity Preserving Graph Convolutional Networks [[Paper](https://arxiv.org/pdf/2011.09643.pdf)] [[Code](https://github.com/ChandlerBang/SimP-GCN)] [[Link](https://zhuanlan.zhihu.com/p/536794358)] 
- ICLR'22 A New Perspective on "How Graph Neural Networks Go Beyond Weisfeiler-Lehman?" [[Paper](https://openreview.net/forum?id=uxgg9o7bI_3)] [No Code] [[Link](https://zhuanlan.zhihu.com/p/465200994)] 
- KDD'22 Model Degradation Hinders Deep Graph Neural Networks [[Paper](https://arxiv.org/pdf/2206.04361.pdf)] [[Code](https://github.com/zwt233/AIR)] [[Link](https://zhuanlan.zhihu.com/p/538767995)]
- KDD'22 Feature Overcorrelation in Deep Graph Neural Networks: A New Perspective [[Paper](https://arxiv.org/pdf/2206.07743.pdf)] [[Code](https://github.com/ChandlerBang/DeCorr)] [[Link](https://zhuanlan.zhihu.com/p/541796205)]

## Graph Data Augmentation

- KDD'20 NodeAug: Semi-Supervised Node Classification with Data Augmentation [[Paper](https://bhooi.github.io/papers/nodeaug_kdd20.pdf)] [No Code] [[Link](https://zhuanlan.zhihu.com/p/466885671)] 
- arXiv'21 Local Augmentation for Graph Neural Networks [[Paper](https://arxiv.org/pdf/2109.03856.pdf)] [No Code] [[Link](https://zhuanlan.zhihu.com/p/467800945)] 
- AAAI'21 Data Augmentation for Graph Neural Networks [[Paper](https://www.aaai.org/AAAI21Papers/AAAI-10012.ZhaoT.pdf)] [[Code](https://github.com/zhao-tong/GAug)] [[Link](https://zhuanlan.zhihu.com/p/466617745)] 
- WWW'21 Graph Contrastive Learning with Adaptive Augmentation [[Paper](https://arxiv.org/abs/2010.14945)] [[Code](https://github.com/CRIPAC-DIG/GCA)] [[Link](https://zhuanlan.zhihu.com/p/446868435)] 
- AAAI'22 Regularizing Graph Neural Networks via Consistency-Diversity Graph Augmentations [[Paper](http://shichuan.org/doc/126.pdf)] [No Code] [[Link](https://zhuanlan.zhihu.com/p/466661027)] 
- AAAI'21 GraphMix: Improved Training of GNNs for Semi-Supervised Learning [[Paper](https://arxiv.org/abs/1909.11715)] [[Code](https://github.com/vikasverma1077/GraphMix)] [[Link](https://zhuanlan.zhihu.com/p/467389235)] 
- CVPR'22 Robust Optimization as Data Augmentation for Large-scale Graphs [[Paper](https://openaccess.thecvf.com/content/CVPR2022/papers/Kong_Robust_Optimization_As_Data_Augmentation_for_Large-Scale_Graphs_CVPR_2022_paper.pdf)] [[Code](https://github.com/devnkong/FLAG)] [[Link](https://zhuanlan.zhihu.com/p/527957286)]
- AAAI'22 SAIL: Self-Augmented Graph Contrastive Learning [[Paper](https://arxiv.org/pdf/2009.00934.pdf)] [No Code] [[Link](https://zhuanlan.zhihu.com/p/538091329)]

## Imbalance Graph Neural Networks

- arXiv'20 Non-Local Graph Neural Networks [[Paper](https://arxiv.org/abs/2005.14612v1)] [No Code] [[Link](https://zhuanlan.zhihu.com/p/444884642)] 
- arXiv'20 Non-IID Graph Neural Networks [[Paper](https://arxiv.org/abs/2005.12386)] [No Code] [[Link](https://zhuanlan.zhihu.com/p/444278767)] 
- WSDM'21 GraphSMOTE: Imbalanced Node Classification on Graphs with Graph Neural Networks [[Paper](https://arxiv.org/abs/2103.08826)] [[Code](https://github.com/TianxiangZhao/GraphSmote)] [[Link](https://zhuanlan.zhihu.com/p/445035949)] 
- IJCAI'21 Multi-Class Imbalanced Graph Convolutional Network Learning [[Paper](https://www.ijcai.org/proceedings/2020/0398.pdf)] [[Code](https://github.com/codeshareabc/DRGCN)] [[Link](https://zhuanlan.zhihu.com/p/446314982)]  
- NeurIPS’21 Shift-Robust GNNs: Overcoming the Limitations of Localized Graph Training Data [[Paper](https://link.zhihu.com/?target=https%3A//proceedings.neurips.cc/paper/2021/file/eb55e369affa90f77dd7dc9e2cd33b16-Paper.pdf)] [[Code](https://github.com/GentleZhu/Shift-Robust-GNNs)] [[Link](https://zhuanlan.zhihu.com/p/522066981)]

# Federated Graph Learning

- Big Data'19 SGNN: A Graph Neural Network Based Federated Learning Approach by Hiding Structure [[Paper](https://ieeexplore.ieee.org/abstract/document/9005983)] [No Code] [[Link](https://zhuanlan.zhihu.com/p/430953193)] 

- arXiv'20 Federated Dynamic GNN with Secure Aggregation [[Paper](https://arxiv.org/abs/2009.07351)] [No Code] [[Link](https://zhuanlan.zhihu.com/p/441686576)] 
- arXiv'21 GIST: Distributed Training for Large-Scale Graph Convolutional Networks [[Paper](https://arxiv.org/abs/2102.10424)] [[Code](https://github.com/wolfecameron/GIST)] [[Link](https://zhuanlan.zhihu.com/p/433427134)]
- TSIPN'21 Distributed Training of Graph Convolutional Networks [[Paper](https://arxiv.org/abs/2007.06281)] [No Code] [[Link](https://zhuanlan.zhihu.com/p/433329525)] 
- ICLR'22 Learn Locally, Correct Globally: A Distributed Algorithm for Training Graph Neural Networks [[Paper](https://arxiv.org/pdf/2111.08202.pdf)] [[Code](https://github.com/MortezaRamezani/llcg)] [[Link](https://zhuanlan.zhihu.com/p/536158540)]

## Personalized and Heterogeneous Federated Learning in CV or NLP

- NeurIPS'18 Workshop Communication-Efficient On-Device Machine Learning: Federated Distillation and Augmentation under Non-IID Private Data [[Paper](https://arxiv.org/abs/1811.11479)] [No Code] [[Link](https://zhuanlan.zhihu.com/p/535652457)]
- NeurIPS'19 Think Locally, Act Globally: Federated Learning with Local and Global Representations [[Paper](https://arxiv.org/pdf/2001.01523.pdf)] [[Code](https://github.com/pliang279/LG-FedAvg)] [[Link](https://zhuanlan.zhihu.com/p/497030361)]
- arXiv'20 Adaptive Personalized Federated Learning [[Paper](https://arxiv.org/abs/2003.13461)] [No Code] [[Link](https://zhuanlan.zhihu.com/p/497269102)]
- AAAI'21 Addressing Class Imbalance in Federated Learning [[Paper](https://arxiv.org/abs/2008.06217)] [[Code](https://github.com/balanced-fl/Addressing-Class-Imbalance-FL)] [[Link](https://zhuanlan.zhihu.com/p/443009189)] 

## Theoretical Analysis of Federated Learning in CV or NLP

- JMLR'17 Communication-Efficient Learning of Deep Networks from Decentralized Data [[Paper](http://proceedings.mlr.press/v54/mcmahan17a/mcmahan17a.pdf)] [[Code](https://github.com/AshwinRJ/Federated-Learning-PyTorch)] [[Link](https://zhuanlan.zhihu.com/p/429370255)]
- arXiv'19 Detailed comparison of communication efficiency of split learning and federated learning [[Paper](https://arxiv.org/pdf/1909.09145.pdf)] [[Link](https://zhuanlan.zhihu.com/p/435255850)] 
- ICLR'20 On the Convergence of FedAvg on Non-IID Data [[Paper](https://arxiv.org/abs/1907.02189)] [[Code](https://github.com/lx10077/fedavgpy)] [[Link](https://zhuanlan.zhihu.com/p/500005337)]

## CV or NLP Model Compression and Quantification in Federated Learning 

- NeurIPS'19 Workshop FedMD: Heterogenous Federated Learning via Model Distillation [[Paper](https://arxiv.org/pdf/1910.03581.pdf)] [[Code](https://github.com/diogenes0319/FedMD_clean)] [[Link](https://zhuanlan.zhihu.com/p/535687915?)]
- NeurIPS'20 Group Knowledge Transfer: Federated Learning of Large CNNs at the Edge [[Paper](https://arxiv.org/pdf/2007.14513.pdf)] [[Code](https://fedml.ai/)] [[Link](https://zhuanlan.zhihu.com/p/536901871)]
- arXiv'22 CDKT-FL: Cross-Device Knowledge Transfer using Proxy Dataset in Federated Learning [[Paper](https://arxiv.org/pdf/2204.01542.pdf)] [[Code](https://github.com/Agent2H/CDKT_FL)] [[Link](https://zhuanlan.zhihu.com/p/528950968)]

## Transfer Federated Graph Learning and Graph Structure Federated Learning

- arXiv'19 Peer-to-Peer Federated Learning on Graphs [[Paper](https://arxiv.org/abs/1901.11173)] [No Code] [[Link](https://zhuanlan.zhihu.com/p/441944011)]
- ICML'21 SpreadGNN: Serverless Multi task Federated Learning for Graph Neural Networks [[Paper](https://arxiv.org/abs/2106.02743)] [[Code](https://github.com/FedML-AI/SpreadGNN)] [[Link](https://zhuanlan.zhihu.com/p/429720860)] 
- PPNA'21 ASFGNN: Automated Separated-Federated Graph Neural Network [[Paper](https://arxiv.org/abs/2011.03248)] [No Code] [[Link](https://zhuanlan.zhihu.com/p/431283541)]
- IJCAI'21 Decentralized Federated Graph Neural Networks [[Paper](https://federated-learning.org/fl-ijcai-2021/FTL-IJCAI21_paper_20.pdf)] [No Code] [[Link](https://zhuanlan.zhihu.com/p/430508567)]
- CVPR'21 Cluster-driven Graph Federated Learning over Multiple Domains [[Paper](https://arxiv.org/abs/2104.14628)] [No Code] [[Link](https://zhuanlan.zhihu.com/p/440527314)]
- ICML'21 Personalized Federated Learning using Hypernetworks [[Paper](https://arxiv.org/abs/2103.04628)] [[Code](https://github.com/AvivSham/pFedHN)] [[Link](https://zhuanlan.zhihu.com/p/431130945)] 

## Intra-Graph Horizontal Federated Learning

- arXiv'20 GraphFL: A Federated Learning Framework for Semi-Supervised Node Classification on Graphs [[Paper](https://arxiv.org/abs/2012.04187)] [No Code] [[Link](https://zhuanlan.zhihu.com/p/431479904)] 
- NeurIPS'21 Subgraph Federated Learning with Missing Neighbor Generation [[Paper](https://arxiv.org/abs/2106.13430)] [[Code](https://github.com/zkhku/fedsage)] [[Link](https://zhuanlan.zhihu.com/p/430789355)] 
- ICML'21 FedGNN: Federated Graph Neural Network for Privacy-Preserving Recommendation [[Paper](https://arxiv.org/abs/2102.04925)] [No Code] [[Link](https://zhuanlan.zhihu.com/p/428783383)]
- arXiv'21 FedGL: Federated Graph Learning Framework with Global Self-Supervision [[Paper](https://arxiv.org/abs/2105.03170)] [No Code] [[Link](https://zhuanlan.zhihu.com/p/431049080)]
- KDD'21 Cross-Node Federated Graph Neural Network for Spatio-Temporal Data Modeling [[Paper](https://arxiv.org/abs/2106.05223)] [[Code](https://github.com/mengcz13/KDD2021_CNFGNN)] [[Link](https://zhuanlan.zhihu.com/p/434839878)] 
- CIKM'21 Federated Knowledge Graphs Embedding [[Paper](https://arxiv.org/abs/2105.07615)] [[Code](https://github.com/HKUST-KnowComp/FKGE)] [[Link](https://zhuanlan.zhihu.com/p/437895959)] 

## Inter-Graph Horizontal Federated Learning

- NeurIPS'21 Federated Graph Classification over Non-IID Graphs [[Paper](https://arxiv.org/abs/2106.13423)] [[Code](https://github.com/Oxfordblue7/GCFL)] [[Link](https://zhuanlan.zhihu.com/p/430623053)] 

## Vertical Federal Learning

- arXiv'21 A Vertical Federated Learning Framework for Graph Convolutional Network [[Paper](https://arxiv.org/abs/2106.11593)] [No Code] [[Link](https://zhuanlan.zhihu.com/p/431900470)] 
- arXiv'21 Vertically Federated Graph Neural Network for Privacy-Preserving Node Classification [[Paper](https://arxiv.org/abs/2005.11903)] [No Code] [[Link](https://zhuanlan.zhihu.com/p/428406637)]

# Privacy Graph Neural Networks

- SIGSAC'16 Deep Learning with Differential Privacy [[Paper](https://arxiv.org/abs/1607.00133v1)] [[Code](https://github.com/lingyunhao/Deep-Learning-with-Differential-Privacy)] [[Link](https://zhuanlan.zhihu.com/p/419216660)]
- ICLR'17 Semi-supervised Knowledge Transfer for Deep Learning from Private Training Data [[Paper](https://arxiv.org/abs/1610.05755)] [[Code](https://github.com/kamathhrishi/PATE)] [[Link](https://zhuanlan.zhihu.com/p/423009101)]
- ICLR'18 Scalable Private Learning With PATE [[Paper](https://arxiv.org/abs/1802.08908)] [[Code](https://github.com/kamathhrishi/PATE)] [[Link](https://zhuanlan.zhihu.com/p/423063552)]
- arXiv'20  When Differential Privacy Meets Graph Neural Networks [[Paper](https://arxiv.org/pdf/2006.05535v1.pdf)]  [[Code](https://github.com/sisaman/LPGNN)]  [[Link](https://zhuanlan.zhihu.com/p/423868946)]
- arXiv'21 Releasing Graph Neural Networks with Differential Privacy [[Paper](https://arxiv.org/abs/2109.08907)] [No Code] [[Link](https://zhuanlan.zhihu.com/p/424463019)]
- arXiv'21 A Graph Federated Architecture with Privacy Preserving Learning [[Paper](https://arxiv.org/abs/2104.13215)] [No Code] [[Link](https://zhuanlan.zhihu.com/p/433728803)]
- IJCAI'21 Secure Deep Graph Generation with Link Differential Privacy [[Paper](https://arxiv.org/abs/2005.00455v3)] [[Code](https://github.com/haonan3/Secure-Network-Release-with-Link-Privacy)] [[Link](https://zhuanlan.zhihu.com/p/417555475)]
- CCS'21 Locally Private Graph Neural Networks [[Paper](https://arxiv.org/pdf/2006.05535.pdf)] [[Code](https://github.com/sisaman/LPGNN)] [[Link](https://zhuanlan.zhihu.com/p/423444455)]

# Survey and Framework Toolkits

- Graph library -- PyG、GarphGallery [[Link](https://zhuanlan.zhihu.com/p/420587332)]

- Graph library -- DIG、AutoGL、CogDL [[Link](https://zhuanlan.zhihu.com/p/422082239)]

- PyTorch Geometric（一）：数据加载 [[Link](https://zhuanlan.zhihu.com/p/425974734)]

- PyTorch Geometric（二）：模型搭建 [[Link](https://zhuanlan.zhihu.com/p/427083823)]

- 基于 GNN 的隐私计算（联邦学习）Review（二）[[Link](https://zhuanlan.zhihu.com/p/432071253)]

- 基于 GNN 的隐私计算（联邦学习）Review（三）[[Link](https://zhuanlan.zhihu.com/p/432126858)]

- introduction [[Link](https://zhuanlan.zhihu.com/p/416264898)]

- Local Differential Privacy: a tutorial [[Paper](https://arxiv.org/abs/1907.11908)] [[Link](https://zhuanlan.zhihu.com/p/416556008)]

- 本地化差分隐私研究综述 [[Paper](https://wenku.baidu.com/view/ca901cf8876fb84ae45c3b3567ec102de3bddf84?fr=xueshu)] [[Link](https://zhuanlan.zhihu.com/p/417209747)]

- 差分隐私 -- Laplace mechanism、Gaussian mechanism、Composition theorem [[Link](https://zhuanlan.zhihu.com/p/425732159)]

- 矩母函数 GMF 及矩的概念 -- 期望、方差、归一化矩、偏态、峰度 [[Link](https://zhuanlan.zhihu.com/p/425898950)] [[Reference](https://towardsdatascience.com/moment-generating-function-explained-27821a739035)]

- Moments Accountant 的理解 [[Link](https://zhuanlan.zhihu.com/p/425780267)] [[Reference](https://zhuanlan.zhihu.com/p/264779199)]

- 基于 GNN 的隐私计算（差分隐私）Review（一）[[Link](https://zhuanlan.zhihu.com/p/426267637)]

- Federated Machine Learning: Concept and Applications [[Paper](https://arxiv.org/abs/1902.04885)] [[Link](https://zhuanlan.zhihu.com/p/427770121)]

- arXiv'21 Graph4Rec: A Universal Toolkit with Graph Neural Networks for Recommender Systems [[Paper](https://arxiv.org/abs/2112.01035)] [[Code](https://github.com/PaddlePaddle/PGL/tree/main/apps/Graph4Rec)] [[Link](https://zhuanlan.zhihu.com/p/443243204)]

- arXiv'21 Federated Graph Learning - A Position Paper [[Paper](https://arxiv.org/abs/2105.11099)] [[Link](https://zhuanlan.zhihu.com/p/431934452)]

- arXiv'21 Federated Learning on Non-IID Data Silos: An Experimental Study [[Paper](https://arxiv.org/abs/2102.02079)] [[Code](https://github.com/Xtra-Computing/NIID-Bench)] [[Link](https://zhuanlan.zhihu.com/p/439561676)] 

- ICLR'21 FedGraphNN: A Federated Learning Benchmark System for Graph Neural Networks [[Paper](https://arxiv.org/abs/2104.07145)] [[Code](https://github.com/FedML-AI/FedGraphNN)] [[Link](https://zhuanlan.zhihu.com/p/429220636)] 

- arXiv‘22 Data Augmentation for Deep Graph Learning: A Survey  [[Paper](https://arxiv.org/pdf/2202.08235.pdf)] [No Code] [[Link](https://zhuanlan.zhihu.com/p/492146847)]

- arXiv'22 A Survey on Graph Structure Learning: Progress and Opportunities [[Paper](https://arxiv.org/pdf/2103.03036.pdf)] [No Code] [[Link](https://zhuanlan.zhihu.com/p/470614896)]

  

  

