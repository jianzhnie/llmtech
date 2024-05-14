## 附录 A 贝尔曼方程

### 定理 A.1. 贝尔曼方程（将 $Q_{\pi}$ 表示成 $Q_{\pi}$ )

假设 $R_{t}$ 是 $S_{t} 、 A_{t} 、 S_{t+1}$ 的函数。那么

$$
Q_{\pi}\left(s_{t}, a_{t}\right)=\mathbb{E}_{S_{t+1}, A_{t+1}}\left[R_{t}+\gamma \cdot Q_{\pi}\left(S_{t+1}, A_{t+1}\right) \mid S_{t}=s_{t}, A_{t}=a_{t}\right]
$$

证明 根据回报的定义 $U_{t}=\sum_{k=t}^{n} \gamma^{k-t} \cdot R_{k}$, 不难验证这个等式:

$$
U_{t}=R_{t}+\gamma \cdot U_{t+1} .
$$

用符号 $\mathcal{S}_{t+1:}=\left\{S_{t+1}, S_{t+2}, \cdots\right\}$ 和 $\mathcal{A}_{t+1:}=\left\{A_{t+1}, A_{t+2}, \cdots\right\}$ 表示从 $t+1$ 时刻起所有的 状态和动作随机变量。根据动作价值函数 $Q_{\pi}$ 的定义,

$$
Q_{\pi}\left(s_{t}, a_{t}\right)=\mathbb{E}_{\mathcal{S}_{t+1:}, \mathcal{A}_{t+1}:}\left[U_{t} \mid S_{t}=s_{t}, A_{t}=a_{t}\right] .
$$

把 $U_{t}$ 替换成 $R_{t}+\gamma \cdot U_{t+1}$, 那么

$$
\begin{aligned}
& Q_{\pi}\left(s_{t}, a_{t}\right)=\mathbb{E}_{\mathcal{S}_{t+1:}, \mathcal{A}_{t+1:}:}\left[R_{t}+\gamma \cdot U_{t+1} \mid S_{t}=s_{t}, A_{t}=a_{t}\right] \\
& =\mathbb{E}_{\mathcal{S}_{t+1:}, \mathcal{A}_{t+1:}:}\left[R_{t} \mid S_{t}=s_{t}, A_{t}=a_{t}\right]+\gamma \cdot \mathbb{E}_{\mathcal{S}_{t+1:}, \mathcal{A}_{t+1}:}\left[U_{t+1} \mid S_{t}=s_{t}, A_{t}=a_{t}\right] .
\end{aligned}
$$

假设 $R_{t}$ 是 $S_{t} 、 A_{t} 、 S_{t+1}$ 的函数。那么, 给定 $s_{t}$ 和 $a_{t}$, 则 $R_{t}$ 随机性唯一的来源就是 $S_{t+1}$, 所以

$$
\mathbb{E}_{\mathcal{S}_{t+1:}, \mathcal{A}_{t+1:}}\left[R_{t} \mid S_{t}=s_{t}, A_{t}=a_{t}\right]=\mathbb{E}_{S_{t+1}}\left[R_{t} \mid S_{t}=s_{t}, A_{t}=a_{t}\right] .
$$

等式 (A.1) 右边 $U_{t+1}$ 的期望可以写成

$$
\begin{aligned}
& \mathbb{E}_{\mathcal{S}_{t+1:}: \mathcal{A}_{t+1}:}\left[U_{t+1} \mid S_{t}=s_{t}, A_{t}=a_{t}\right] \\
& =\mathbb{E}_{S_{t+1}, A_{t+1}}\left[\mathbb{E}_{\mathcal{S}_{t+2:}, \mathcal{A}_{t+2:}:}\left[U_{t+1} \mid S_{t+1}, A_{t+1}\right] \mid S_{t}=s_{t}, A_{t}=a_{t}\right] \\
& =\mathbb{E}_{S_{t+1}, A_{t+1}}\left[Q_{\pi}\left(S_{t+1}, A_{t+1}\right) \mid S_{t}=s_{t}, A_{t}=a_{t}\right] .
\end{aligned}
$$

由公式 (A.1)、(A.2)、(A.3) 可得定理。

### 定理 A.2.贝尔曼方程（将 $Q_{\pi}$ 表示成 $V_{\pi}$ ）

假设 $R_{t}$ 是 $S_{t} 、 A_{t} 、 S_{t+1}$ 的函数。那么

$$
Q_{\pi}\left(s_{t}, a_{t}\right)=\mathbb{E}_{S_{t+1}}\left[R_{t}+\gamma \cdot V_{\pi}\left(S_{t+1}\right) \mid S_{t}=s_{t}, A_{t}=a_{t}\right] .
$$

证明 由于 $V_{\pi}\left(S_{t+1}\right)=\mathbb{E}_{A_{t+1}}\left[Q\left(S_{t+1}, A_{t+1}\right)\right]$, 由定理 A.1 可得定理 A.2。

### 定理 A.3.贝尔曼方程 (将 $V_{\pi}$ 表示成 $V_{\pi}$ )

假设 $R_{t}$ 是 $S_{t} 、 A_{t} 、 S_{t+1}$ 的函数。那么

$$
V_{\pi}\left(s_{t}\right)=\mathbb{E}_{A_{t}, S_{t+1}}\left[R_{t}+\gamma \cdot V_{\pi}\left(S_{t+1}\right) \mid S_{t}=s_{t}\right] .
$$

证明 由于 $V_{\pi}\left(S_{t}\right)=\mathbb{E}_{A_{t}}\left[Q\left(S_{t}, A_{t}\right)\right]$, 由定理 A.2 可得定理 A.3。

### 定理 A.4. 最优贝尔曼方程

假设 $R_{t}$ 是 $S_{t} 、 A_{t} 、 S_{t+1}$ 的函数。那么

$$
Q_{\star}\left(s_{t}, a_{t}\right)=\mathbb{E}_{S_{t+1} \sim p\left(\cdot \mid s_{t}, a_{t}\right)}\left[R_{t}+\gamma \cdot \max _{A \in \mathcal{A}} Q_{\star}\left(S_{t+1}, A\right) \mid S_{t}=s_{t}, A_{t}=a_{t}\right] .
$$

![](https://cdn.mathpix.com/cropped/2023_02_03_f46f5cf0e4de5b9996dcg-312.jpg?height=60&width=1410&top_left_y=650&top_left_x=366)

$$
Q_{\pi^{\star}}\left(s_{t}, a_{t}\right)=\mathbb{E}_{S_{t+1}, A_{t+1}}\left[R_{t}+\gamma \cdot Q_{\pi^{\star}}\left(S_{t+1}, A_{t+1}\right) \mid S_{t}=s_{t}, A_{t}=a_{t}\right] .
$$

根据定义, 最优动作价值函数是

$$
Q_{\star}(s, a) \triangleq \max _{\pi} Q_{\pi}(s, a), \quad \forall s \in \mathcal{S}, \quad a \in \mathcal{A} .
$$

所以 $Q_{\pi^{\star}}(s, a)$ 就是 $Q_{\star}(s, a)$ 。于是

$$
Q_{\star}\left(s_{t}, a_{t}\right)=\mathbb{E}_{S_{t+1}, A_{t+1}}\left[R_{t}+\gamma \cdot Q_{\star}\left(S_{t+1}, A_{t+1}\right) \mid S_{t}=s_{t}, A_{t}=a_{t}\right] .
$$

因为动作 $A_{t+1}=\operatorname{argmax}_{A} Q_{\star}\left(S_{t+1}, A\right)$ 是状态 $S_{t+1}$ 的确定性函数, 所以

$$
Q_{\star}\left(s_{t}, a_{t}\right)=\mathbb{E}_{S_{t+1}}\left[R_{t}+\gamma \cdot \max _{A \in \mathcal{A}} Q_{\star}\left(S_{t+1}, A\right) \mid S_{t}=s_{t}, A_{t}=a_{t}\right] .
$$



## 参考文献

[1] P. Abbeel, A. Y. Ng. Apprenticeship learning via inverse reinforcement learning. C. E. Brodley, editor, 21st International Conference of Machine Learning, Banff, Alberta, Canada, 2004. ACM, 2004

[2] M. S. Abdulla, S. Bhatnagar. Reinforcement learning based algorithms for average cost markov decision processes. Discrete Event Dynamic Systems, 2007. 17(1): 23 52

[3] Z. Ahmed, N. Le Roux, M. Norouzi, D. Schuurmans. Understanding the impact of entropy on policy optimization. International Conference on Machine Learning (ICML). 2019

[4] L. V. Allis, et al. Searching for Solutions in Games and Artificial Intelligence. Ponsen \& Looijen, 1994

[5] B. D. Argall, S. Chernova, M. Veloso, B. Browning. A survey of robot learning from demonstration. Robotics and Autonomous Systems, 2009. 57(5): 469 483

[6] D. Bahdanau, K. Cho, Y. Bengio. Neural machine translation by jointly learning to align and translate. International Conference on Learning Representations (ICLR). 2015

[7] M. Bain, C. Sammut. A framework for behavioural cloning. Machine Intelligence, Oxford, UK, 1995. Oxford, England: Oxford University Press, 1995

[8] N. Bard, J. N. Foerster, S. Chandar, N. Burch, M. Lanctot, H. F. Song, E. Parisotto, V. Dumoulin, S. Moitra, E. Hughes, et al. The hanabi challenge: A new frontier for ai research. Artificial Intelligence, 2020. 280: 103216

[9] A. G. Barto, R. S. Sutton, C. W. Anderson. Neuronlike adaptive elements that can solve difficult learning control problems. IEEE transactions on systems, man, and cybernetics, 1983. (5): 834 846

[10] P. Baudis, J. Gailly. Pachi: State of the art open source go program. 13th International Conference of Advances in Computer Game, Tilburg, The Netherlands, 2011. New York City: Springer, 2011

[11] M. G. Bellemare, W. Dabney, R. Munos. A distributional perspective on reinforcement learning. 34th International Conference on Machine Learning, Sydney, NSW, Australia, 2017. PMLR, 2017

[12] D. P. Bertsekas. Constrained Optimization and Lagrange Multiplier Methods. Cambridge, Massachusetts: Academic Press, 2014

[13] S. Bhatnagar, S. Kumar. A simultaneous perturbation stochastic approximation-based actor-critic algorithm for markov decision processes. IEEE Transactions on Automatic Control, 2004. 49(4): 592 598

[14] S. Bhatnagar, R. S. Sutton, M. Ghavamzadeh, M. Lee. Natural actor-critic algorithms. Automatica, 2009. 45(11): 2471 2482

[15] A. Boularias, J. Kober, J. Peters. Relative entropy inverse reinforcement learning. 14th International Conference on Artificial Intelligence and Statistics, Fort Lauderdale, USA, 2011. JMLR.org, 2011

[16] C. Boutilier. Planning, learning and coordination in multiagent decision processes. Y. Shoham, editor, Proceedings of the Sixth Conference on Theoretical Aspects of Rationality and Knowledge, De Zeeuwse Stromen, The Netherlands, March 17-20 1996. Morgan Kaufmann, 1996 195-210

[17] B. Bouzy, B. Helmstetter. Monte-carlo go developments. 10th International Conference of Advances in Computer Games, Graz, Austria, 2003. Philadelphia, United States: Kluwer, 2003

[18] S. Boyd, L. Vandenberghe. Convex Optimization. Cambridge, England: Cambridge University Press, 2004

[19] I. Bratko, T. Urbancic. Transfer of control skill by machine learning. Engineering Applications of Artificial Intelligence, 1997. 10(1): 63 71

[20] C. B. Browne, E. Powley, D. Whitehouse, S. M. Lucas, P. I. Cowling, P. Rohlfshagen, S. Tavener, D. Perez, S. Samothrakis, S. Colton. A survey of monte carlo tree search methods. IEEE Transactions on Computational Intelligence and AI in Games, 2012. 4(1): 1 43

[21] M. Buro. From simple features to sophisticated evaluation functions. First International Conference of Computers and Games, Tsukuba, Japan, 1998. New York City: Springer, 1998

[22] L. Buşoniu, R. Babuška, B. De Schutter. Multi-agent reinforcement learning: An overview. Innovations in multi-agent systems and applications-1, 183-221. Springer, 2010

[23] M. Campbell, A. J. Hoane Jr, F.-h. Hsu. Deep blue. Artificial Intelligence, 2002. 134(1 2): 57 83

[24] G. Chaslot, S. Bakkes, I. Szita, P. Spronck. Monte-carlo tree search: A new framework for game ai. 4th Artificial Intelligence and Interactive Digital Entertainment Conference, Stanford, California, 2008. Palo Alto, California: The AAAI Press, 2008

[25] G. Chaslot, J.-T. Saito, B. Bouzy, J. Uiterwijk, H. J. Van Den Herik. Monte-carlo strategies for computer go. 18th BeNeLux Conference on Artificial Intelligence, Namur, Belgium, 2006. Namen: University of Namur, 2006

[26] G. M. J.-B. C. Chaslot. Monte-Carlo Tree Search. Maastricht: Maastricht University, 2010

[27] M. Chen, A. Beutel, P. Covington, S. Jain, F. Belletti, E. H. Chi. Top-k off-policy correction for a reinforce recommender system. 12th ACM International Conference on Web Search and Data Mining, Melbourne, VIC, Australia, 2019. New York City: ACM, 2019

[28] K. Cho, B. van Merrienboer, Çaglar Gülçehre, D. Bahdanau, F. Bougares, H. Schwenk, Y. Bengio. Learning phrase representations using rnn encoder-decoder for statistical machine translation. EMNLP. 2014

[29] Y. Chow, O. Nachum, M. Ghavamzadeh. Path consistency learning in Tsallis entropy regularized mdps. International Conference on Machine Learning (ICML). 2018 979-988

[30] A. R. Conn, N. I. Gould, P. L. Toint. Trust Region Methods. Philadelphia, Pennsylvania, United States: SIAM, 2000

[31] R. Coulom. Efficient selectivity and backup operators in monte-carlo tree search. 5th International Conference of Computers and Games, Turin, Italy, 2006. New York City: Springer, 2006

[32] R. Coulom. Computing "elo ratings" of move patterns in the game of go. Journal of the International Computer Games Association, 2007. 30(4): 198 208

[33] J. Dean, S. Ghemawat. Mapreduce: Simplified data processing on large clusters. Communications of the ACM, 2008. 51(1): 107-113

[34] T. Degris, P. M. Pilarski, R. S. Sutton. Model-free reinforcement learning with continuous action in practice. American Control Conference (ACC). 2012

[35] M. Enzenberger, M. Müller, B. Arneson, R. Segal. Fuego: An open-source framework for board games and go engine based on monte carlo tree search. IEEE Transactions on Computational Intelligence and AI in Games, 2010. 2(4): 259 270

[36] W. Fedus, P. Ramachandran, R. Agarwal, Y. Bengio, H. Larochelle, M. Rowland, W. Dabney. Revisiting fundamentals of experience replay. 37th International Conference on Machine Learning, Virtual Event, 2020. PMLR, 2020

[37] C. Finn, S. Levine, P. Abbeel. Guided cost learning: Deep inverse optimal control via policy optimization. 33rd International Conference on Machine Learning, New York City, NY, USA, 2016. JMLR.org, 2016

[38] J. Foerster, G. Farquhar, T. Afouras, N. Nardelli, S. Whiteson. Counterfactual multi-agent policy gradients. AAAI Conference on Artificial Intelligence. 2018

[39] J. Foerster, N. Nardelli, G. Farquhar, T. Afouras, P. H. Torr, P. Kohli, S. Whiteson. Stabilising experience replay for deep multi-agent reinforcement learning. International Conference on Machine Learning (ICML). 2017

[40] M. Fortunato, M. G. Azar, B. Piot, J. Menick, M. Hessel, I. Osband, A. Graves, V. Mnih, R. Munos, D. Hassabis, O. Pietquin, C. Blundell, S. Legg. Noisy networks for exploration. 6th International Conference on Learning Representations, Vancouver, BC, Canada, 2018. OpenReview.net, 2018

[41] S. Fujimoto, H. van Hoof, D. Meger. Addressing function approximation error in actor-critic methods. 35th International Conference on Machine Learning, Stockholmsmässan, Stockholm, Sweden, 2018. PMLR.org, 2018

[42] I. J. Goodfellow, J. Pouget-Abadie, M. Mirza, B. Xu, D. Warde-Farley, S. Ozair, A. C. Courville, Y. Bengio. Generative adversarial nets. 28th Annual Conference on Neural Information Processing Systems, Montreal, Quebec, Canada, 2014. 2014

[43] J. K. Gupta, M. Egorov, M. Kochenderfer. Cooperative multi-agent control using deep reinforcement learning. International Conference on Autonomous Agents and Multiagent Systems. 2017

[44] T. Haarnoja, H. Tang, P. Abbeel, S. Levine. Reinforcement learning with deep energy-based policies. International Conference on Machine Learning (ICML). 2017

[45] R. Hafner, M. Riedmiller. Reinforcement learning in feedback control. Machine Learning, 2011. 84(1 2): $137 \sim 169$

[46] M. Hausknecht, P. Stone. Deep recurrent Q-learning for partially observable MDPs. AAAI Fall Symposium on Sequential Decision Making for Intelligent Agents. 2015

[47] P. Henderson, R. Islam, P. Bachman, J. Pineau, D. Precup, D. Meger. Deep reinforcement learning that matters. 32nd AAAI Conference on Artificial Intelligence, New Orleans, Louisiana, USA, 2018. Palo Alto, California: AAAI Press, 2018

[48] M. Hessel, J. Modayil, H. Van Hasselt, T. Schaul, G. Ostrovski, W. Dabney, D. Horgan, B. Piot, M. Azar, D. Silver. Rainbow: Combining improvements in deep reinforcement learning. Proceedings of the AAAI Conference on Artificial Intelligence, volume 32. 2018

[49] J. Ho, S. Ermon. Generative adversarial imitation learning. 29th Annual Conference on Neural Information Processing Systems, Barcelona, Spain, 2016. 2016

[50] Y.-C. Ho. Team decision theory and information structures. Proceedings of the IEEE, 1980. 68(6): 644 654

[51] S. Hochreiter, J. Schmidhuber. Long short-term memory. Neural Computation, 1997. 9(8): 1735 1780

[52] J. J. Hopfield. Neural networks and physical systems with emergent collective computational abilities. Proceedings of the National Academy of Sciences, 1982. 79(8): 2554 2558

[53] J. Hu, M. P. Wellman. Nash q-learning for general-sum stochastic games. Journal of Machine Learning Research, 2003. 4(Nov): 1039 1069

[54] Y. Hu, Q. Da, A. Zeng, Y. Yu, Y. Xu. Reinforcement learning to rank in e-commerce search engine: Formalization, analysis, and application. 24th ACM SIGKDD International Conference on Knowledge Discovery \& Data Mining, London, UK, 2018. New York City: ACM, 2018

[55] L. C. B. III. Residual algorithms: Reinforcement learning with function approximation. 12th International Conference on Machine Learning, Tahoe City, California, 1995. Burlington, Massachusetts: Morgan Kaufmann, 1995

[56] S. Iqbal, F. Sha. Actor-attention-critic for multi-agent reinforcement learning. International Conference on Machine Learning (ICML). 2019

[57] T. Jaakkola, M. I. Jordan, S. P. Singh. On the convergence of stochastic iterative dynamic programming algorithms. Neural Computation, 1994. 6(6): 1185 1201

[58] Y. Keneshloo, T. Shi, N. Ramakrishnan, C. K. Reddy. Deep reinforcement learning for sequence-to-sequence models. IEEE Transactions on Neural Networks and Learning Systems, 2019. 31(7): 2469 2489

[59] L. Kocsis, C. Szepesvári. Bandit based monte-carlo planning. 17th European Conference on Machine Learning, Berlin, Germany, 2006. New York City: Springer, 2006

[60] V. R. Konda, J. N. Tsitsiklis. Actor-critic algorithms. Advances in Neural Information Processing Systems (NIPS). 2000

[61] M. G. Lagoudakis, R. E. Parr. Learning in zero-sum team markov games using factored value functions. Advances in Neural Information Processing Systems. 2002

[62] M. Lauer, M. Riedmiller. An algorithm for distributed reinforcement learning in cooperative multi-agent systems. International Conference on Machine Learning. 2000

[63] K. Lee, S. Choi, S. Oh. Sparse markov decision processes with causal sparse tsallis entropy regularization for reinforcement learning. IEEE Robotics and Automation Letters, 2018. 3(3): 1466 1473

[64] S. Levine, V. Koltun. Continuous inverse optimal control with locally optimal examples. 29th International Conference on Machine Learning, Edinburgh, Scotland, UK, 2012. icml.cc / Omnipress, 2012 [65] S. Levine, A. Kumar, G. Tucker, J. Fu. Offline reinforcement learning: Tutorial, review, and perspectives on open problems. ArXiv, 2020. abs/2005.01643

[66] M. Li, D. G. Andersen, J. W. Park, A. J. Smola, A. Ahmed, V. Josifovski, J. Long, E. J. Shekita, B.-Y. Su. Scaling distributed machine learning with the parameter server. USENIX Symposium on Operating Systems Design and Implementation (OSDI). 2014

[67] T. P. Lillicrap, J. J. Hunt, A. Pritzel, N. Heess, T. Erez, Y. Tassa, D. Silver, D. Wierstra. Continuous control with deep reinforcement learning. 4th International Conference on Learning Representations, San Juan, Puerto Rico, 2016. ICLR.org, 2016

[68] L.-J. Lin. Reinforcement learning for robots using neural networks. Technical report, Carnegie-Mellon Univ Pittsburgh PA School of Computer Science, 1993

[69] M. L. Littman. Markov games as a framework for multi-agent reinforcement learning. International Conference on Machine Learning (ICML). 1994

[70] M. L. Littman. Friend-or-foe Q-learning in general-sum games. International Conference on Machine Learning (ICML). 2001

[71] H. Liu, K. Simonyan, Y. Yang. Darts: Differentiable architecture search. 7th International Conference on Learning Representations, New Orleans, LA, USA, 2019. OpenReview.net, 2019

[72] R. Lowe, Y. I. Wu, A. Tamar, J. Harb, O. P. Abbeel, I. Mordatch. Multi-agent actor-critic for mixed cooperativecompetitive environments. Advances in Neural Information Processing Systems (NIPS). 2017

[73] P. Marbach, J. N. Tsitsiklis. Simulation-based optimization of Markov reward processes: Implementation issues. IEEE Conference on Decision and Control. 1999

[74] P. W. Mirowski, R. Pascanu, F. Viola, H. Soyer, A. Ballard, A. Banino, M. Denil, R. Goroshin, L. Sifre, K. Kavukcuoglu, D. Kumaran, R. Hadsell. Learning to navigate in complex environments. 5th International Conference on Learning Representations, Toulon, France, 2017. OpenReview.net, 2017

[75] V. Mnih, A. P. Badia, M. Mirza, A. Graves, T. Lillicrap, T. Harley, D. Silver, K. Kavukcuoglu. Asynchronous methods for deep reinforcement learning. International Conference on Machine Learning (ICML). 2016

[76] V. Mnih, K. Kavukcuoglu, D. Silver, A. Graves, I. Antonoglou, D. Wierstra, M. A. Riedmiller. Playing atari with deep reinforcement learning. ArXiv, 2013. abs/1312.5602

[77] V. Mnih, K. Kavukcuoglu, D. Silver, A. A. Rusu, J. Veness, M. G. Bellemare, A. Graves, M. A. Riedmiller, A. Fidjeland, G. Ostrovski, S. Petersen, C. Beattie, A. Sadik, I. Antonoglou, H. King, D. Kumaran, D. Wierstra, S. Legg, D. Hassabis. Human-level control through deep reinforcement learning. Nature, 2015. 518: 529 533

[78] P. Moritz, R. Nishihara, S. Wang, A. Tumanov, R. Liaw, E. Liang, M. Elibol, Z. Yang, W. Paul, M. I. Jordan, I. Stoica. Ray: a distributed framework for emerging AI applications. USENIX Symposium on Operating Systems Design and Implementation (OSDI). 2018

[79] M. Müller. Computer go. Artificial Intelligence, 2002. 134(1 2): 145 179

[80] A. Nair, P. Srinivasan, S. Blackwell, C. Alcicek, R. Fearon, A. D. Maria, V. Panneershelvam, M. Suleyman, C. Beattie, S. Petersen, S. Legg, V. Mnih, K. Kavukcuoglu, D. Silver. Massively parallel methods for deep reinforcement learning. ArXiv, 2015. abs/1507.04296

[81] A. Y. Ng, S. J. Russell. Algorithms for inverse reinforcement learning. P. Langley, editor, 17th International Conference on Machine Learning, Stanford University, Stanford, CA, USA, 2000. Burlington, Massachusetts: Morgan Kaufmann, 2000

[82] J. Nocedal, S. Wright. Numerical Optimization. Berlin/Heidelberg, Germany: Springer Science \& Business Media, 2006

[83] B. O'Donoghue, R. Munos, K. Kavukcuoglu, V. Mnih. Combining policy gradient and Q-learning. International Conference on Learning Representations (ICLR). 2017

[84] F. A. Oliehoek, M. T. Spaan, N. Vlassis. Optimal and approximate q-value functions for decentralized pomdps. Journal of Artificial Intelligence Research, 2008. 32: 289 353

[85] D. V. Prokhorov, D. C. Wunsch. Adaptive critic designs. IEEE Transactions on Neural Networks, 1997. 8(5): $997 \sim 1007$

[86] T. Rashid, M. Samvelyan, C. Schroeder, G. Farquhar, J. Foerster, S. Whiteson. QMIX: Monotonic value function factorisation for deep multi-agent reinforcement learning. International Conference on Machine Learning (ICML). 2018

[87] S. Ross, D. Bagnell. Efficient reductions for imitation learning. 13th International Conference on Artificial Intelligence and Statistics, Chia Laguna Resort, Sardinia, Italy, 2010. JMLR.org, 2010

[88] G. A. Rummery, M. Niranjan. Online Q-learning Using Connectionist Systems, volume 37. UK: University of Cambridge, 1994

[89] M. Samvelyan, T. Rashid, C. Schroeder de Witt, G. Farquhar, N. Nardelli, T. G. Rudner, C.-M. Hung, P. H. Torr, J. Foerster, S. Whiteson. The StarCraft Multi-Agent Challenge. International Conference on Autonomous Agents and MultiAgent Systems. 2019

[90] S. Schaal. Learning from demonstration. Advances in Neural Information Processing Systems, Denver, CO, USA, 1997. Cambridge, Massachusetts: MIT Press, 1997

[91] J. Schaeffer, N. Burch, Y. Björnsson, A. Kishimoto, M. Müller, R. Lake, P. Lu, S. Sutphen. Checkers is solved. Science, 2007. 317(5844): 1518 1522

[92] J. Schaeffer, J. Culberson, N. Treloar, B. Knight, P. Lu, D. Szafron. A world championship caliber checkers program. Artificial Intelligence, 1992. 53(2 3): 273 289

[93] T. Schaul, J. Quan, I. Antonoglou, D. Silver. Prioritized experience replay. International Conference on Learning Representations. 2015

[94] J. Schulman, S. Levine, P. Abbeel, M. Jordan, P. Moritz. Trust region policy optimization. 32nd International Conference on Machine Learning, Lille, France, 2015. United States: PMLR, 2015

[95] J. Shi, Y. Yu, Q. Da, S. Chen, A. Zeng. Virtual-taobao: Virtualizing real-world online retail environment for reinforcement learning. 33rd AAAI Conference on Artificial Intelligence, Honolulu, Hawaii, USA, 2019. Palo Alto, California: AAAI Press, 2019

[96] W. Shi, S. Song, C. Wu. Soft policy gradient method for maximum entropy deep reinforcement learning. S. Kraus, editor, Proceedings of the Twenty-Eighth International Joint Conference on Artificial Intelligence, Macao, China, 2019. ijcai.org, 2019 3425 3431

[97] Y. Shoham, K. Leyton-Brown. Multiagent Systems: Algorithmic, Game-Theoretic, and Logical Foundations. Cambridge, England: Cambridge University Press, 2008

[98] D. Silver, A. Huang, C. J. Maddison, A. Guez, L. Sifre, G. Van Den Driessche, J. Schrittwieser, I. Antonoglou, V. Panneershelvam, M. Lanctot, et al. Mastering the game of go with deep neural networks and tree search. Nature, 2016. 529(7587): 484 489

[99] D. Silver, G. Lever, N. Heess, T. Degris, D. Wierstra, M. A. Riedmiller. Deterministic policy gradient algorithms. 31th International Conference on Machine Learning, Beijing, China, 2014. JMLR.org, 2014

[100] D. Silver, J. Schrittwieser, K. Simonyan, I. Antonoglou, A. Huang, A. Guez, T. Hubert, L. Baker, M. Lai, A. Bolton, et al. Mastering the game of go without human knowledge. Nature, 2017. 550(7676): 354 359

[101] P. Stone, M. Veloso. Multiagent systems: A survey from a machine learning perspective. Autonomous Robots, 2000. 8(3): 345 383

[102] P. Sunehag, G. Lever, A. Gruslys, W. M. Czarnecki, V. Zambaldi, M. Jaderberg, M. Lanctot, N. Sonnerat, J. Z. Leibo, K. Tuyls, T. Graepel. Value-decomposition networks for cooperative multi-agent learning based on team reward. 17th International Conference on Autonomous Agents and MultiAgent Systems. 2018

[103] R. S. Sutton. Generalization in reinforcement learning: Successful examples using sparse coarse coding. Advances in Neural Information Processing Systems (NIPS). 1996

[104] R. S. Sutton, A. G. Barto. Reinforcement Learning: An Introduction. Cambridge, Massachusetts: MIT press, 2018

[105] R. S. Sutton, D. A. McAllester, S. P. Singh, Y. Mansour. Policy gradient methods for reinforcement learning with function approximation. Advances in Neural Information Processing Systems. 2000 [106] U. Syed, M. H. Bowling, R. E. Schapire. Apprenticeship learning using linear programming. 25th International Conference on Machine Learning, Helsinki, Finland, 2008. New York City, USA: ACM, 2008

[107] A. Tampuu, T. Matiisen, D. Kodelja, I. Kuzovkin, K. Korjus, J. Aru, J. Aru, R. Vicente. Multiagent cooperation and competition with deep reinforcement learning. PloS one, 2017. 12(4): e0172395

[108] M. Tan. Multi-agent reinforcement learning: Independent vs. cooperative agents. International Conference on Machine Learning (ICML). 1993

[109] X. Tang, Z. T. Qin, F. Zhang, Z. Wang, Z. Xu, Y. Ma, H. Zhu, J. Ye. A deep value-network based approach for multi-driver order dispatching. 25th ACM SIGKDD International Conference on Knowledge Discovery \& Data Mining, Anchorage, AK, USA, 2019. New York City: ACM, 2019

[110] G. Tesauro, G. R. Galperin. On-line policy improvement using monte-carlo search. 9th Advances in Neural Information Processing Systems, Denver, CO, USA, 1996. Cambridge, Massachusetts: MIT Press, 1996

[111] C. Tsallis. Possible generalization of boltzmann-gibbs statistics. Journal of Statistical Physics, 1988. 52(1 2): $479 \sim 487$

[112] J. N. Tsitsiklis. Asynchronous stochastic approximation and q-learning. Machine Learning, 1994. 16(3): $185 \sim 202$

[113] J. N. Tsitsiklis, B. Van Roy. An analysis of temporal-difference learning with function approximation. IEEE Transactions on Automatic Control, 1997. 42(5): 674 690

[114] H. J. Van Den Herik, J. W. Uiterwijk, J. Van Rijswijck. Games solved: Now and in the future. Artificial Intelligence, 2002. 134(1 2): 277 311

[115] H. van Hasselt. Double q-learning. Advances in Neural Information Processing Systems (NIPS). 2010

[116] H. van Hasselt, A. Guez, D. Silver. Deep reinforcement learning with double q-learning. Proceedings of the AAAI conference on artificial intelligence, volume 30. 2016

[117] H. van Seijen. Effective multi-step temporal-difference learning for non-linear function approximation. ArXiv, 2016. abs/1608.05151

[118] A. Vaswani, N. Shazeer, N. Parmar, J. Uszkoreit, L. Jones, A. N. Gomez, Ł. Kaiser, I. Polosukhin. Attention is all you need. Advances in Neural Information Processing Systems (NIPS). 2017

[119] N. Vlassis. A concise introduction to multiagent systems and distributed artificial intelligence. Synthesis Lectures on Artificial Intelligence and Machine Learning, 2007. 1(1): 1 71

[120] X. Wang, T. Sandholm. Reinforcement learning to play an optimal nash equilibrium in team markov games. NIPS. 2002

[121] Z. Wang, T. Schaul, M. Hessel, H. Hasselt, M. Lanctot, N. Freitas. Dueling network architectures for deep reinforcement learning. International Conference on Machine Learning (ICML). 2016

[122] C. J. Watkins, P. Dayan. Q-learning. Machine Learning, 1992. 8(3 4): 279 292

[123] C. J. C. H. Watkins. Learning from delayed rewards. 1989

[124] G. Weiss. Multiagent Systems: A Modern Approach to Distributed Artificial Intelligence. Cambridge, Massachusetts: MIT Press, 1999

[125] R. J. Williams. Reinforcement-Learning Connectionist Systems. Boston, Massachusetts: College of Computer Science, Northeastern University, 1987

[126] R. J. Williams. Simple statistical gradient-following algorithms for connectionist reinforcement learning. Machine learning, 1992. 8(3 4): 229 256

[127] R. J. Williams, J. Peng. Function optimization using connectionist reinforcement learning algorithms. Connection Science, 1991. 3(3): 241 268

[128] W. Yang, X. Li, Z. Zhang. A regularized approach to sparse optimal policy in reinforcement learning. Advances in Neural Information Processing Systems. 2019 5940-5950

[129] Z. Yang, Y. Chen, M. Hong, Z. Wang. Provably global convergence of actor-critic: A case for linear quadratic regulator with ergodic cost. Advances in Neural Information Processing Systems (NeurIPS). 2019 8353-8365

[130] T. Yoshikawa. Decomposition of dynamic team decision problems. IEEE Transactions on Automatic Control, 1978. 23(4): 627 632

[131] M. Zaharia, R. S. Xin, P. Wendell, T. Das, M. Armbrust, A. Dave, X. Meng, J. Rosen, S. Venkataraman, M. J. Franklin, A. Ghodsi, J. Gonzalez, S. Shenker, I. Stoica. Apache spark: A unified engine for big data processing. Communications of the ACM, 2016. 59(11): 56 65

[132] K. Zhang, Z. Yang, T. Baar. Multi-agent reinforcement learning: A selective overview of theories and algorithms. ArXiv, 2019. abs/1911.10635

[133] X. Zhao, L. Zhang, L. Xia, Z. Ding, D. Yin, J. Tang. Deep reinforcement learning for list-wise recommendations. ArXiv, 2018. abs/1801.00209

[134] V. Zhong, C. Xiong, R. Socher. Seq2sql: Generating structured queries from natural language using reinforcement learning. ArXiv, 2017. abs/1709.00103

[135] B. D. Ziebart, A. L. Maas, J. A. Bagnell, A. K. Dey. Maximum entropy inverse reinforcement learning. 23rd AAAI Conference on Artificial Intelligence, Chicago, Illinois, USA, 2008. Palo Alto, California, USA: AAAI Press, 2008

[136] B. Zoph, Q. V. Le. Neural architecture search with reinforcement learning. 5th International Conference on Learning Representations, Toulon, France, 2017. OpenReview.net, 2017
