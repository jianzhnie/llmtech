## 创新点及贡献

1、研究了最值点附近联合行为值函数的分解性质，然后从理论分析层面进行网络结构涉及从而逼近理论分析结果，其中主要采用了 multiple attention heads 机制来实现对理论分析结果的逼近。

## 研究痛点

1、文章重点研究的还是值分解技术，VDN 的假设过于严格，且忽略了全局信息；QMIX 将 mix 过程视为一个黑盒；QTRAN 的优化问题约束需要通过惩罚项进行解决，从而偏离了真实的结果。

## 算法流程

算法框架如下

![img](https://liushunyu.github.io/img/in-post/2020-07-07-%E5%BC%BA%E5%8C%96%E5%AD%A6%E4%B9%A0%E8%AE%BA%E6%96%87%EF%BC%8816%EF%BC%89Qatten.assets/image-20200707121725186.png)

### 主要思路

1、将联合动作值函数看作是独立动作值函数的函数

![img](https://liushunyu.github.io/img/in-post/2020-07-07-%E5%BC%BA%E5%8C%96%E5%AD%A6%E4%B9%A0%E8%AE%BA%E6%96%87%EF%BC%8816%EF%BC%89Qatten.assets/image-20200707122308260.png)

- 其中每个独立值函数必须与联合动作值函数相关

![img](https://liushunyu.github.io/img/in-post/2020-07-07-%E5%BC%BA%E5%8C%96%E5%AD%A6%E4%B9%A0%E8%AE%BA%E6%96%87%EF%BC%8816%EF%BC%89Qatten.assets/image-20200707122554464.png)

- 在极值点处联合动作值函数对动作的导数为零

![img](https://liushunyu.github.io/img/in-post/2020-07-07-%E5%BC%BA%E5%8C%96%E5%AD%A6%E4%B9%A0%E8%AE%BA%E6%96%87%EF%BC%8816%EF%BC%89Qatten.assets/image-20200707122713202.png)

2、在极值点处独立动作值函数对动作的导数也为零，且可以对其进行泰勒局部展开

![img](https://liushunyu.github.io/img/in-post/2020-07-07-%E5%BC%BA%E5%8C%96%E5%AD%A6%E4%B9%A0%E8%AE%BA%E6%96%87%EF%BC%8816%EF%BC%89Qatten.assets/image-20200707122824468.png)

3、将联合动作值函数在各个独立动作值函数上进行泰勒展开，并将上述独立值函数的泰勒局部展开代入，可以证明得联合动作值函数将变成独立动作值函数的一阶线性组合，具体证明看论文。

- 此处比较巧妙的一个点是当忽略高阶项的时候，我们发现独立动作值函数的二次项部分也被忽略了，只留下了独立动作值函函数的一阶项。

![img](https://liushunyu.github.io/img/in-post/2020-07-07-%E5%BC%BA%E5%8C%96%E5%AD%A6%E4%B9%A0%E8%AE%BA%E6%96%87%EF%BC%8816%EF%BC%89Qatten.assets/image-20200707122910197.png)

4、根据上述定理设计 multiple attention heads 网络架构对理论分析结果进行逼近，直接看结构图还是很好理解的，其中不同的 attention heads 对应不同阶次的近似，而且 attention heads 的数量也决定了逼近的阶次。

![img](https://liushunyu.github.io/img/in-post/2020-07-07-%E5%BC%BA%E5%8C%96%E5%AD%A6%E4%B9%A0%E8%AE%BA%E6%96%87%EF%BC%8816%EF%BC%89Qatten.assets/image-20200707123649378.png)

5、Qatten 算法是符合 IGM 定理的，在 QTRAN 中有提及。

![img](https://liushunyu.github.io/img/in-post/2020-07-07-%E5%BC%BA%E5%8C%96%E5%AD%A6%E4%B9%A0%E8%AE%BA%E6%96%87%EF%BC%8816%EF%BC%89Qatten.assets/image-20200707123839141.png)

6、此外为了放松 self-attention 所施加的权重边界限制，提高 Qatten 的表现能力，论文还提出了 Weighted Head Q-value 的方法

![img](https://liushunyu.github.io/img/in-post/2020-07-07-%E5%BC%BA%E5%8C%96%E5%AD%A6%E4%B9%A0%E8%AE%BA%E6%96%87%EF%BC%8816%EF%BC%89Qatten.assets/image-20200707124341570.png)

- 该方法满足 Theorem 1 的结果

![img](https://liushunyu.github.io/img/in-post/2020-07-07-%E5%BC%BA%E5%8C%96%E5%AD%A6%E4%B9%A0%E8%AE%BA%E6%96%87%EF%BC%8816%EF%BC%89Qatten.assets/image-20200707124448048.png)

## 实验

1、在 StarCraft II decentralized micromanagement tasks 上进行实验

2、从实验效果上看 Qatten 只比 QMIX 好一点，而且值得注意的是 QTRAN 的效果竟然那么差，看来将约束转化为惩罚项从而求解优化问题带来的偏差还是挺大的。

![img](https://liushunyu.github.io/img/in-post/2020-07-07-%E5%BC%BA%E5%8C%96%E5%AD%A6%E4%B9%A0%E8%AE%BA%E6%96%87%EF%BC%8816%EF%BC%89Qatten.assets/image-20200707123857539.png)

## 其他补充

1、该论文从理论分析到网络结构设计的思路还是很不错的，整体很流畅。