李航

决策树（decision）是一种基本的分类与回归算法。

决策树呈树形结构，在分类问题中，表示基于特征对实例进行分类的过程。

它可以认为是if-then规则的集合，也可以认为定义在特征空间与类空间上
的条件概率分布。

其主要优点在于模型具有可读性，分类速度快。

学习时，利用训练数据，根据损失函数最小化的原则建立决策树模型。
预测时，对新的数据，利用决策树模型进行分类。

决策树的学习通常包括三个部分：特征选择、决策树生成和决策树的修剪

决策树的思想主要来源于Quinlan在1986年提出的ID3算法和1993年的C4.5算
法，以及Breiman等人在1984年提出的CART算法

决策树的一些优点：

易于理解和解释。数可以可视化。
几乎不需要数据预处理。其他方法经常需要数据标准化，创建虚拟变量和删除缺失值。决策树还不支持缺失值。
使用树的花费（例如预测数据）是训练数据点(data points)数量的对数。
可以同时处理数值变量和分类变量。其他方法大都适用于分析一种变量的集合。
可以处理多值输出变量问题。
使用白盒模型。如果一个情况被观察到，使用逻辑判断容易表示这种规则。相反，如果是黑盒模型（例如人工神经网络），结果会非常难解释。
可以使用统计检验检验模型。这样做被认为是提高模型的可行度。
即使对真实模型来说，假设无效的情况下，也可以较好的适用。



决策树的一些缺点：

决策树学习可能创建一个过于复杂的树，并不能很好的预测数据。也就是过拟合。修剪机制（现在不支持），设置一个叶子节点需要的最小样本数量，或者数的最大深度，可以避免过拟合。
决策树可能是不稳定的，因为即使非常小的变异，可能会产生一颗完全不同的树。这个问题通过decision trees with an ensemble来缓解。
学习一颗最优的决策树是一个NP-完全问题under several aspects of optimality and even for simple concepts。因此，传统决策树算法基于启发式算法，例如贪婪算法，即每个节点创建最优决策。这些算法不能产生一个全家最优的决策树。对样本和特征随机抽样可以降低整体效果偏差。
概念难以学习，因为决策树没有很好的解释他们，例如，XOR, parity or multiplexer problems.
如果某些分类占优势，决策树将会创建一棵有偏差的树。因此，建议在训练之前，先抽样使样本均衡

