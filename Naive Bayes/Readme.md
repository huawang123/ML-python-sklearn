李航：

朴素贝叶斯（naive bayes）法是基于贝叶斯定理与特征条件独立假设的分类方法。

对于给定的训练数据集，首先基于特征条件独立假设学习学习输入输出的联合概率分
布；然后基于此模型，对给定的输入X,利用贝叶斯定理求出后延概率最大的输入y。

朴素贝叶斯法实现简单，学习效率高。