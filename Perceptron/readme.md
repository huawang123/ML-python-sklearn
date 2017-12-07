

感知机（perception）是二分类的线性分类模型，其输入为实例的特征向量，输出为实例的类别，取+1和-1.

感知机对应于输入空间（特征空间）中将实例化分为正负两类的分离超平面，为此，导入基于误分类的损失函数，利用梯度下降法对损失函数进行极小化，求得感知机模型。

感知及算法具有简单而易于实现的优点，分为原始形式和对偶形式。

感知机预测是利用学习到的感知机模型对新的输入实例进行分类。

感知机1957年由Rosenblatt提出，是神经网络和支持向量机的基础。

收敛性证明结果：

k < (R/Y)^2 ,其中，k为误分类次数，R为||xi||的最大值i=1,2,3...n, Y

为yi(Wopt*xi + Bopt)的最小值。

这个定理表明，误分类次数是有上界的，经过有限次搜索可以找到将训练数据完全正确分开的分离超平面，也就是说，当训练数据集线性可分时，感知基学习算法原始形式迭代是收敛的。但是当初始值及更新样本不定时学出来的超平面是变化的，故需要加约束条件，SVM就是加了约束条件。当训练数据集线性不可分时，感知机学习算法不收敛，迭代结果将会发生震荡。

对偶形式的基本思想，将权重w和偏置b表示为实例xi和yi的线性组合形式，通过求解其系数，从而

求解最终的的w和b