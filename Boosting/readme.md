1.正则化
   XGBoost在代价函数里加入了正则项，用于控制模型的复杂度。正则项里包含了树的叶子节点个数、每个叶子节点上输出的score的L2模的平方和。从Bias-variancetradeoff角度来讲，正则项降低了模型的variance，使学习出来的模型更加简单，防止过拟合，这也是xgboost优于传统GBDT的一个特性。
2.并行处理
    XGBoost工具支持并行。Boosting不是一种串行的结构吗?怎么并行的？注意XGBoost的并行不是tree粒度的并行，XGBoost也是一次迭代完才能进行下一次迭代的（第t次迭代的代价函数里包含了前面t-1次迭代的预测值）。XGBoost的并行是在特征粒度上的。
    我们知道，决策树的学习最耗时的一个步骤就是对特征的值进行排序（因为要确定最佳分割点），XGBoost在训练之前，预先对数据进行了排序，然后保存为block结构，后面的迭代中重复地使用这个结构，大大减小计算量。这个block结构也使得并行成为了可能，在进行节点的分裂时，需要计算每个特征的增益，最终选增益最大的那个特征去做分裂，那么各个特征的增益计算就可以开多线程进行。
  3.灵活性

    XGBoost支持用户自定义目标函数和评估函数，只要目标函数二阶可导就行。
  4.缺失值处理
     对于特征的值有缺失的样本，xgboost可以自动学习出它的分裂方向。

  5.剪枝

     XGBoost 先从顶到底建立所有可以建立的子树，再从底到顶反向进行剪枝。比起GBM，这样不容易陷入局部最优解。

  6.内置交叉验证
    XGBoost允许在每一轮boosting迭代中使用交叉验证。因此，可以方便地获得最优boosting迭代次数。而GBM使用网格搜索，只能检测有限个值。


三
XGBooST详解： 
1.数据格式

XGBoost可以加载多种数据格式的训练数据：　　
libsvm 格式的文本数据；
Numpy 的二维数组；
XGBoost 的二进制的缓存文件。加载的数据存储在对象 DMatrix 中。
下面一一列举：
加载libsvm格式的数据
>>> dtrain1 = xgb.DMatrix('train.svm.txt')

加载二进制的缓存文件
>>> dtrain2 = xgb.DMatrix('train.svm.buffer')

加载numpy的数组
>>> data = np.random.rand(5,10) # 5 entities, each contains 10 features
>>> label = np.random.randint(2, size=5) # binary target
>>> dtrain = xgb.DMatrix( data, label=label)


将scipy.sparse格式的数据转化为 DMatrix 格式
>>> csr = scipy.sparse.csr_matrix( (dat, (row,col)) ) >>> dtrain = xgb.DMatrix( csr )

将 DMatrix 格式的数据保存成XGBoost的二进制格式，在下次加载时可以提高加载速度，使用方式如下
>>> dtrain = xgb.DMatrix('train.svm.txt')
>>> dtrain.save_binary("train.buffer")


可以用如下方式处理 DMatrix中的缺失值：
>>> dtrain = xgb.DMatrix( data, label=label, missing = -999.0)



当需要给样本设置权重时，可以用如下方式
>>> w = np.random.rand(5,1)
>>> dtrain = xgb.DMatrix( data, label=label, missing = -999.0, weight=w)
2.参数设置

XGBoost使用key-value字典的方式存储参数：

    params = {    
'booster': 'gbtree',    'objective': 'multi:softmax',  # 多分类的问题    'num_class': 10,            # 类别数，与 multisoftmax 并用   
 'gamma': 0.1,       # 用于控制是否后剪枝的参数,越大越保守，一般0.1、0.2这样子。
 'max_depth': 12,               # 构建树的深度，越大越容易过拟合    'lambda': 2,         # 控制模型复杂度的权重值的L2正则化项参数，参数越大，模型越不容易过拟合。
 'subsample': 0.7,              # 随机采样训练样本
 'colsample_bytree': 0.7,       # 生成树时进行的列采样    'min_child_weight': 3,    'silent': 1,                   # 设置成1则没有运行信息输出，最好是设置为0.
 'eta': 0.007,                  # 如同学习率 
 'seed': 1000,    'nthread': 4,                  # cpu 线程数}

3.训练模型

有了参数列表和数据就可以训练模型了 

num_round = 10
bst = xgb.train( plst, dtrain, num_round, evallist )
4.模型预测
# X_test类型可以是二维List，也可以是numpy的数组
dtest = DMatrix(X_test) ans = model.predict(dtest)
5.保存模型
在训练完成之后可以将模型保存下来，也可以查看模型内部的结构
 bst.save_model('test.model')
导出模型和特征映射（Map）
       你可以导出模型到txt文件并浏览模型的含义：
# dump model
bst.dump_model('dump.raw.txt')
# dump model with feature map
bst.dump_model('dump.raw.txt','featmap.txt')
6.加载模型

通过如下方式可以加载模型：
bst = xgb.Booster({'nthread':4}) # init model
bst.load_model("model.bin")      # load data
