import xgboost as xgb
import numpy as np
from sklearn.tree import DecisionTreeClassifier

# 1、xgBoost的基本使用
# 2、自定义损失函数的梯度和二阶导
# 3、binary:logistic/logitraw

# 定义f: theta * x
def g_h(y_hat, y):
    p = 1.0 / (1.0 + np.exp(-y_hat))
    g = p - y.get_label()
    h = p * (1.0-p)
    return g, h


def error_rate(y_hat, y):
    return 'error', float(sum(y.get_label() != (y_hat > 0.5))) / len(y_hat)


if __name__ == "__main__":
    # 读取数据
    data_train = xgb.DMatrix('agaricus_train.txt')
    data_test = xgb.DMatrix('agaricus_test.txt')
    print (data_train)
    print (type(data_train))

    # 设置参数
    param = {'max_depth': 3, 'eta': 0.3, 'silent': 1, 'objective': 'binary:logitraw'}
    # max_depth : 树的最大深度  一般为3-10
    # eta ： 阻尼值即学习率，为1时是原始模型，一般小于0.1，如果太小学习次数增加
    # silent ： 为0时输出树的信息和剪枝信息，为1时不输出
    # objective : 目标函数（损失函数）
    watchlist = [(data_test, 'eval'), (data_train, 'train')]
    n_round = 7
    # bst = xgb.train(param, data_train, num_boost_round=n_round, evals=watchlist)
    bst = xgb.train(param, data_train, num_boost_round=n_round, evals=watchlist, obj=g_h, feval=error_rate)

    # 计算错误率
    y_hat = bst.predict(data_test)
    y = data_test.get_label()
    print (y_hat)
    print (y)
    error = sum(y != (y_hat > 0.5))
    error_rate = float(error) / len(y_hat)
    print ('样本总数：\t', len(y_hat))
    print ('错误数目：\t%4d' % error)
    print ('错误率：\t%.5f%%' % (100*error_rate))
