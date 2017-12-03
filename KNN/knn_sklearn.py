# -*- coding: utf-8 -*-
"""
使用sklearn实现的KNN算法进行分类的一个实例，
使用数据集是Kaggle数字手写体数据库
"""

import pandas as pd
import numpy as np
from sklearn import neighbors
from sklearn.decomposition import PCA
import sklearn

# 加载数据集
def load_data(filename, n, mode):
    data_pd = pd.read_csv(filename)
    data = np.asarray(data_pd)
    pca = PCA(n_components=n)
    if not mode == 'test':
        dateset = pca.fit_transform(data[:, 1:])
        return dateset, data[:, 0]
    else:
        dateset = pca.fit_transform(data)
        return dateset, 1

def main(train_data_path, test_data_path, n_dim):
    train_data, train_label = load_data(train_data_path, n_dim, 'train')
    print("Train set :" + repr(len(train_data)))
    test_data, _ = load_data(test_data_path, n_dim, 'test')
    print("Test set :" + repr(len(test_data)))
    '''
    KNeighborsClassifier(n_neighbors=5, weights='uniform', 
                         algorithm='auto', leaf_size=30, 
                         p=2, metric='minkowski', 
                         metric_params=None, n_jobs=1, **kwargs)
    n_neighbors: 默认值为5，表示查询k个最近邻的数目
    algorithm:   {‘auto’, ‘ball_tree’, ‘kd_tree’, ‘brute’},指定用于计算最近邻的算法，auto表示试图采用最适合的算法计算最近邻
    leaf_size:   传递给‘ball_tree’或‘kd_tree’的叶子大小
    metric:      用于树的距离度量。默认'minkowski与P = 2（即欧氏度量）
    n_jobs:      并行工作的数量，如果设为-1，则作业的数量被设置为CPU内核的数量
    查看官方api：http://scikit-learn.org/dev/modules/generated/sklearn.neighbors.KNeighborsClassifier.html#sklearn.neighbors.KNeighborsClassifier
    '''
    knn = neighbors.KNeighborsClassifier()
    # 训练数据集
    knn.fit(train_data, train_label)
    # 训练准确率
    score = knn.score(train_data, train_label)
    print(">Training accuracy = " + repr(score))
    predictions = []
    for index in range(len(test_data)):
        # 预测
        result = knn.predict([test_data[index]])
        # 预测，返回概率数组
        predict2 = knn.predict_proba([test_data[index]])
        predictions.append([index + 1, result[0]])
        print(">Index : %s, predicted = %s" % (index + 1, result[0]))
    columns = ['ImageId', 'Label']
    save_file = pd.DataFrame(columns=columns, data=predictions)
    save_file.to_csv('m.csv', index=False, encoding="utf-8")

if __name__ == "__main__":
    train_data_path = 'train.csv'
    test_data_path = 'test.csv'
    n_dim = 6
    main(train_data_path, test_data_path, n_dim)
