# -*- coding: utf-8 -*-
"""
使用python实现的KNN算法进行分类的一个实例，
使用数据集是Kaggle数字手写体数据库
"""
import pandas as pd
import numpy as np
import math
import operator
from sklearn.decomposition import PCA

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

# 计算距离
def euclideanDistance(instance1, instance2, length):
    distance = 0
    for index in range(length):
        distance = pow((instance1[index] - instance2[index]), 2)
    return math.sqrt(distance)


# 返回K个最近邻
def getNeighbors(trainingSet, train_label, testInstance, k):
    distances = []
    length = len(testInstance) - 1
    # 计算每一个测试实例到训练集实例的距离
    for index in range(len(trainingSet)):
        dist = euclideanDistance(testInstance, trainingSet[index], length)
        distances.append(dist)
    # 对所有的距离进行排序
    sortedDistIndicies = np.asarray(distances).argsort()
    neighbors = []
    # 返回k个最近邻
    for index in range(k):
        dex = sortedDistIndicies[index]
        neighbors.append((dex, train_label[dex]))
    return neighbors


# 对k个近邻进行合并，返回value最大的key
def getResponse(neighbors):
    classVotes = {}
    for index in range(len(neighbors)):
        response = neighbors[index][-1]
        if response in classVotes:
            classVotes[response] += 1
        else:
            classVotes[response] = 1
    # 排序
    sortedVotes = sorted(classVotes.items(), key=operator.itemgetter(1), reverse=True)
    return sortedVotes[0][0]

def main(train_data_path, test_data_path, top_k, n_dim):
    train_data, train_label = load_data(train_data_path, n_dim, 'train')
    print("Train set :" + repr(len(train_data)))
    test_data, _ = load_data(test_data_path, n_dim, 'test')
    print("Test set :" + repr(len(test_data)))
    predictions = []
    for index in range(len(test_data)):
        neighbors = getNeighbors(train_data, train_label, test_data[index], top_k)
        result = getResponse(neighbors)
        predictions.append([index + 1, result])
        print(">Index : %s, predicted = %s" % (index + 1, result))
    columns = ['ImageId', 'Label']
    save_file = pd.DataFrame(columns=columns, data=predictions)
    save_file.to_csv('mm.csv', index=False, encoding="utf-8")

if __name__ == "__main__":
    train_data_path = 'train.csv'
    test_data_path = 'test.csv'
    top_k = 5
    n_dim = 6
    main(train_data_path, test_data_path, top_k, n_dim)
