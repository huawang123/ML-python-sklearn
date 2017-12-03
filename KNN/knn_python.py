# -*- coding: utf-8 -*-
"""
使用python实现的KNN算法进行分类的一个实例，
使用数据集是Kaggle数字手写体数据库
"""
import pandas as pd
import numpy as np
import math
import operator

# 加载数据集
def load_data(filename, mode):
    data_pd = pd.read_csv(filename)
    data = np.asarray(data_pd)
    if not mode == 'test':
        return data[:, 1:], data[:, 1]
    else:
        return data, 1

# 计算距离
def euclideanDistance(instance1, instance2, length):
    distance = 0
    for index in range(length):
        distance = pow((instance1[index] - instance2[index]), 2)
    return math.sqrt(distance)


# 返回K个最近邻
def getNeighbors(trainingSet, testInstance, k):
    distances = []
    length = len(testInstance) - 1
    # 计算每一个测试实例到训练集实例的距离
    for index in range(len(trainingSet)):
        dist = euclideanDistance(testInstance, trainingSet[index], length)
        distances.append((trainingSet[index], dist))
    # 对所有的距离进行排序
    distances.sort(key=operator.itemgetter(1))
    neighbors = []
    # 返回k个最近邻
    for index in range(k):
        neighbors.append(distances[index][0])
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

def main(train_data_path, test_data_path, top_k):
    train_data, train_label = load_data(train_data_path, 'train')
    print("Train set :" + repr(len(train_data)))
    test_data, _ = load_data(test_data_path, 'test')
    print("Test set :" + repr(len(test_data)))
    predictions = []
    for index in range(len(test_data)):
        neighbors = getNeighbors(train_data, test_data[index], top_k)
        result = getResponse(neighbors)
        predictions.append([index + 1, result])
        print(">predicted = " + repr(result))
    columns = ['ImageId', 'Label']
    save_file = pd.DataFrame(columns=columns, data=predictions)
    save_file.to_csv('prediction.csv', index=False, encoding="utf-8")

if __name__ == "__main__":
    train_data_path = 'train.csv'
    test_data_path = 'test.csv'
    top_k = 5
    main(train_data_path, test_data_path, top_k)
