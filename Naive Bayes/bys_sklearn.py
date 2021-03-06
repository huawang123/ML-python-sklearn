# -*- coding: utf-8 -*-

"""

使用sklearn实现的贝叶斯算法进行分类的一个实例，

使用数据集是Kaggle数字手写体数据库

"""



import pandas as pd

import numpy as np

from sklearn.naive_bayes import GaussianNB
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

    bys = GaussianNB()

    # 训练数据集

    bys.fit(train_data, train_label)

    # 训练准确率

    score = bys.score(train_data, train_label)

    print(">Training accuracy = " + repr(score))

    predictions = []

    for index in range(len(test_data)):

        # 预测

        result = bys.predict([test_data[index]])

        predict = bys.predict_proba([test_data[index]])

        predictions.append([index + 1, result[0]])

        print(">Index : %s, predicted = %s" % (index + 1, result[0]))

    columns = ['ImageId', 'Label']

    save_file = pd.DataFrame(columns=columns, data=predictions)

    save_file.to_csv('bys.csv', index=False, encoding="utf-8")



if __name__ == "__main__":

    train_data_path = 'train.csv'

    test_data_path = 'test.csv'

    n_dim = 6

    main(train_data_path, test_data_path, n_dim)