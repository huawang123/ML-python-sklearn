#coding=UTF8
from numpy import *
import operator

def createDataSet():
    """
    函数作用：构建一组训练数据（训练样本），共4个样本
    同时给出了这4个样本的标签，及labels
    """
    group = array([
        [1.0, 1.1],
        [1.0, 1.0],
        [0. , 0. ],
        [0. , 0.1]
                 ])
    labels = ['A', 'A', 'B', 'B']
    return group, labels

def classify0(inX, dataset, labels, k):
    """
    inX 是输入的测试样本，是一个[x, y]样式的
    dataset 是训练样本集
    labels 是训练样本标签
    k 是top k最相近的
    """
    # shape返回矩阵的[行数，列数]，
    # 那么shape[0]获取数据集的行数，
    # 行数就是样本的数量
    dataSetSize = dataset.shape[0]

    """
    下面的求距离过程就是按照欧氏距离的公式计算的。
    即 根号(x^2+y^2)
    """

    diffMat = tile(inX, (dataSetSize, 1)) - dataset
    sqDiffMat = diffMat ** 2

    sqDistance = sqDiffMat.sum(axis=1)

    # 对平方和进行开根号
    distance = sqDistance ** 0.5

    # 按照升序进行快速排序，返回的是原数组的下标。
    # 比如，x = [30, 10, 20, 40]
    # 升序排序后应该是[10,20,30,40],他们的原下标是[1,2,0,3]
    # 那么，numpy.argsort(x) = [1, 2, 0, 3]
    sortedDistIndicies = distance.argsort()

    # 存放最终的分类结果及相应的结果投票数
    classCount = {}
    # 投票过程，就是统计前k个最近的样本所属类别包含的样本个数
    for i in range(k):

        voteIlabel = labels[sortedDistIndicies[i]]
        # classCount.get(voteIlabel, 0)返回voteIlabel的值，如果不存在，则返回0
        # 然后将票数增1
        classCount[voteIlabel] = classCount.get(voteIlabel, 0) + 1

    # 把分类结果进行排序，然后返回得票数最多的分类结果
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]

if __name__== "__main__":
    # 导入数据
    dataset, labels = createDataSet()
    inX = [0.1, 0.1]
    # 简单分类
    className = classify0(inX, dataset, labels, 3)
    print('the class of test sample is %s' %className)
