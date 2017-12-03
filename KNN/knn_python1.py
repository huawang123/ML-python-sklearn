# coding=UTF8
from numpy import *
import operator
from os import listdir

def classify0(inX, dataset, labels, k):
    """
    inX 是输入的测试样本，是一个[x, y]样式的
    dataset 是训练样本集
    labels 是训练样本标签
    k 是top k最相近的
    """
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


def file2matrix(filename):
    """
    从文件中读入训练数据，并存储为矩阵
    """
    fr = open(filename)
    arrayOlines = fr.readlines()
    numberOfLines = len(arrayOlines)  # 获取 n=样本的行数
    returnMat = zeros((numberOfLines, 3))  # 创建一个2维矩阵用于存放训练样本数据，一共有n行，每一行存放3个数据
    classLabelVector = []  # 创建一个1维数组用于存放训练样本标签。
    index = 0
    for line in arrayOlines:
        # 把回车符号给去掉
        line = line.strip()
        # 把每一行数据用\t分割
        listFromLine = line.split('\t')
        # 把分割好的数据放至数据集，其中index是该样本数据的下标，就是放到第几行
        returnMat[index, :] = listFromLine[0:3]
        # 把该样本对应的标签放至标签集，顺序与样本集对应。
        classLabelVector.append(int(listFromLine[-1]))
        index += 1
    return returnMat, classLabelVector


def autoNorm(dataSet):
    """
    训练数据归一化
    """
    # 获取数据集中每一列的最小数值
    # 以createDataSet()中的数据为例，group.min(0)=[0,0]
    minVals = dataSet.min(0)
    # 获取数据集中每一列的最大数值
    # group.max(0)=[1, 1.1]
    maxVals = dataSet.max(0)
    # 最大值与最小的差值
    ranges = maxVals - minVals
    # 创建一个与dataSet同shape的全0矩阵，用于存放归一化后的数据
    normDataSet = zeros(shape(dataSet))
    m = dataSet.shape[0]
    # 把最小值扩充为与dataSet同shape，然后作差，具体tile请翻看 第三节 代码中的tile
    normDataSet = dataSet - tile(minVals, (m, 1))
    # 把最大最小差值扩充为dataSet同shape，然后作商，是指对应元素进行除法运算，而不是矩阵除法。
    # 矩阵除法在numpy中要用linalg.solve(A,B)
    normDataSet = normDataSet / tile(ranges, (m, 1))
    return normDataSet, ranges, minVals


def datingClassTest():
    # 将数据集中10%的数据留作测试用，其余的90%用于训练
    hoRatio = 0.10
    datingDataMat, datingLabels = file2matrix('datingTestSet2.txt')  # load data setfrom file
    normMat, ranges, minVals = autoNorm(datingDataMat)
    m = normMat.shape[0]
    numTestVecs = int(m * hoRatio)
    errorCount = 0.0
    for i in range(numTestVecs):
        classifierResult = classify0(normMat[i, :], normMat[numTestVecs:m, :], datingLabels[numTestVecs:m], 3)
        print("the classifier came back with: %d, the real answer is: %d, result is :%s" % (
        classifierResult, datingLabels[i], classifierResult == datingLabels[i]))
        if (classifierResult != datingLabels[i]): errorCount += 1.0
    print("the total error rate is: %f" % (errorCount / float(numTestVecs)))
    print(errorCount)

if __name__ == "__main__":
    """
    分类约会网站实例
    """
    datingClassTest()



