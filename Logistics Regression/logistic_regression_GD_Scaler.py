import math
import numpy as np
from sklearn.preprocessing import StandardScaler

def plot_w(weights, data, label):
    """
    画出数据集和逻辑斯谛最佳回归直线
    :param weights:
    """
    import matplotlib
    # Force matplotlib to not use any Xwindows backend.
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    xcord1 = []; ycord1 = []
    xcord2 = []; ycord2 = []
    for i in range(np.size(data, 0)):
        if int(label[i])== 1:
            xcord1.append(data[i][0]); ycord1.append(data[i][1])
        else:
            xcord2.append(data[i][0]); ycord2.append(data[i][1])
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(xcord1, ycord1, s=30, c='red', marker='s')
    ax.scatter(xcord2, ycord2, s=30, c='green')
    if weights is not None:
        x = range(-3, 3, 1)
        y = (-weights[2]-weights[0]*x)/weights[1]
        ax.plot(x, y)
    plt.xlabel('X1'); plt.ylabel('X2')
    plt.show()
    plt.savefig('asd.png', dpi=300)

class LogisticRegression(object):

    def __init__(self,data, label):
        self.learning_step = 0.00001
        self.max_iteration = 5000
        self.model = self.__gradascent(data, label)

    def __sigmoid(self, wx):
        return [1.0 / (1 + math.exp(-i)) for i in wx]

    def __gradascent(self, data, y):
        """
        逻辑斯谛回归梯度上升优化算法
        :return:权值向量
        """
        x = []
        self.weights = np.zeros((np.size(data, 1) + 1))
        for index in range(len(y)):
            x_ = list(data[index])
            x_.append(1.0)
            x.append(x_)
        x = np.array(x)
        lr = 0.001  # 学习率
        maxcycles = 10000
        for k in range(maxcycles):  # 最大迭代次数
            h = self.__sigmoid(np.matmul(x, self.weights.transpose()))  # 矩阵内积
            error = (y - h)  # 向量减法
            self.weights += lr * np.matmul(x.transpose(), np.array(error))  # 矩阵内积
        plot_w(self.weights, x, y)

    def predict(self, x):
        x = list(x)
        x.append(1.0)
        wx = sum([self.weights[i] * x[i] for i in range(len(x))])
        Ewx = math.exp(wx)
        p1 =  Ewx / (1 + Ewx)
        p0 = 1 / (1 + Ewx)
        if p1 > p0:
            return 1
        else:
            return 0

data = np.array(
            [[-0.017612, 14.053064],
             [-1.395634, 1.662541],
             [-0.752157, 6.538620],
             [-1.322371, 7.152853],
             [0.423363,	11.054677],
             [0.406704,	7.067335],
             [0.667394,	12.741452],
             [-2.460150, -0.866805],
             [0.569411,	9.548755],
             [-0.026632, 10.427743]], dtype=float)
label = np.array([0, 1, 0, 0, 0, 1, 0, 1, 0, 0])
target = [2.0, 1.0]
scaler = StandardScaler().fit(data)
data = scaler.transform(data)
target = scaler.transform([target])[0]
model = LogisticRegression(data, label)
prediction = model.predict(target)
print('Target belong %s' % prediction)

