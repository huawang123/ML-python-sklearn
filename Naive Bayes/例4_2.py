
import numpy as np

class bayes(object):
    def __init__(self, data, label, num_class, L):
        # data : (list) samples_nums * [features_nums]
        self.data = data
        self.label = label
        self.num_class = num_class
        self.L = L
        self.p_prams = []
        self.p_label = np.zeros(self.num_class)
        self.fea_condition = []
        self.model = self.__model()
    def __model(self):
        self.__get_p_gram()

    def __get_p_gram(self):
        self.__generation_features_conditional()
        start = 0
        for i in range(self.num_class):
            # i 代表第i类
            Ik = list(self.label).count(i)
            self.p_label[i] = (Ik + self.L)/ (len(self.label) + self.num_class * self.L)
            end = Ik + start
            index_sort = np.argsort(self.label)
            condition_k = []#condition_k   是一个列表，保存第I类每个特征不同取值的概率
            for index, condition in enumerate(self.fea_condition):
                temp = self.data[index_sort[start:end], index].reshape(1, -1)[0]
                condition_kj = []  # 保存第index个特征不同取值的概率
                for c in condition:
                    condition_kj.append((list(temp).count(c) + self.L)/(end - start + len(condition) * self.L))
                condition_k.append(condition_kj)
            start = end
            self.p_prams.append(condition_k)

    def  __generation_features_conditional(self):
        #找出每种特征出现的所有情况
        features_nums = self.data.shape[1]
        for j in range(features_nums):
            # j代表第j个特征
            self.fea_condition.append(np.unique(self.data[:, j]))

    def classify(self, target):
        p = list(self.p_label)
        for index, _ in enumerate(p):
            for fea_index,fea in enumerate(list(target)):
                fea_local = list(self.fea_condition[index]).index(fea)#每个特征值在所属的S集合中的位置
                p[index] *= self.p_prams[index][fea_index][fea_local]
        c = np.asarray(p).argsort()[-1]
        return p, c


data = np.array([[1, 1], [1, 2], [1, 2], [1, 1], [1, 1], [2, 1], [2, 2], [2, 2],[2, 3], [2, 3], [3, 3], [3, 2], [3, 2], [3, 3], [3, 3]])
label = np.array([0, 0, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0])
target = [2, 1]
num_class = 2
L = 1 #拉普拉斯平滑参数
model = bayes(data, label, num_class, L)
p, c = model.classify(target)
print('Target belong %s, \nP is %s.\n' % (c, p[c]))