import numpy as np

data = np.array([[3, 3], [4, 3], [1, 1]])
label = np.array([1, 1, -1])
class preceptron(object):
    def __init__(self, data,label):
        self.a = np.zeros([len(data), 1])
        self.b = 0
        self.l = 1
        self.data = data
        self.label = label

    def func(self):
        gram_matrix = self.__get_gram_matrix(self.data)
        flag = True
        index = 0
        temp = list(np.tile(self.a, [len(self.data)]))
        while flag:
            index += 1
            i = index % len(self.data)
            self.__updata_wb(gram_matrix, self.label, i)
            temp[i] = np.hstack(self.a)
            count = 0
            for dx, tmp in enumerate(list(temp)[1:]):
                if list(tmp) == list(temp[dx-1]):
                    count = count + 1
            if count == len(temp) - 1:
                flag = False
        return np.sum(self.a * self.data, axis=0), self.b

    def __get_gram_matrix(self, data):
        return np.matmul(data, np.transpose(data))

    def __updata_wb(self, gram_matrix, label, i):
        sum = 0
        for j in range(len(self.a)):
            sum += self.a[j] * label[j] * gram_matrix[j][i]
        if label[i] * (sum  + self.b) <= 0:
            self.a[i] += self.l
            self.b += label[i]
            return self.__updata_wb(gram_matrix, label, i)
        else:
            return


w, b = preceptron(data,label).func()
print('W %s, \nb %s.\n' % (w, b))