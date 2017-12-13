import numpy as np

class Tree(object):
    def __init__(self,node_type, Class = None, feature_index = None, feature = None):
        self.node_type = node_type
        self.dict = {}
        self.Class = Class
        self.feature_index = feature_index
        self.feature = feature

    def add_tree(self,val,tree):
        self.dict[val] = tree

    def predict(self,features):
        if self.node_type == 'leaf':
            return self.Class
        if features[self.feature_index] == self.feature:
            tree = self.dict[self.feature]
        else:
            tree = self.dict[-1]
        return tree.predict(features)

class Id3_tree(object):
    def __init__(self, data, label, features):
        self.leaf = 'leaf'
        self.internal = 'internal'
        self.root = self.__build(data, label, features)

    def __build(self, data, labels,features):
        label_kinds = np.unique(labels)
        if len(np.unique(label_kinds)) == 1:
            return Tree(self.leaf, label_kinds[0])
        features_num = len(features)
        if features_num == 0:
            return Tree(self.leaf, label_kinds[0])

        Ga = self.__caclulate_ga(data,labels,features_num)
        ga_fea_min = min(Ga[0])
        fea_local = list(Ga[0]).index(ga_fea_min)
        ga_fea_index = 0
        for dx, i in enumerate(Ga[1:]):
            ga_fea_min_tmp = min(i)
            if ga_fea_min_tmp < ga_fea_min:
                fea_local = list(i).index(ga_fea_min_tmp)
                ga_fea_min = ga_fea_min_tmp
                ga_fea_index = dx + 1
        data_tmp = np.hstack((data[:, :ga_fea_index], data[:, ga_fea_index + 1:]))
        sub_features = list(filter(lambda x: x != ga_fea_index, features))
        feature_s = np.unique(data[:, ga_fea_index])
        tree = Tree(self.internal, feature_index=features[ga_fea_index], feature=feature_s[fea_local])
        dx_y = np.where(data[:, ga_fea_index] == feature_s[fea_local])
        sub_tree = self.__build(data_tmp[dx_y], labels[dx_y], sub_features)
        tree.add_tree(feature_s[fea_local], sub_tree)
        dx_n = np.where(data[:, ga_fea_index] != feature_s[fea_local])
        sub_tree = self.__build(data_tmp[dx_n], labels[dx_n], sub_features)
        tree.add_tree(-1, sub_tree)

        return tree

    def __caclulate_q(self, labels):
        label_kinds = np.unique(labels)
        q = 0
        for label in label_kinds:
            count = list(labels).count(label)
            p = float(count) / float(len(labels))
            q += p * (1 - p)
        return q

    def __caclulate_ga(self, data, labels, features_num):
        Ga = []
        for feature_index in range(features_num):
            feature_s = np.unique(data[:, feature_index])
            Gai = np.zeros(len(feature_s))
            for index, feature in enumerate(feature_s):
                dx_y = np.where(data[:, feature_index] == feature)
                p = float(len(dx_y[0])) / float(len(labels))
                q_y = self.__caclulate_q(labels[dx_y])
                dx_n = np.where(data[:, feature_index] != feature)
                q_n = self.__caclulate_q(labels[dx_n])
                dx = np.where(data[:, feature_index] == feature)
                Gai[index] += (p * q_y + (1 - p) * q_n)
            Ga.append(Gai)
        return Ga

data = np.array([[1,2,2,3],
                 [1,2,2,2],
                 [1,1,2,2],
                 [1,1,1,3],
                 [1,2,2,3],
                 [2,2,2,3],
                 [2,2,2,2],
                 [2,1,1,2],
                 [2,2,1,1],
                 [2,2,1,1],
                 [3,2,1,1],
                 [3,2,1,2],
                 [3,1,2,2],
                 [3,1,2,1],
                 [3,2,2,3]])
label = np.array([0,0,1,1,0,0,0,1,1,1,1,1,1,1,0])
target = [3,1,2,1]
id3_tree = Id3_tree(data, label, [i for i in range(4)])
prediction = id3_tree.root.predict(target)
print('Target belong %s' % prediction)