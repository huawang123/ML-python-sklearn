import numpy as np

class Tree(object):
    def __init__(self,node_type, Class = None, features = None):
        self.node_type = node_type
        self.dict = {}
        self.Class = Class
        self.feature_index = features

    def add_tree(self,val,tree):
        self.dict[val] = tree

    def predict(self,features):
        if self.node_type == 'leaf':
            return self.Class

        tree = self.dict[features[self.feature_index]]
        return tree.predict(features)

class C45_tree(object):
    def __init__(self, data, label, features, epsilon):
        self.leaf = 'leaf'
        self.internal = 'internal'
        self.epsilon = epsilon
        self.root = self.__build(data, label, features)

    def __build(self, data, labels,features):
        label_kinds = np.unique(labels)
        if len(np.unique(label_kinds)) == 1:
            return Tree(self.leaf, label_kinds[0])
        (max_class, max_len) = max([(i, len(list(filter(lambda x: x == i, labels))))
                                    for i in range(len(label_kinds))],key=lambda x: x[1])
        features_num = len(features)
        if features_num == 0:
            return Tree(self.leaf, label_kinds[0])

        Hd = self.__caclulate_hd(labels)
        Hda, Ha = self.__caclulate_hda_ha(data,labels,features_num)
        Gda = np.tile(Hd, features_num) - Hda
        Grda = Gda / Ha
        max_contribution_feature = list(Grda).index(np.max(Grda))
        if Grda[max_contribution_feature] < self.epsilon:
            return Tree(self.leaf, Class=max_class)
        data_tmp = np.hstack((data[:, :max_contribution_feature], data[:, max_contribution_feature + 1:]))
        sub_features = list(filter(lambda x: x != max_contribution_feature, features))
        tree = Tree(self.internal, features=max_contribution_feature)
        feature_s = np.unique(data[:, max_contribution_feature])
        for feature_index, feature in enumerate(feature_s):
            dx = np.where(data[:, max_contribution_feature] == feature)
            sub_tree = self.__build(data_tmp[dx[0]], labels[dx[0]], sub_features)
            tree.add_tree(feature, sub_tree)
        return tree

    def __caclulate_hd(self, labels):
        label_kinds = np.unique(labels)
        Hd = 0
        for label in label_kinds:
            count = list(labels).count(label)
            p = float(count) / float(len(labels))
            Hd -= p * np.log2(p)
        return Hd

    def __caclulate_hda_ha(self, data, labels, features_num):
        Hda = np.zeros(features_num)
        Ha = np.zeros(features_num)
        for feature_index in range(features_num):
            feature_s = np.unique(data[:, feature_index])
            for feature in feature_s:
                dx = np.where(data[:, feature_index] == feature)
                p = float(len(dx[0])) / float(len(labels))
                h = self.__caclulate_hd(labels[dx])
                Hda[feature_index] += p * h
                Ha[feature_index] -= p * np.log2(p)
        return Hda, Ha

def predict(test_set,tree):
    result = []
    for features in test_set:
        tmp_predict = tree.predict(features)
        result.append(tmp_predict)
    return np.array(result)

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
c45_tree = C45_tree(data, label, [i for i in range(4)], 0.1)
prediction = c45_tree.root.predict([3,1,2,1])
print('Target belong %s' % prediction)
