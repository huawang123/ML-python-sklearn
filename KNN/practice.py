
import numpy as np

"""
节点
"""
class Node:
    def __init__(self, data, parent, dim):
        self.data = data
        self.parent = parent
        self.lChild = None
        self.rChild = None
        self.dim = dim
    def setLChild(self, lChild):
        self.lChild = lChild
    def setRChild(self, rChild):
        self.rChild = rChild

"""
KdTree
"""
class KdTree:
    def __init__(self, train):
        self.root = self.__build(train, 0, None)

    def __build(self, train, depth, parent): # 递归建树
        (m,k) = train.shape

        if m == 0:
            return None

        train = train[train[:, depth % k].argsort()]

        root = Node(train[int(m/2)], parent, depth % k)
        root.setLChild(self.__build(train[:int(m/2), :], depth+1, root))
        root.setRChild(self.__build(train[int(m/2) + 1:, :], depth+1, root))
        return root

    def findNearestPointAndDistance(self, point): # 查找与point距离最近的点
        point = np.array(point)
        node = self.__findSmallestSubSpace(point, self.root)
        return self.__searchUp(point, node, node, np.linalg.norm(point-node.data))

    def __searchUp(self, point, node, nearestPoint, nearestDistance):
        if node.parent == None:
            return [nearestPoint, nearestDistance]

        distance = np.linalg.norm(node.parent.data - point)
        if distance < nearestDistance:
            nearestDistance = distance
            nearestPoint = node.parent

        distance = np.abs(node.data[node.dim]-point[node.dim])
        if distance < nearestDistance:
            [d, p] = self.__searchDown(point, node)
            if d < nearestDistance:
                nearestDistance = d
                nearestPoint = p

        return self.__searchUp(point, node.parent, nearestPoint, nearestDistance)

    def __searchDown(self, point, node):

        nearestDistance = np.linalg.norm(node.data - point)
        nearestPoint = node

        if node.lChild != None:
            [d, n] = self.__searchDown(point, node.lChild)
            if d < nearestDistance:
                nearestDistance = d
                nearestPoint = node.lChild

        if node.rChild != None:
            [d, n] = self.__searchDown(point, node.rChild)
            if d < nearestDistance:
                nearestDistance = d
                nearestPoint = node.rChild

        return [nearestDistance, nearestPoint]

    def __findSmallestSubSpace(self, point, node): # 找到这个点所在的最小的子空间
        """
        从根节点出发，递归地向下访问kd树。如果point当前维的坐标小于切分点的坐标，则
        移动到左子节点，否则移动到右子节点。直到子节点为叶节点为止。
        """
        if point[node.dim] < node.data[node.dim]:
            if node.lChild == None:
                return node
            else:
                return self.__findSmallestSubSpace(point, node.lChild)
        else:
            if node.rChild == None:
                return node
            else:
                return self.__findSmallestSubSpace(point, node.rChild)

#main

train = np.array([[2,3],[5,4],[9,6],[4,7],[8,1],[7,2]])

target = np.array([3.5, 5])

kdTree = KdTree(train)
[p, d] = kdTree.findNearestPointAndDistance(target)

print (p.data, d)
print ('---------------------')

(m,k) = train.shape
for i in range(m):
    print (train[i], np.linalg.norm(train[i]-target))