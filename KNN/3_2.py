import numpy as np

class node(object):
    def __init__(self, data, dim, parent):
        self.data = data
        self.dim = dim
        self.parent = parent
        self.l_child_node = None
        self.r_child_node = None

class setup_tree(object):
    def __init__(self, train):
        self.root = self.__build(data=train, depth=0, parent=None)

    def __build(self, data, parent=None, depth=0):
        n, f = np.shape(data)
        if n == 0:
            return None
        dim_sort = depth % f
        data_sort = data[data[:, dim_sort].argsort()]
        current_node = data_sort[int(n / 2), :]
        root = node(current_node, dim_sort, parent)
        root.l_child_node = self.__build(data_sort[: int(n / 2), :], depth=depth + 1, parent=root)
        root.r_child_node = self.__build(data_sort[(int(n / 2) + 1):, :], depth=depth + 1, parent=root)
        return root

    def find_nearest_point_distance(self, target):
        nearest_region_node = self.__find_nearest_region_node(self.root, target)
        point, nearest_node = self.__serch_up(nearest_region_node, nearest_region_node, target)
        return point, nearest_node

    def __serch_up(self, nearest_region_node_parents, nearest_region_node, target):
        current_distant = np.linalg.norm(nearest_region_node.data - target)
        if nearest_region_node_parents.parent == None:
            return nearest_region_node, current_distant

        if abs((nearest_region_node_parents.parent.data[nearest_region_node_parents.parent.dim] -
                    target[nearest_region_node.parent.dim])) < current_distant:
            if list(nearest_region_node.data) == list(nearest_region_node.parent.l_child_node.data):
                if not nearest_region_node.parent.r_child_node == None:
                    other_node = nearest_region_node.parent.r_child_node
                else:
                    other_node = None
            else:
                if not nearest_region_node.parent.l_child_node == None:
                    other_node = nearest_region_node.parent.l_child_node
                else:
                    other_node = None
            if not other_node == None:
                distant, nearest_node = self.__serch_down(other_node, target)
                if distant < current_distant:
                    nearest_region_node = nearest_node
                    current_distant = distant

        parent_distant = np.linalg.norm(nearest_region_node.parent.data - target)
        if current_distant > parent_distant:
            nearest_region_node = nearest_region_node.parent
            current_distant = parent_distant
        return self.__serch_up(nearest_region_node.parent, nearest_region_node, target)

    def __serch_down(self, node, target):
        nearest_node = node
        current_distant = np.linalg.norm(node.data - target)
        if not node.l_child_node == None:
            distant, nearest_node = self.__serch_down(nearest_node.l_child_node, target)
            if distant < current_distant:
                nearest_node = nearest_node.l_child_node
                current_distant = distant
        if not node.r_child_node == None:
            distant, nearest_node = self.__serch_down(nearest_node.r_child_node, target)
            if distant < current_distant:
                nearest_node = nearest_node.r_child_node
                current_distant = distant
        return current_distant, nearest_node

    def __find_nearest_region_node(self, node, target):
        if target[node.dim] < node.data[node.dim]:
            if node.l_child_node == None:
                return node
            else:
                node = self.__find_nearest_region_node(node.l_child_node, target)
        else:
            if node.r_child_node == None:
                return node
            else:
                node = self.__find_nearest_region_node(node.r_child_node, target)
        return node




# train = np.array([[2,3],[5,4],[9,6],[4,7],[8,1],[7,2]])
#
# target = np.array([3, 4.5])

train = np.array([[2, 3, 2], [5, 4, 4], [9, 6, 3], [4, 7, 6], [8, 1, 3], [7, 2, 6]])

target = np.array([3.5, 5, 4])

kd_tree = setup_tree(train)
[p, d] = kd_tree.find_nearest_point_distance(target)

print (p.data, d)
print ('---------------------')

(m,k) = train.shape
for i in range(m):
    print (train[i], np.linalg.norm(train[i]-target))
