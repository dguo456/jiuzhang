# 1506 · All Nodes Distance K in Binary Tree
"""
We are given a binary tree (with root node root), a target node, and an integer value K.
Return a list of the values of all nodes that have a distance K from the target node. 
The answer can be returned in any order.
"""

# Definition of TreeNode:
class TreeNode:
    def __init__(self, val):
        self.val = val
        self.left, self.right = None, None

"""
一个宽度优先搜索（BFS）的解法, 分成两个步骤:
1.  把二叉树转换为普通的图，存储在邻接表里
2.  用宽度优先搜索算法，搜索 K 层就找到了要找的点
"""
from collections import deque

class Solution:
    """
    @param root: the root of the tree
    @param target: the target
    @param K: the given K
    @return: All Nodes Distance K in Binary Tree
    """
    def distanceK(self, root, target, K):
        # 邻接表
        graph = {}
        self.build_tree(graph, None, root)

        # BFS
        queue = deque([target])
        visited = set([target])
        level = 0

        while queue:
            level += 1
            if level == K + 1:
                return [x.val for x in queue]
            len_queue = len(queue)
            for _ in range(len_queue):
                node = queue.popleft()
                for neighbor in graph[node]:
                    if neighbor in visited:
                        continue
                    queue.append(neighbor)
                    visited.add(neighbor)

        return []

    def build_tree(self, graph, father, child):
        if child is None:
            return
        self.add_edge(graph, child, child.left)
        self.add_edge(graph, child, child.right)
        self.add_edge(graph, child, father)

        self.build_tree(graph, child, child.left)
        self.build_tree(graph, child, child.right)

    def add_edge(self, graph, nodeA, nodeB):
        if nodeB is None:
            return
        if nodeA not in graph:
            graph[nodeA] = set()
        # 这里注意节点可能为None，所以不能用else
        graph[nodeA].add(nodeB)



# Method.2      DFS
"""
分成两个部分考虑，一个是 target 的儿子节点们，一个是 target 的父亲节点的另外一边的节点们。
1.  找到 root 到 target 的这条路径上的所有点
2.  循环这个路径上的点，处理可能的离 target 具体为 K 的点
"""
class Solution:
    """
    @param root: the root of the tree
    @param target: the target
    @param K: the given K
    @return: All Nodes Distance K in Binary Tree
    """
    def distanceK(self, root, target, K):
        self.result = set()
        self.path = []
        self.find_path([], root, target)
        self.dfs_subtree(self.path[0], K)
        for i in range(1, len(self.path)):
            child = self.path[i - 1]
            parent = self.path[i]
            if i == K:
                self.result.add(parent.val)
                break
            if parent.left == child:
                self.dfs_subtree(parent.right, K - i - 1)
            else:
                self.dfs_subtree(parent.left, K - i - 1)

        return sorted(self.result)

    def find_path(self, path, root, target):
        if root is None:
            return

        path.append(root)
        if root == target:
            self.path = path[::-1]
            return
        self.find_path(path, root.left, target)
        self.find_path(path, root.right, target)
        path.pop()

    def dfs_subtree(self, root, K):
        if K < 0:
            return
        if root is None:
            return
        if K == 0:
            self.result.add(root.val)
            return
        self.dfs_subtree(root.left, K - 1)
        self.dfs_subtree(root.right, K - 1)