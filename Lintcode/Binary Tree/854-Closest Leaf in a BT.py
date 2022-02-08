# 854 · Closest Leaf in a Binary Tree
"""
Description
Given a binary tree where every node has a unique value, and a target key k.
Find the value of the nearest leaf node to target k in the tree. If there are multiple cases, 
you should follow these priorities:

1.  The leaf node is in the left subtree of the node with k;
2.  The leaf node is in the right subtree of the node with k;
3.  The leaf node is not in the subtree of the node with k.
root represents a binary tree with at least 1 node and at most 1000 nodes.
Every node has a unique node.val in range [1, 1000][1,1000].
There exists a node in the given binary tree for which node.val == k.
"""

#         -4
#        /  \
#      (2)    6
#      / \  
#     3   1
#    /   / \
#   5   8   9
#  / \   \   \
# 4  7    10  11

# Method.1      双BFS
from collections import deque

class Solution:
    """
    @param root: the root
    @param k: an integer
    @return: the value of the nearest leaf node to target k in the tree
    """
    def findClosestLeaf(self, root, k):
        if not root:
            return None

        node_to_par = {}
        visited = set()
        target_node, node_to_par = self.find_target_node(root, k, node_to_par)
        
        queue = deque([target_node])
        while queue:
            curr = queue.popleft()
            visited.add(curr)
            
            if not curr.left and not curr.right:
                return curr.val    # guaranteed find
            
            # 根据 priority，左 -> 右 -> 父
            if curr.left and curr.left not in visited:
                queue.append(curr.left)
                visited.add(curr.left)
            
            if curr.right and curr.right not in visited:
                queue.append(curr.right)
                visited.add(curr.right)
            
            # 见上图实例, 节点2的最近叶子结点是6，需要记录其父节点
            parent = node_to_par.get(curr)
            if parent and parent not in visited:
                queue.append(parent)
                visited.add(parent)
                
    def find_target_node(self, root, k, node_to_par):
        # 实际上只关心 target_node 一条路的父节点，但搜索是从上到下，无法反推，所以只能记录
        # 与 target_node 平层及以上的父节点
        queue = deque([root])
        while queue:
            curr = queue.popleft()
            if curr.val == k:
                return curr, node_to_par    # guaranteed find
            
            if curr.left:
                queue.append(curr.left)
                node_to_par[curr.left] = curr
            if curr.right:
                queue.append(curr.right)
                node_to_par[curr.right] = curr