# 878 · Boundary of Binary Tree
"""
Description
Given a binary tree, return the values of its boundary in anti-clockwise direction starting from root. 
Boundary includes left boundary, leaves, and right boundary in order without duplicate nodes.

Left boundary is defined as the path from root to the left-most node. 
Right boundary is defined as the path from root to the right-most node. 
If the root doesn't have left subtree or right subtree, then the root itself is left boundary or right boundary. 
Note this definition only applies to the input binary tree, and not applies to any subtrees.

The left-most node is defined as a leaf node you could reach when you always firstly travel to the 
left subtree if exists. If not, travel to the right subtree. Repeat until you reach a leaf node.
The right-most node is also defined by the same way with left and right exchanged.
"""
# Input: {1,2,3,4,5,6,#,#,#,7,8,9,10}
# Output: [1,2,4,7,8,9,10,6,3]
# Explanation: 
#           1
#      /          \
#     2            3
#    / \          / 
#   4   5        6   
#      / \      / \
#     7   8    9  10  
#   The left boundary are node 1,2,4. (4 is the left-most node according to definition)
#   The leaves are node 4,7,8,9,10.
#   The right boundary are node 1,3,6,10. (10 is the right-most node).
#   So order them in anti-clockwise without duplicate nodes we have [1,2,4,7,8,9,10,6,3].


# Definition of TreeNode:
class TreeNode:
    def __init__(self, val):
        self.val = val
        self.left, self.right = None, None

class Solution:
    """
    @param root: a TreeNode
    @return: a list of integer
    """
    def boundaryOfBinaryTree(self, root):
        if root is None:
            return []
            
        left_boundary = self.find_left_boundary(root.left)
        leaves = self.find_leaves(root)
        right_boundary = self.find_right_boundary(root.right)
        
        if left_boundary and leaves and left_boundary[-1] == leaves[0]:
            leaves = leaves[1:]
        if leaves and right_boundary and leaves[-1] == right_boundary[-1]:
            leaves = leaves[:-1]
        return [root.val] + left_boundary + leaves + list(reversed(right_boundary))

    def find_leaves(self, root):
        stack = [root]
        leaves = []
        while stack:
            node = stack.pop()
            if not node.left and not node.right:
                leaves.append(node.val)
            if node.right:      # 这里一定注意！！！先右节点入栈！！！
                stack.append(node.right)
            if node.left:
                stack.append(node.left)
        return leaves
        
    def find_left_boundary(self, root):
        left_boundary = []
        while root is not None:
            left_boundary.append(root.val)
            if root.left:
                root = root.left
            elif root.right:        # 这里一定要elif，没发往左走再往右找
                root = root.right
            else:
                break       # 到叶子就break
        return left_boundary
        
    def find_right_boundary(self, root):
        right_boundary = []
        while root is not None:
            right_boundary.append(root.val)
            if root.right:
                root = root.right
            elif root.left:     # 这里一定要elif，没发往右走再往左找
                root = root.left
            else:
                break       # 到叶子就break
        return right_boundary