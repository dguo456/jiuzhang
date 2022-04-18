# 910 · Largest BST Subtree
"""
Given a binary tree, find the largest subtree which is a Binary Search Tree (BST), 
where largest means subtree with largest number of nodes in it.
"""

# Definition of TreeNode:
class TreeNode:
    def __init__(self, val):
        self.val = val
        self.left, self.right = None, None

import sys

class Solution:
    """
    @param root: the root
    @return: the largest subtree's size which is a Binary Search Tree
    """
    def largestBSTSubtree(self, root):
        if root is None:
            return 0

        _, size, _, _, = self.dfs(root)
        return size

    def dfs(self, root):
        if root is None:
            return True, 0, sys.maxsize, -sys.maxsize

        # 标准分治,返回值用result type，需要四个返回值
        l_bst, l_size, l_min, l_max = self.dfs(root.left)
        r_bst, r_size, r_min, r_max = self.dfs(root.right)

        # 如果左子树，右子树均为BST并且: 左子树里的最大值 < 当前root < 右子树里的最小值，则为BST
        is_BST = l_bst and r_bst and root.val > l_max and root.val < r_min

        if is_BST:
            size = l_size + r_size + 1
        else:
            size = max(l_size, r_size)

        return is_BST, size, min(l_min, r_min, root.val), max(l_max, r_max, root.val)



"""Another Method"""
# A class to store a BST node
class Node:
    # constructor
    def __init__(self, val, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right
 
 
# Recursive function to calculate the size of a given binary tree
def size(root):
 
    # base case: empty tree has size 0
    if root is None:
        return 0
 
    # recursively calculate the size of the left and right subtrees and
    # return the sum of their sizes + 1 (for root node)
    return size(root.left) + 1 + size(root.right)
 
 
# Recursive function to determine if a given binary tree is a BST or not
# by keeping a valid range (starting from [-INFINITY, INFINITY]) and
# keep shrinking it down for each node as we go down recursively
def isBST(node, min, max):
 
    # base case
    if node is None:
        return True
 
    # if the node's value falls outside the valid range
    if node.val < min or node.val > max:
        return False
 
    # recursively check left and right subtrees with updated range
    return isBST(node.left, min, node.val) and isBST(node.right, node.val, max)
 
 
# Recursive function to find the size of the largest BST in a given binary tree
def findLargestBST(root):
 
    if isBST(root, -sys.maxsize, sys.maxsize):
        return size(root)
 
    return max(findLargestBST(root.left), findLargestBST(root.right))
 
 
if __name__ == '__main__':
 
    ''' Construct the following tree
              10
            /    \
           /      \
          15       8
         /  \     / \
        /    \   /   \
       12    20 5     2
    '''
 
    root = Node(10)
 
    root.left = Node(15)
    root.right = Node(8)
 
    root.left.left = Node(12)
    root.left.right = Node(20)
 
    root.right.left = Node(5)
    root.right.right = Node(2)
 
    print('The size of the largest BST is', findLargestBST(root))



"""
The time complexity of this approach is O(n2), where n is the size of the BST, 
and requires space proportional to the tree's height for the call stack. 
We can improve time complexity to O(n) by traversing the tree in a bottom-up manner 
where information is exchanged between the child nodes and parent node, 
which helps determine if the subtree rooted under any node is a BST in constant time.

We know that a binary tree is a BST if the following properties hold for every tree node:
1.  The left and right subtrees of every tree node are BST.
2.  A node's value should be more than the largest value in the left subtree and 
    less than the smallest value in the right subtree.
To determine if a subtree rooted under a node is a BST or not, the left subtree should provide 
information about the maximum value in it. The right subtree should provide information about 
the minimum value in it. Also, the parent node should be notified when both left and right child are also BST.
"""

# A class to store information about a binary tree
class SubTreeInfo:
 
    # min, max: stores the minimum and the maximum value rooted under the current node
    # `min`, `max` fields are relevant only if `isBST` flag is true
    # size: stores size of the largest BST rooted under the current node
    # isBST: true if the binary tree rooted under the current node is a BST
 
    def __init__(self, min, max, size, isBST):
        self.min = min
        self.max = max
        self.size = size
        self.isBST = isBST
 
 
# Recursive function to find the size of the largest BST in a given binary tree
def findLargestBST(root):
 
    # Base case: empty tree
    if root is None:
        return SubTreeInfo(sys.maxsize, -sys.maxsize, 0, True)
 
    # Recur for the left and right subtrees
    left = findLargestBST(root.left)
    right = findLargestBST(root.right)
 
    # Check if a binary tree rooted under the current root is a BST
 
    # 1. Left and right subtree are also BST
    # 2. The value of the root node should be more than the largest value
    #    in the left subtree
    # 3. The value of the root node should be less than the smallest value
    #    in the right subtree
    if left.isBST and right.isBST and (left.max < root.val < right.min):
        info = SubTreeInfo(min(root.val, min(left.min, right.min)),
                        max(root.val, max(left.max, right.max)),
                        left.size + 1 + right.size, True)
    else:
 
        # If a binary tree rooted under the current root is not a BST,
        # return the largest BST size in its left and right subtree
 
        info = SubTreeInfo(0, 0, max(left.size, right.size), False)
 
    return info
 
 
if __name__ == '__main__':
 
    ''' Construct the following tree
                  10
                /    \
               /      \
              15       8
             / \      / \
            /   \    /   \
           12   20  5     9
          / \      / \     \
         /   \    /   \     \
        2    14  4    7     10
    '''
 
    root = Node(10)
 
    root.left = Node(15)
    root.right = Node(8)
 
    root.left.left = Node(12)
    root.left.right = Node(20)
    root.right.left = Node(5)
    root.right.right = Node(9)
 
    root.left.left.left = Node(2)
    root.left.left.right = Node(14)
    root.right.left.left = Node(4)
    root.right.left.right = Node(7)
 
    root.right.right.right = Node(10)
 
    print('The size of the largest BST is', findLargestBST(root).size)