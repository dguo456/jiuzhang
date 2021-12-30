# 126 · Max Tree
"""
Description
Given an integer array with no duplicates. A max tree building on this array is defined as follow:

The root is the maximum number in the array
The left subtree and right subtree are the max trees of the subarray divided by the root number.
Construct the max tree by the given array.
"""

# Definition of TreeNode:
class TreeNode:
    def __init__(self, val):
        self.val = val
        self.left, self.right = None, None


# Method.1      LintCode Official上的分治答案，但是会超时
class Solution:
    """
    @param A: Given an integer array with no duplicates.
    @return: The root of max tree.
    """
    def maxTree(self, A):
        if not A:
            return None

        max_pos = 0
        for i in range(len(A)):
            if A[i] > A[max_pos]:
                max_pos = i

        root = TreeNode(A[max_pos])
        if max_pos > 0:
            root.left = self.dfs(A, 0, max_pos - 1)
        if max_pos < len(A) - 1:
            root.right = self.dfs(A, max_pos + 1, len(A) - 1)

        return root

    def dfs(self, A, start, end):
        # 这里没有跳出条件，用下面的if来决定要不要递归到下一层

        max_pos = start
        for i in range(start, end + 1):
            if A[i] > A[max_pos]:
                max_pos = i

        root = TreeNode(A[max_pos])
        if start < max_pos:
            root.left = self.dfs(A, start, max_pos-1)
        if end > max_pos:
            root.right = self.dfs(A, max_pos + 1, end)

        return root




# Method.2      单调栈
"""
使用九章算法强化班中讲到的单调栈。保存一个单调递减栈。每个数从栈中被 pop 出的时候，就知道它往左和往右的第一个比他大的数的位置了。
时间复杂度 O(n)，而暴力算法最坏情况下会有 O(n^2)
利用数组实现基本数据结构的调整，当前遍历到的数字比stack中的最后一个大时，将stack中的最后一个数字转变为当前节点的左子树，
循环调整至stack为空或者stack中的最后节点值大于新节点的值。如果stack不为空，说明stack中的最后一个节点值大于新节点值，
则将新节点设为stack中的最后一个节点的右子树，将新节点存入stack。
"""
class Solution:
    """
    @param A: Given an integer array with no duplicates.
    @return: The root of max tree.
    """
    def maxTree(self, A):
        stack = []
        for num in A:
            node = TreeNode(num)		                #新建节点
            while stack and num > stack[-1].val:		#如果stack中的最后一个节点比新节点小
                node.left = stack.pop()					#当前新节点的左子树为stack的最后一个节点
                
            if stack:									#如果stack不为空
                stack[-1].right = node					#将新节点设为stack最后一个节点的右子树
                
            stack.append(node)

        return stack[0]