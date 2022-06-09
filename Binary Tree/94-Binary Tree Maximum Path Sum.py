# 94 · Binary Tree Maximum Path Sum

"""
Definition of TreeNode:
class TreeNode:
    def __init__(self, val):
        self.val = val
        self.left, self.right = None, None
"""
import sys

class Solution:
    """
    @param root: The root of binary tree.
    @return: An integer
    """
    def maxPathSum(self, root):
        self.maxSum = -sys.maxsize-1 
        self.dfs(root)
        return self.maxSum

    # Method.1      写法1
    def dfs(self, root):
        if root is None:
            return 0
        
        leftSum = self.dfs(root.left)
        rightSum = self.dfs(root.right)

        # 在遍历树的过程中，要看两样，一是当前subtree子树的最大值，一是当前subtree能对parent tree父树贡献什么。
        # 返回的是对parent tree贡献最大的那一支subtree，还要比0大，如果subtree的左右支均为负数则排除
        self.maxSum = max(self.maxSum, root.val + leftSum + rightSum)
        return max(leftSum + root.val, rightSum + root.val, 0)

    
    # Method.2      写法2
    def helper(self, root):
        if not root: 
            return 0 
        
        leftSum = self.helper(root.left)
        if leftSum < 0:
            leftSum = 0
        rightSum = self.helper(root.right)
        if rightSum < 0:
            rightSum = 0
        
        # 三者之和放在全局self.maxSum里。返回的值是可供上一层父节点接龙的值
        # 局部最大值是遍历算法，挑选左右子树最大值是分治法，这里的解法是合二为一
        self.maxSum = max(self.maxSum, root.val + leftSum + rightSum)
        return root.val + max(leftSum, rightSum)



# 475 · Binary Tree Maximum Path Sum II
class Solution:
    """
    @param root: the root of binary tree.
    @return: An integer
    """
    # Method.1      采用遍历做法，每加一个数，都要和全局最优解进行比较
    def maxPathSum2(self, root):
        self.result = -sys.maxsize
        value_sum = 0
        self.dfs(root, value_sum)
        return self.result
        
    def dfs(self, root, value_sum):
        value_sum += root.val
        
        if value_sum > self.result:
            self.result = value_sum 
        
        if root.left is not None:
            self.dfs(root.left, value_sum)
        
        if root.right is not None:
            self.dfs(root.right, value_sum)


    # Method.2      简单版94, 限定必须从root出发, 因此只需一个返回值
    def max_path_sum2(self, root) -> int:
        if not root:
            return -1
        
        result = self.helper(root)
        return result
    
    def helper(self, root):
        if not root:
            return 0
        
        left_max = self.helper(root.left)
        right_max = self.helper(root.right)
        
        max_len = max(left_max + root.val, right_max + root.val, root.val)
        
        return max_len


    # Method.3      同Method2 每次选0、左子树和、右子数和中的最大值和自己加起来即可
    def max_path_sum2(self, root) -> int:
        if not root:
            return 0

        if not root.left and not root.right:
            return root.val

        left_max = self.max_path_sum2(root.left)
        right_max = self.max_path_sum2(root.right)

        max_sum_branch = max(left_max, right_max, 0)

        return root.val + max_sum_branch
