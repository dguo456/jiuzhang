# 163 · Unique Binary Search Trees
"""
一棵树由根节点，左子树和右子树构成。
对于目标n，根节点可以是1, 2, ..., n中的任意一个，假设根节点为k，那么左子树的可能性就是numTrees(k-1)种，右子树的可能性就是numTrees(n-k)种，
他们的乘积就根节点为k时整个树的可能性。把所有k的可能性累加就是最终结果。
"""
class Solution:
    # @paramn n: An integer
    # @return: An integer
    def numTrees(self, n):
        ans = {0: 1,
               1: 1,
               2: 2}
        return self.helper(n, ans)
    
    def helper(self, n, ans):
        if n in ans:
            return ans[n]
        else:
            # for each root node, there are 
            # (numTrees(left_subtree)) * (numTrees(right_subtree)) unique BST's
            res = 0
            for i in range(n):
                res += self.helper(i, ans) * self.helper(n - i - 1, ans)
            ans[n] = res
            return res

# Method.2 Dynamic Programming
class Solution:
    # @paramn n: An integer
    # @return: An integer
    def numTrees(self, n):
        # write your code here
        dp = [1, 1, 2]
        if n <= 2:
            return dp[n]
        else:
            dp += [0 for i in range(n-2)]
            for i in range(3, n + 1):
                for j in range(1, i+1):
                    dp[i] += dp[j-1] * dp[i-j]
            return dp[n]


# 164. Unique Binary Search Trees II
class TreeNode:
    def __init__(self, val):
        self.val = val
        self.left, self.right = None, None

class Solution:
    # @paramn n: An integer
    # @return: A list of root
    def generateTrees(self, n):
        # write your code here
        return self.dfs(1, n)
        
    def dfs(self, start, end):
        if start > end: return [None]
        res = []
        for rootval in range(start, end+1):
            LeftTree = self.dfs(start, rootval-1)
            RightTree = self.dfs(rootval+1, end)
            for i in LeftTree:
                for j in RightTree:
                    root = TreeNode(rootval)
                    root.left = i
                    root.right = j
                    res.append(root)
        return res
