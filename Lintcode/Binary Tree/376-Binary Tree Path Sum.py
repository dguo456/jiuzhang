# 376 · Binary Tree Path Sum
"""
Definition of TreeNode:
class TreeNode:
    def __init__(self, val):
        self.val = val
        self.left, self.right = None, None
"""
class Solution:
    """
    @param: root: the root of binary tree
    @param: target: An integer
    @return: all valid paths
    """
    # Traversal
    def binaryTreePathSum(self, root, target):
        results = []
        self.dfs(root, target, [], results)
        return results
        
    def dfs(self, root, target, path, results):
        if root is None:
            return
        path.append(root.val)
        
        if root.left is None and root.right is None and root.val == target:
            results.append(path[:])
        
        # 在target上做减法可以使用全局变量root.val，不用分别对左子树右子树分开操作，对比 Method2
        if root.left:
            self.dfs(root.left, target-root.val, path, results)
        if root.right:
            self.dfs(root.right, target-root.val, path, results)
        
        path.pop() # need to pop before backtrack

    # Traversal II
    def binaryTreePathSumII(self, root, target):
        if not root:
            return []
        result = []
        self.traversal(root, target, [root.val], result)
        return result

    def traversal(self, root, target, path, result):
        if root is None:
            return
        if root.left is None and root.right is None and sum(path) == target:
            result.append(path)

        # 因为这里分成左右子树分别遍历，所以不能path+[root.val]，所以就得在主函数中append进root.val，对比permutation
        if root.left:
            self.traversal(root.left, target, path+[root.left.val], result)
        if root.right:
            self.traversal(root.right, target, path+[root.right.val], result)


# 246 · Binary Tree Path Sum II
"""
主函数采用Divide and Conquer, 即所有的解的集合是从左子树节点出发返回的解的集合 + 从右子树节点返回的解的集合 + 从根节点本身出发的集合。
左右子树很简单，递归调用自个就行了
从根节点出发的解的集合其实就是 376. Binary Tree Path Sum 这道题的解，稍作修改，把叶子节点的条件改成任意节点就行了
"""
class Solution:
    """
    @param: root: the root of binary tree
    @param: target: An integer
    @return: all valid paths
    """
    def binaryTreePathSum2(self, root, target):
        if root is None:
            return []

        from_left = self.binaryTreePathSum2(root.left, target)
        from_right = self.binaryTreePathSum2(root.right, target)
        from_root = self.binaryTreePathSum(root, target)

        return from_left + from_right + from_root

    # Same with 376
    def binaryTreePathSum(self, root, target):
        results = []
        self.dfs(root, target, [root.val], results)
        return results

    def dfs(self, root, target, path, results):
        if root is None:
            return
        # path.append(root.val)
        if sum(path) == target:
            results.append(path[:])

        if root.left:
            self.dfs(root.left, target, path + [root.left.val], results)
        if root.right:
            self.dfs(root.right, target, path + [root.right.val], results)

        # path.pop()

# Method.2
class Solution:
    # @param {TreeNode} root the root of binary tree
    # @param {int} target an integer
    # @return {int[][]} all valid paths
    def binaryTreePathSum2(self, root, target):
        # Write your code here
        result = []
        path = []
        if root is None:
            return result
        self.dfs(root, path, result, 0,  target)
        return result

    def dfs(self, root, path, result, depth, target):
        if root is None:
            return
        path.append(root.val)
        tmp = target
        for i in range(depth , -1, -1):
            tmp -= path[i]
            if tmp == 0:
                result.append(path[i:])

        self.dfs(root.left, path, result, depth + 1, target)
        self.dfs(root.right, path, result, depth + 1, target)

        path.pop()



# 472 · Binary Tree Path Sum III
"""
246加强版, DFS twice + visited
left, right, parent
"""

class Solution:
    """
    @param: root: the root of binary tree
    @param: target: An integer
    @return: all valid paths
    """
    def binaryTreePathSum3(self, root, target):
        results = []
        self.dfs(root, target, results)
        
        return results
        
    def dfs(self, root, target, results):
        if root is None:
            return
        
        path = []
        self.findSum(root, None, target, path, results)
        
        self.dfs(root.left, target, results)
        self.dfs(root.right, target, results)
        
    def findSum(self, root, father, target, path, results):
        path.append(root.val)
        target -= root.val
        
        if target == 0:
            results.append(path[:])
        
        if root.parent not in [None, father]:
            self.findSum(root.parent, root, target, path, results)
            
        if root.left not in [None, father]:
            self.findSum(root.left, root, target, path, results)
            
        if root.right not in [None, father]:
            self.findSum(root.right, root, target, path, results)
            
        path.pop()



# 863 · Binary Tree Path Sum IV
"""
最基本的做法就是直接用二维数组保存这棵树, 很容易根据列表解析出来这棵二叉树.
然后进行一次遍历就可以得到答案.
"""

class Solution:
    """
    @param nums: a list of integers
    @return: return an integer
    """
    def pathSum(self, nums):
        l, size = len(nums), 1
        self.ans, self.depth = 0, nums[-1] // 100
        
        for i in range(self.depth - 1):
            size *= 2
            
        g = [[-1 for j in range(size)] for i in range(self.depth)]
        
        for i in range(l):
            dep, pos = nums[i] // 100, (nums[i] // 10) % 10
            g[dep-1][pos-1] = nums[i] % 10
            
        self.dfs(g, 0, 0, 0)
        return self.ans
        
    def dfs(self, g, d, p, sum):
        if g[d][p] == -1:
            return
        
        sum += g[d][p]
        
        if (d == self.depth - 1 or (g[d+1][2*p] == -1 and g[d+1][2*p+1] == -1)):
            self.ans += sum
            return
        
        self.dfs(g, d+1, 2*p, sum)
        self.dfs(g, d+1, 2*p+1, sum)