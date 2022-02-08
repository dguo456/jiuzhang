# 689 · Two Sum IV - Input is a BST
class Solution:
    """
    @param: : the root of tree
    @param: : the target sum
    @return: two numbers from tree which sum is n
    """

    def twoSum(self, root, n):
        if root is None:
            return None

        node_set = set()
        self.results = None
        self.dfs(root, n, node_set)
        return self.results

    def dfs(self, root, n, node_set):
        if root is None:
            return

        self.dfs(root.left, n, node_set)

        if n - root.val in node_set:
            self.results = [n - root.val, root.val]
        else:
            node_set.add(root.val)

        self.dfs(root.right, n, node_set)