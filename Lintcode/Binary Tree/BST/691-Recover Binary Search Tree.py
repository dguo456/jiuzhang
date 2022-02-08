# 691 · Recover Binary Search Tree
"""In a binary search tree, (Only) two nodes are swapped. 
Find out these nodes and swap them. If there no node swapped, return original root of tree."""

class Solution:
    """
    @param root: the given tree
    @return: the tree after swapping
    """
    def bstSwappedNode(self, root):
        if root is None:
            return None

        self.prev = None
        self.first, self.second = None, None
        self.dfs(root)

        # 只需交换两个node的val
        if self.first and self.second:
            self.first.val, self.second.val = self.second.val, self.first.val

        return root

    def dfs(self, root):
        if root is None:
            return

        self.dfs(root.left)

        if self.prev:
            if self.first is None and self.prev.val > root.val:
                self.first = self.prev
                self.second = root      # 这里注意，如果要交换的两点是父子关系，就必须得把second赋值成当前root
            elif self.first and self.prev.val > root.val:
                self.second = root
        self.prev = root

        self.dfs(root.right)