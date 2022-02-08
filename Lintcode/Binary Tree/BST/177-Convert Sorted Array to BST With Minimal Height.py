# 177 · Convert Sorted Array to Binary Search Tree With Minimal Height. (1359)
"""Given a sorted (increasing order) array, Convert it to a binary search tree with minimal height."""

# Definition of TreeNode:
class TreeNode:
    def __init__(self, val):
        self.val = val
        self.left, self.right = None, None

class Solution:
    """
    @param: A: an integer array
    @return: A tree node
    """
    def sortedArrayToBST(self, A):
        if A is None:
            return None

        return self.dfs(A, 0, len(A)-1)

    def dfs(self, A, start, end):
        if start > end:
            return

        if start == end:
            return TreeNode(A[start])

        mid = (start + end) // 2

        root = TreeNode(A[mid])
        root.left = self.dfs(A, start, mid - 1)
        root.right = self.dfs(A, mid + 1, end)

        return root



# Method.2
class Solution:
    """
    @param nums: the sorted array
    @return: the root of the tree
    """
    def convertSortedArraytoBinarySearchTree(self, nums):
        if not nums or len(nums) == 0:
            return None

        left, right = 0, len(nums) - 1
        mid = (left + right) // 2

        root = TreeNode(nums[mid])
        root.left = self.convertSortedArraytoBinarySearchTree(nums[:mid])
        root.right = self.convertSortedArraytoBinarySearchTree(nums[mid+1:])

        return root