# 900 · Closest Binary Search Tree Value (same with 1033 and 1188 and 1746)
"""
Given a non-empty binary search tree and a target value, find the value in the BST that is closest to the target.

Given target value is a floating point.
You are guaranteed to have only one unique value in the BST that is closest to the target.
"""

# Definition of TreeNode:
class TreeNode:
    def __init__(self, val):
        self.val = val
        self.left, self.right = None, None

class Solution:
    """
    @param root: the given BST
    @param target: the given target
    @return: the value in the BST that is closest to the target
    """
    def closestValue(self, root, target):
        if root is None:
            return None

        lower_bound = self.get_lower_bound(root, target)
        upper_bound = self.get_upper_bound(root, target)

        if lower_bound is None:
            return upper_bound.val
        if upper_bound is None:
            return lower_bound.val

        if target - lower_bound.val <= upper_bound.val - target:
            return lower_bound.val
        else:
            return upper_bound.val

    def get_lower_bound(self, root, target):
        if root is None:
            return

        if root.val > target:
            return self.get_lower_bound(root.left, target)      # 这里注意要return
        else:
            right = self.get_lower_bound(root.right, target)

        return root if not right else right

    def get_upper_bound(self, root, target):
        if root is None:
            return

        if root.val < target:
            return self.get_upper_bound(root.right, target)     # 这里注意要return
        else:
            left = self.get_upper_bound(root.left, target)
            
        return root if not left else left




# 901 · Closest Binary Search Tree Value II
class Solution:
    """
    @param root: the given BST
    @param target: the given target
    @param k: the given k
    @return: k values in the BST that are closest to the target
    """
    def closestKValues(self, root, target, k):
        if root is None:
            return None

        tree_list = self.convert_tree_to_list(root)
        left_index, right_index = self.binary_search(tree_list, target)

        return self.get_K_closest_vals(tree_list, target, k, left_index, right_index)

    def convert_tree_to_list(self, root):
        if root is None:
            return []

        left = self.convert_tree_to_list(root.left)
        right = self.convert_tree_to_list(root.right)

        return left + [root.val] + right

    def binary_search(self, tree_list, target):
        left, right = 0, len(tree_list) - 1

        while left + 1 < right:
            mid = (left + right) // 2
            if tree_list[mid] < target:
                left = mid
            else:
                right = mid

        return [left, right]

    def get_K_closest_vals(self, tree_list, target, k, left_index, right_index):
        
        results = []
        while left_index >= 0 and right_index <= len(tree_list) - 1 and k > 0:
            if target - tree_list[left_index] < tree_list[right_index] - target:
                results.append(tree_list[left_index])
                left_index -= 1
            else:
                results.append(tree_list[right_index])
                right_index += 1
            
            k -= 1

        while left_index >= 0 and k > 0:
            results.append(tree_list[left_index])
            left_index -= 1
            k -= 1

        while right_index <= len(tree_list) - 1 and k > 0:
            results.append(tree_list[right_index])
            right_index += 1
            k -= 1

        return results