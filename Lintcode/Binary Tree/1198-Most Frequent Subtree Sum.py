# 1198 · Most Frequent Subtree Sum
"""
Given the root of a tree, you are asked to find the most frequent subtree sum. 
The subtree sum of a node is defined as the sum of all the node values formed by the subtree 
rooted at that node (including the node itself). So what is the most frequent subtree sum value? 
If there is a tie, return all the values with the highest frequency in any order.

Input:
{5,2,-3}
Output:
[-3,2,4]
Explanation:
  5
 /  \
2   -3
since all the values happen only once, return all of them in any order.
"""

# Definition of TreeNode:
class TreeNode:
    def __init__(self, val):
        self.val = val
        self.left, self.right = None, None

class Solution:
    """
    @param root: the root
    @return: all the values with the highest frequency in any order
    """
    def findFrequentTreeSum(self, root):
        if root is None:
            return []

        self.value2count = {}
        self.max_count = 0

        self.helper(root)

        results = []
        for curr_sum, count in self.value2count.items():
            if count == self.max_count:
                results.append(curr_sum)

        return results

    def helper(self, root):
        if not root:
            return 0

        left_sum = self.helper(root.left)
        right_sum = self.helper(root.right)

        total_sum = left_sum + right_sum + root.val

        self.value2count[total_sum] = self.value2count.get(total_sum, 0) + 1
        self.max_count = max(self.value2count[total_sum], self.max_count)

        return total_sum