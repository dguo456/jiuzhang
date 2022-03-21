# 1307 · Verify Preorder Sequence in Binary Search Tree
"""Given an array of numbers, verify whether it is the correct preorder traversal sequence 
    of a binary search tree.    You may assume each number in the sequence is unique."""

import sys

class Solution:
    """
    @param preorder: List[int]
    @return: return a boolean
    """
    def verifyPreorder(self, preorder):
        if not preorder or len(preorder) == 0:
            return True
        
        lower_bound = -sys.maxsize
        stack = []
        stack.append(preorder[0])
        i = 1

        while i < len(preorder):

            if preorder[i] < lower_bound:
                return False

            while stack and preorder[i] > stack[-1]:
                lower_bound = stack[-1]
                stack.pop()

            stack.append(preorder[i])
            i += 1

        return True