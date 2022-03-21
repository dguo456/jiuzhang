# 989 · Array Nesting
"""
A zero-indexed array A of length N contains all integers from 0 to N-1. Find and return the longest length 
of set S, where S[i] = {A[i], A[A[i]], A[A[A[i]]], ... } subjected to the rule below.

Input: [5,4,0,3,1,6,2]
Output: 4
Explanation: 
A[0] = 5, A[1] = 4, A[2] = 0, A[3] = 3, A[4] = 1, A[5] = 6, A[6] = 2.
One of the longest S[K]:
S[0] = {A[0], A[5], A[6], A[2]} = {5, 6, 2, 0}
"""

class Solution:
    """
    @param nums: an array
    @return: the longest length of set S
    """
    def arrayNesting(self, nums):
        
        result, step, n = 0, 0, len(nums)
        visited = [False] * n

        for i in range(n):

            while not visited[i]:
                visited[i] = True
                i = nums[i]
                step = step + 1
            
            result = max(step, result)
            step = 0

        return result