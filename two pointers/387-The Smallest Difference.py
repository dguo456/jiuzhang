# 387 · The Smallest Difference
"""
Given two array of integers(the first array is array A, the second array is array B), 
now we are going to find a element in array A which is A[i], and another element in array B which is B[j], 
so that the difference between A[i] and B[j] is as small as possible, return their smallest difference.

Input: A = [3, 6, 7, 4], B = [2, 8, 9, 3]
Output: 0
Explanation: A[0] - B[3] = 0
"""
import sys

class Solution:
    """
    @param A: An integer array
    @param B: An integer array
    @return: Their smallest difference.
    """
    def smallestDifference(self, A, B):
        index_A, index_B = 0, 0
        result = sys.maxsize

        A.sort()
        B.sort()
        while index_A < len(A) and index_B < len(B):

            result = min(abs(A[index_A] - B[index_B]), result)

            if A[index_A] < B[index_B]:
                index_A += 1
            else:
                index_B += 1

        return result