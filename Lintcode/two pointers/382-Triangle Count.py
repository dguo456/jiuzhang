# 382 · Triangle Count
"""
Given an array of integers, how many three numbers can be found in the array, so that we can 
build an triangle whose three edges length is the three numbers that we find?
Input: [3, 4, 6, 7]
Output: 3
Explanation:
They are (3, 4, 6), (3, 6, 7), (4, 6, 7)
"""

class Solution:
    """
    @param S: A list of integers
    @return: An integer
    """
    def triangleCount(self, S):
        if S is None or len(S) < 3:
            return 0
            
        S.sort()
        result = 0
        
        for i in range(2, len(S)):

            left, right = 0, i-1
            
            while left < right:
                if S[left] + S[right] > S[i]:
                    result += right - left    # 注意这里直接count所有right-left种可能
                    right -= 1
                else:
                    left += 1
                    
        return result