# 383 · Container With Most Water
"""
Given n non-negative integers a1, a2, ..., an, where each represents a point at coordinate (i, ai). 
n vertical lines are drawn such that the two endpoints of line i is at (i, ai) and (i, 0). 
Find two lines, which together with x-axis forms a container, such that the container contains the most water.
Input: [1, 3, 2]
Output: 2
Explanation:
Selecting a1, a2, capacity is 1 * 1 = 1
Selecting a1, a3, capacity is 1 * 2 = 2
Selecting a2, a3, capacity is 2 * 1 = 2
"""
class Solution:
    """
    @param heights: a vector of integers
    @return: an integer
    """
    def maxArea(self, heights):
        if not heights or len(heights) < 2:
            return 0

        left, right = 0, len(heights) - 1
        max_capacity = 0

        while left < right:
            if heights[left] < heights[right]:
                capacity = heights[left] * (right - left)
                left += 1
            else:
                capacity = heights[right] * (right - left)
                right -= 1
            
            max_capacity = max(capacity, max_capacity)

        return max_capacity