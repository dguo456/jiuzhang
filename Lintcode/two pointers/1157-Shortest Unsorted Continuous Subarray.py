# 1157 · Shortest Unsorted Continuous Subarray
"""
Given an integer array, you need to find one continuous subarray that if you only sort this subarray 
in ascending order, then the whole array will be sorted in ascending order, too.
You need to find the shortest such subarray and output its length.

Input: [2, 6, 4, 8, 10, 9, 15]
Output: 5
Explanation: You need to sort [6, 4, 8, 10, 9] in ascending order to make the whole array sorted in ascending order.
"""

# Method.1      Greedy Search
class Solution:
    """
    @param nums: an array
    @return: the shortest subarray's length
    """
    def findUnsortedSubarray(self, nums):
        sorted_nums = list(sorted(nums))
        start, end = 0, 0
        
        for i in range(len(nums)):
            if nums[i] != sorted_nums[i]:
                start = i
                break       # 注意这里，找到了就break

        for i in range(len(nums) - 1, -1, -1):
            if nums[i] != sorted_nums[i]:
                end = i
                break       # 注意这里，找到了就break

        # 注意这里不能有等于的情况
        if end > start:
            return end - start + 1

        return 0



# Method.2      利用排序树组的性质：左边的最大值小于右边的最小值，右边的最小值大于左边的最大值
class Solution:
    """
    @param nums: an array
    @return: the shortest subarray's length
    """
    def findUnsortedSubarray(self, nums):
        if not nums or len(nums) < 2:
            return 0

        left_max, right_min = nums[0], nums[-1]
        start, end = 0, 0

        for i in range(len(nums)):
            left_max = max(nums[i], left_max)
            if left_max > nums[i]:
                end = i

        for i in range(len(nums) - 1, -1, -1):
            right_min = min(nums[i], right_min)
            if right_min < nums[i]:
                start = i

        if start < end:
            return end - start + 1

        return 0