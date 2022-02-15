# 39 · Recover Rotated Sorted Array
"""Given a rotated sorted array, return it to sorted array in-place.（Ascending）"""

class Solution:
    """
    @param nums: An integer array
    @return: nothing
    """
    def recoverRotatedSortedArray(self, nums):
        if not nums or len(nums) < 2:
            return nums

        if nums[0] < nums[-1]:
            return nums

        for i in range(1, len(nums)):
            if nums[i] < nums[i-1]:
                # 三部翻转法，注意最后整体翻转要nums[:]
                nums[:i] = nums[:i][::-1]
                nums[i:] = nums[i:][::-1]
                nums[:] = nums[::-1]

        return nums