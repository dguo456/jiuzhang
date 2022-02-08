# 172 · Remove Element              Move Zero简化版 不要求保证移动完之后元素顺序
"""
Given an array and a value, remove all occurrences of that value in place and return the new length.
The order of elements can be changed, and the elements after the new length don't matter.
"""

def removeElement(A, elem):
        if not A or len(A) == 0:
            return 0

        # Method.1      从后往前
        # right = len(A)-1
        # for left in range(len(A) - 1, -1, -1):
        #     if A[left] == elem:
        #         A[left], A[right] = A[right], A[left]
        #         right -= 1

        # return right+1

        # Method.2      从前往后
        left = 0
        for right in range(len(A)):
            if A[right] == elem:
                A[left], A[right] = A[right], A[left]
                left += 1

        return A[left:]



# 539 · Move Zeroes                 要求保证移动完之后元素顺序不变
"""
Description
Given an array nums, write a function to move all 0's to the end of it while maintaining the 
relative order of the non-zero elements.

You must do this in-place without making a copy of the array.
Minimize the total number of operations.

Example 1:
Input: nums = [0, 1, 0, 3, 12],
Output: [1, 3, 12, 0, 0].

Example 2:
Input: nums = [0, 0, 0, 3, 1],
Output: [3, 1, 0, 0, 0].
"""
# Method.1
class Solution:
    """
    @param nums: an integer array
    @return: nothing
    """
    def moveZeroes(self, nums):
        if not nums or len(nums) == 0:
            return []

        pivot = 0
        for index in range(len(nums)):
            if nums[index] != 0:
                nums[pivot], nums[index] = nums[index], nums[pivot]
                pivot += 1

        return nums


# Method.2      
class Solution:
    """
    @param nums: an integer array
    @return: nothing
    """
    def moveZeroes(self, nums):
        if not nums or len(nums) < 2:
            return nums
        
        count = 0
        left, right = 0, 0

        while right < len(nums):
            if nums[right] != 0:
                nums[left] = nums[right]
                left += 1
            else:
                count += 1
            right += 1

        if count != 0:
            for i in range(count):
                nums[len(nums) - i - 1] = 0



# 1886 · Moving Target
"""
Given an array nums and an integer target ,
you need to move the element which equal to target to the front of the array, and the order of 
the remain elements can not be changed.
All your move operations should be performed on the original array.
"""
# Example 1:
# Input:
# nums = [5, 1, 6, 1]
# target=1
# Output: 
# [1, 1, 5, 6]

from typing import (
    List,
)

class Solution:
    """
    @param nums: a list of integer
    @param target: an integer
    @return: nothing
    """
    def MoveTarget(self, nums: List[int], target: int):
        count = 0
        left, right = len(nums) - 1, len(nums) - 1

        while left >= 0:
            if nums[left] != target:
                nums[right] = nums[left]
                right -= 1
            else:
                count += 1
            left -= 1

        for i in range(count):
            nums[i] = target