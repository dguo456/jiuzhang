# 100 · Remove Duplicates from Sorted Array
"""
Given a sorted array, 'remove' the duplicates in place such that each element appear only once 
and return the 'new' length.
Do not allocate extra space for another array, you must do this in place with constant memory.
Input:  nums = []       Output: 0
"""
class Solution:
    """
    @param: nums: An ineger array
    @return: An integer
    """
    def removeDuplicates(self, nums):
        if not nums or len(nums) == 0:
            return 0
            
        index = 0
        for i in range(1, len(nums)):
            if nums[i] != nums[index]:
                index += 1
                nums[index] = nums[i]
                
        return index+1




# 101 · Remove Duplicates from Sorted Array II
class Solution:
    """
    @param: nums: An ineger array
    @return: An integer
    """
    def removeDuplicates(self, nums):
        if not nums and len(nums) == 0:
            return 0
        if len(nums) == 1:
            return 1
        
        results = []
        prev = None
        count = 0
        
        for num in nums:
            if num != prev:
                results.append(num)
                prev = num
                count = 1
            elif count < 2:
                results.append(num)
                count += 1
        
        res_len = 0
        for num in results:
            nums[res_len] = num
            res_len += 1
            
        return res_len