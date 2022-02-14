# 838 · Subarray Sum Equals K
"""
Given an array of integers and an integer k, you need to find the total number of 
continuous subarrays whose sum equals to k.

Input: nums = [2,1,-1,1,2] and k = 3
Output: 4
Explanation:
subarray [0,1], [1,4], [0,3] and [3,4]
"""

# 计算Prefix_sum的过程中，
# 用HashMap 记录当前prefix_sum出现的次数，
# 当prefix_sum - k 出现在HashMap中，叠加count
# 时间复杂度为O(n)

class Solution:
    """
    @param nums: a list of integer
    @param k: an integer
    @return: return an integer, denote the number of continuous subarrays whose sum equals to k
    """
    def subarraySumEqualsK(self, nums, k):
        if not nums or len(nums) == 0:
            return 0

        count = 0 
        prefix_sum = 0
        prefix_hash = {0: 1}    # 这里key是sum，value是出现次数，看返回值如果是count，value就定为出现次数，
                                # 如果是返回区间index，则初始化value为index
        for num in nums:
            prefix_sum += num 
            if prefix_sum - k in prefix_hash:
                count += prefix_hash[prefix_sum - k]
            if prefix_sum in prefix_hash:
                prefix_hash[prefix_sum] += 1 
            else:
                prefix_hash[prefix_sum] = 1 
            
        return count




# 1844 · subarray sum equals to k II
"""
Given an array of integers and an integer k, you need to find the minimum size of continuous 
non-empty subarrays whose sum equals to k, and return its length.
if there are no such subarray, return -1.

Input: 
nums = [2,1,-1,4,2,-3] and k = 3
Output: 2
"""
class Solution:
    """
    @param nums: a list of integer
    @param k: an integer
    @return: return an integer, denote the minimum length of continuous subarrays whose sum equals to k
    """
    def subarraySumEqualsKII(self, nums, k):
        if not nums or len(nums) == 0:
            return 0

        min_length = len(nums) + 1
        prefix_sum = 0
        prefix_hash = {0: -1}

        for i in range(len(nums)):
            prefix_sum += nums[i]
            if prefix_sum - k in prefix_hash:
                min_length = min(i - prefix_hash[prefix_sum - k], min_length)
            prefix_hash[prefix_sum] = i
        
        # 这里注意因为题目要求没找到返回-1，所以需要判断返回值的结果是否被更新
        if min_length == len(nums) + 1:
            return -1
        
        return min_length




# 911 · Maximum Size Subarray Sum Equals k
"""
Given an array nums and a target value k, find the maximum length of a subarray that sums to k. 
If there isn't one, return 0 instead.

Input:  nums = [1, -1, 5, -2, 3], k = 3
Output: 4
Explanation:
because the subarray [1, -1, 5, -2] sums to 3 and is the longest.
"""
class Solution:
    """
    @param nums: an array
    @param k: a target value
    @return: the maximum length of a subarray that sums to k
    """
    def maxSubArrayLen(self, nums, k):
        if not nums or len(nums) == 0:
            return 0

        max_length = 0
        prefix_sum = 0
        prefix_hash = {0: -1}

        for i in range(len(nums)):
            prefix_sum += nums[i]
            if prefix_sum - k in prefix_hash:
                max_length = max(i - prefix_hash[prefix_sum - k], max_length)
            # 这里注意，因为是求最大长度，所以如果key已经存在了就不能更新其value，否则就不是最大长度了
            if prefix_sum in prefix_hash:
                continue 
            prefix_hash[prefix_sum] = i

        return max_length