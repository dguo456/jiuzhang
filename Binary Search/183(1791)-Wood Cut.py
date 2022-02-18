# 183 · Wood Cut
"""
Given n pieces of wood with length L[i] (integer array). Cut them into small pieces to guarantee 
you could have equal or more than k pieces with the same length. What is the longest length 
you can get from the n pieces of wood? Given L & k, return the maximum length of the small pieces.

Example 1
Input:
L = [232, 124, 456]
k = 7
Output: 114
Explanation: We can cut it into 7 pieces if any piece is 114cm long, however we can't cut it 
into 7 pieces if any piece is 115cm long.
Big O: O(n log Len), where Len is the longest length of the wood.

Example 2
Input:
L = [1, 2, 3]
k = 7
Output: 0
Explanation: It is obvious we can't make it.
"""
class Solution:
    """
    @param L: Given n pieces of wood with length L[i]
    @param k: An integer
    @return: The maximum length of the small pieces
    """
    def woodCut(self, L, k):
        if not L or len(L) == 0 or sum(L) < k:
            return 0

        start, end = 1, max(L)
        while start + 1 < end:
            mid = (start + end) // 2
            if self.get_pieces(L, mid) >= k:
                start = mid
            else:
                end = mid
                
        if self.get_pieces(L, end) >= k:
            return end
        if self.get_pieces(L, start) >= k:
            return start
            
        return 0
        
    def get_pieces(self, L, length):
        pieces = 0
        for l in L:
            pieces += l // length
        return pieces




# 1791 · Simple queries
"""
Give you two arrays, the first array may have duplicate, The length of the returned array is the same as 
the second array. For each element a in the second array, how many numbers are in the first array <=a.

Input:  nums = [3, 2, 4, 3, 5, 1],sub = [2, 4]
Output:  [2, 5] 
Explanation: <=2 numbers are [1,2]，<=4 numbers are [1,2,3,3,4]
"""
class Solution:
    """
    @param nums: 
    @param sub: 
    @return: return a Integer array
    """
    def SimpleQueries (self, nums, sub):
        if not nums or len(nums) == 0:
            return [0] * len(sub)

        nums.sort()
        results = []
        mmin, mmax = nums[0], nums[-1]

        for s in sub:
            if s < mmin:
                results.append(0)	#如果当前元素小于最小值，存入0
            elif s > mmax:
                results.append(len(nums))  #如果当前元素大于最大值，存入n
            else:
                index = self.BinarySearch(nums, s)	#找到一个小于等于当前数字的位置
                while index < len(nums) and nums[index] == s:	#如果相等就继续增加答案
                    index += 1
                results.append(index)
        return results

    def BinarySearch(self, nums, target):    	#二分查找
        start, end = 0, len(nums) - 1

        while start + 1 < end:
            mid = (start + end) // 2
            if nums[mid] == target:
                return mid
            elif nums[mid] < target:
                start = mid
            else:
                end = mid

        if nums[mid] < target:
            return mid + 1
        return mid