# 362 · Sliding Window Maximum
"""
Given an array of n integer with duplicate number, and a moving window(size k), 
move the window at each iteration from the start of the array, find the maximum number 
inside the window at each moving.

Input:
[1,2,7,7,8]     3
输出:
[7,7,8]
"""
# 使用九章算法强化班中讲过的单调的双端队列
from collections import deque

class Solution:
    """
    @param: nums: A list of integers
    @param: k: An integer
    @return: The maximum number inside the window at each moving
    """
    def max_sliding_window(self, nums, k):
        if not nums or k <= 0:
            return []
        if k >= len(nums):
            return [max(nums)]

        dq = deque([])  # Deque -- 双端队列 (Doubly Ended Queue), deque+popleft相当于栈stack

        for i in range(k-1):
            self.push(dq, nums, i)

        results = []
        for i in range(k-1, len(nums)):
            self.push(dq, nums, i)
            results.append(nums[dq[0]])
            if dq[0] == i - (k - 1):
                dq.popleft()

        return results

    def push(self, dq, nums, index):
        while dq and nums[dq[-1]] < nums[index]:
            dq.pop()
        dq.append(index)