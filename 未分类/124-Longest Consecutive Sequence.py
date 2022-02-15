# 124 · Longest Consecutive Sequence
"""
Given an unsorted array num of integers, find the length of the longest consecutive elements sequence.

Input:  num = [100, 4, 200, 1, 3, 2]
Output: 4
Explanation:
The longest consecutive elements sequence is [1, 2, 3, 4]. Return its length:4
"""

class Solution:
    """
    @param num: A list of integers
    @return: An integer
    """
    def longestConsecutive(self, nums):
        hashset = set(nums)
        longest = 0

        for num in nums:
            down = num - 1
            while down in hashset:
                hashset.discard(down)
                down -= 1

            up = num + 1
            while up in hashset:
                hashset.discard(up)
                up += 1

            longest = max(longest, up - down - 1)

        return longest