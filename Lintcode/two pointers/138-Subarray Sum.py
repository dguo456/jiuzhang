# 138 · Subarray Sum
"""
Given an integer array, find a subarray where the sum of numbers is zero. 
Your code should return the index of the first number and the index of the last number.

Input:  [-3, 1, 2, -3, 4]
Output: [0, 2] or [1, 3].
Explanation: return anyone that the sum is 0.
"""
class Solution:
    """
    @param nums: A list of integers
    @return: A list of integers includes the index of the first number and the index of the last number
    """
    def subarraySum(self, nums):
        if not nums or len(nums) == 0:
            return []

        prefix_hash = {0: -1}
        prefix_sum = 0

        for index, value in enumerate(nums):
            prefix_sum += value
            if prefix_sum in prefix_hash:
                return [prefix_hash[prefix_sum] + 1, index]
            prefix_hash[prefix_sum] = index

        return [-1, -1]





# 404 · Subarray Sum II
"""
Given an positive integer array A and an interval. Return the number of subarrays (注意是subarray，不是排列组合)
whose sum is in the range of given interval.

Input: A = [1, 2, 3, 4], start = 1, end = 3
Output: 4
Explanation: All possible subarrays: [1](sum = 1), [1, 2](sum = 3), [2](sum = 2), [3](sum = 3).
"""

class Solution:
    """
    @param A: An integer array
    @param start: An integer
    @param end: An integer
    @return: the number of possible answer
    """
    def subarraySumII(self, A, start, end):
        n = len(A)
        left_sum, right_sum = 0, 0
        left_index, right_index = 0, 0
        result = 0

        for i in range(n):
            left_index = max(i, left_index)
            right_index = max(i, right_index)

            while left_index < n and left_sum + A[left_index] < start:
                left_sum += A[left_index]
                left_index += 1

            while right_index < n and right_sum + A[right_index] <= end:
                right_sum += A[right_index]
                right_index += 1

            if right_index - left_index > 0:
                result += right_index - left_index

            if left_index > i:
                left_sum -= A[i]

            if right_index > i:
                right_sum -= A[i]

        return result





# 139 · Subarray Sum Closest
"""
Given an integer array, find a subarray with sum closest to zero.
Return the indexes of the first number and last number.

Input:        [-3,1,1,-3,5] 
Output:       [0,2]
Explanation:  [0,2], [1,3], [1,1], [2,2], [0,4]
"""
import sys

class Solution:
    """
    @param: nums: A list of integers
    @return: A list of integers includes the index of the first number and the index of the last number
    """
    def subarraySumClosest(self, nums):
        if not nums or len(nums) == 0:
            return []

        prefix_sum = [(0, -1)]
        for i, num in enumerate(nums):
            # 记住这种构建前缀和列表的方法
            prefix_sum.append((prefix_sum[-1][0]+num, i))
        
        prefix_sum.sort()   # 因为这里需要对前缀和列表进行排序，所以构建成列表而不是dict
        
        closest, result = sys.maxsize, []
        
        for i in range(1, len(prefix_sum)):
            if closest > prefix_sum[i][0] - prefix_sum[i-1][0]:

                closest = prefix_sum[i][0] - prefix_sum[i-1][0]
                left = min(prefix_sum[i][1], prefix_sum[i-1][1]) + 1
                right = max(prefix_sum[i][1], prefix_sum[i-1][1])
                result = [left, right]
                
        return result




# 405 · Submatrix Sum
"""
Given an integer matrix, find a submatrix where the sum of numbers is zero. 
Your code should return the coordinate of the left-up and right-down number.
If there are multiple answers, you can return any of them.

Input:
[
  [1, 5, 7],
  [3, 7, -8],
  [4, -8 ,9]
]
Output: [[1, 1], [2, 2]]
"""

# 用前缀和优化, 令 sum[i][j] = sum[0][j] + sum[1][j] + ... + sum[i][j]
# 然后枚举上下边界, 这样就相当于在一行内, 求一个数组连续子串和为0的问题了.
class Solution:
    """
    @param: matrix: an integer matrix
    @return: the coordinate of the left-up and right-down number
    """
    def submatrixSum(self, matrix):
        if not matrix or not matrix[0]:
            return []

        n, m = len(matrix), len(matrix[0])

        for top_row in range(n):
            array = [0] * m

            for bottom_row in range(top_row, n):
                prefix_hash = {0: -1}
                prefix_sum = 0

                for col in range(m):
                    array[col] += matrix[bottom_row][col]
                    prefix_sum += array[col]

                    if prefix_sum in prefix_hash:
                        return [(top_row, prefix_hash[prefix_sum] + 1), (bottom_row, col)]

                    prefix_hash[prefix_sum] = col

        return []




# 406 · Minimum Size Subarray Sum
"""
Given an array of n positive integers and a positive integer s, find the minimal length 
of a subarray of which the sum ≥ s. If there isn't one, return -1 instead.
Input: [2,3,1,2,4,3], s = 7
Output: 2
Explanation: The subarray [4,3] has the minimal length under the problem constraint.
"""
class Solution:
    """
    @param nums: an array of integers
    @param s: An integer
    @return: an integer representing the minimum size of subarray
    """
    def minimumSize(self, nums, s):
        if not nums or len(nums) == 0:
            return -1

        right = 0
        temp_sum = 0
        result = len(nums) + 1

        for left in range(len(nums)):
            # 这里仔细思考temp_sum < s的条件，可以做成sliding window的感觉，右边停了之后左边收缩
            while right < len(nums) and temp_sum < s:
                temp_sum += nums[right]
                right += 1
            if temp_sum >= s:
                result = min(right - left, result)

            temp_sum -= nums[left]

        if result == len(nums) + 1:
            return -1

        return result