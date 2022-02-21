# subarray系列包含 - 双指针、前缀和、动态规划、以及隔板法等算法的综合
# 138、404、139、838、1844、911、994、405、406、191、1075、41、42、43、44、45、620，621

# 如果一道题你可以用暴力解决出来，而且题目恰好有连续的限制， 那么滑动窗口和前缀和等技巧就应该被想到。
# 关键是，如何快速得到某个子数组的和呢，比如说给你一个数组 nums，让你实现一个接口 sum(i, j)，这个接口要返回 nums[i..j] 的和，
# 而且会被多次调用，你怎么实现这个接口呢？因为接口要被多次调用，显然不能每次都去遍历 nums[i..j]，
# 有没有一种快速的方法在 O(1) 时间内算出 nums[i..j] 呢？这就需要前缀和技巧了。 主要用于处理数组区间的问题。


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




# 994 · Contiguous Array
"""
Given a binary array, find the maximum length of a contiguous subarray with equal number of 0 and 1.
Input: [0,1]
Output: 2
Explanation: [0, 1] is the longest contiguous subarray with equal number of 0 and 1.
"""
# 把0换成-1，转化为 sum=0 问题
class Solution:
    """
    @param nums: a binary array
    @return: the maximum length of a contiguous subarray
    """
    def findMaxLength(self, nums):
        prefix_sum, longest = 0, 0
        prefix_hash = {0: -1}
        
        for i, num in enumerate(nums):
            prefix_sum += num if num == 1 else -1
            
            if prefix_sum in prefix_hash:
                longest = max(i - prefix_hash[prefix_sum], longest)
            else:
                prefix_hash[prefix_sum] = i 
                
        return longest





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




# 191 · Maximum Product Subarray
"""
Find the contiguous subarray within an array (containing at least one number) which has the largest product.

Example 1:
Input:[2,3,-2,4]
Output:6

Example 2:
Input:[-1,2,4,1]
Output:8
"""

# 线性dp。由于数据中有正有负，所以我们利用两个dp数组来完成。用f[i]来保存计算到第i个时的最大值，用g[i]来保持计算到第i个时的最小值。
# 在得出了第i-1的dp值之后，即包含i-1的可能值区间为[g[i-1],f[i-1]]（左闭右闭区间）。我们考虑第i个数的情况。
#   若nums[i]为正数，可能值区间为[g[i-1]×nums[i],f[i-1]×nums[i]]，和nums[i]。
#   若nums[i]为负数，可能值区间为[f[i-1]×nums[i],g[i-1]×nums[i]]，和nums[i]。
# 所以我们直接根据上述的情况，就能得出包含nums[i]的最大值f[i]=max(max(f[i-1]×nums[i], g[i-1]×nums[i]), nums[i])。
# 同理，g[i]=min(min(f[i-1] x ×[i], g[i-1] × nums[i]), nums[i])。

# 复杂度分析
# 时间复杂度O(n)
# 枚举了数组的长度
# 空间复杂度O(n)
# 消耗了等长的空间

class Solution:
    """
    @param nums: An array of integers
    @return: An integer
    """
    def maxProduct(self, nums):
        if not nums:
            return None
            
        global_max = prev_max = prev_min = nums[0]
        for num in nums[1:]:
            if num > 0:
                curt_max = max(num, prev_max * num)
                curt_min = min(num, prev_min * num)
            else:
                curt_max = max(num, prev_min * num)
                curt_min = min(num, prev_max * num)
            
            global_max = max(global_max, curt_max)
            prev_max, prev_min = curt_max, curt_min
            
        return global_max




# 1075 · Subarray Product Less Than K
"""
Your are given an array of positive integers nums. Count and print the number of 
(contiguous) subarrays where the product of all the elements in the subarray is less than k.

Example 1:
	Input:  nums = [10, 5, 2, 6], k = 100
	Output:  8
	
	Explanation:
	The 8 subarrays that have product less than 100 are: [10], [5], [2], [6], [10, 5], [5, 2], [2, 6], [5, 2, 6].
	Note that [10, 5, 2] is not included as the product of 100 is not strictly less than k.
"""

# 算法：双指针/滑动窗口    利用滑动窗口的方法，维护一个乘积小于k的窗口，窗口大小等于该窗口内子数组的数量
class Solution:
    """
    @param nums: an array
    @param k: an integer
    @return: the number of subarrays where the product of all the elements in the subarray is less than k
    """
    def numSubarrayProductLessThanK(self, nums, k):
        if not nums or len(nums) == 0 or k <= 1:
            return 0

        count, product = 0, 1
        left = 0

        for right in range(len(nums)):
            product *= nums[right]
            while product >= k:
                product //= nums[left]
                left += 1
            count += right - left + 1

        return count


# 41 · Maximum Subarray (compare with 149)
"""
Given an array of integers, find a contiguous subarray which has the largest sum.

Input:  nums = [-2,2,-3,4,-1,2,1,-5,3]
Output: 6
"""

# 使用前缀和的方法，计算每个位置为结尾的 subarray 的最大值是多少。
class Solution:
    """
    @param nums: A list of integers
    @return: A integer indicate the sum of max subarray
    """
    def maxSubArray(self, nums):
        if not nums or len(nums) == 0:
            return 0

        # prefix_sum记录前i个数的和，max_Sum记录全局最大值，min_Sum记录前i个数中0-k的最小值
        min_sum, max_sum = 0, -sys.maxsize
        prefix_sum = 0
        
        for num in nums:
            prefix_sum += num
            max_sum = max(max_sum, prefix_sum - min_sum)
            min_sum = min(min_sum, prefix_sum)
            
        return max_sum

    # Method.2      DP
    def maxSubArray(self, nums):
        if not nums or len(nums) == 0:
            return -1

        dp = [0] * len(nums)
        dp[0] = nums[0]

        for i in range(1, len(nums)):
            dp[i] = nums[i] + max(0, dp[i-1])

        return max(dp)




# 44 · Minimum Subarray
"""
Given an array of integers, find the continuous subarray with smallest sum.
Return the sum of the subarray.

Input:  array = [1,-1,-2,1,-4]
Output: -6
"""
import sys

class Solution:
    """
    @param: nums: a list of integers
    @return: A integer indicate the sum of minimum subarray
    """
    def minSubArray(self, nums):
        # prefix sum: p[k] = nums[0] + nums[1] + ...+ nums[k]
        # prefix min subarray sum from k to 0: p[k] - max(p[k-1],p[k-2,...,p[1], p[0]]) for k = 1 ~ n-1
        # p[0] = nums[0] - 0
        
        if not nums or len(nums) == 0:
            return 0

        min_sum, max_sum = sys.maxsize, 0
        prefix_sum = 0
        
        for num in nums:
            prefix_sum += num
            min_sum = min(min_sum, prefix_sum - max_sum)
            max_sum = max(max_sum, prefix_sum)
            
        return min_sum





# 42 · Maximum Subarray II
"""
Given an array of integers, find two non-overlapping subarrays which have the largest sum.
The number in each subarray should be contiguous. Return the largest sum.

Input:  nums = [1, 3, -1, 2, -1, 2]
Output: 7
"""
# 这个题的思路是，因为 两个subarray 一定不重叠,所以必定存在一条分割线,分开这两个 subarrays
# 所以 最后的部分里：max(max, left[i] + right[i + 1]) 这里是在枚举 这条分割线的位置
# 然后 left[] 和 right[] 里分别存的是，某个位置往左的 maximum subarray 和往右的 maximum subarray

class Solution:
    """
    @param: nums: A list of integers
    @return: An integer denotes the sum of max two non-overlapping subarrays
    """
    def maxTwoSubArrays(self, nums):
        if not nums or len(nums) < 2:
            return -1
        
        n = len(nums)
        left_maxsum, right_maxsum = [0] * n, [0] * n

        # 构建从left开始的 maxSum
        prefix_sum = 0
        max_sum, min_sum = -sys.maxsize - 1, 0
        for i in range(n):
            prefix_sum += nums[i]
            max_sum = max(max_sum, prefix_sum - min_sum)
            min_sum = min(min_sum, prefix_sum)
            
            left_maxsum[i] = max_sum

        # 构建从right开始的maxSum
        prefix_sum = 0
        max_sum, min_sum = -sys.maxsize - 1, 0    
        for i in range(n - 1, -1, -1):
            prefix_sum += nums[i]
            max_sum = max(max_sum, prefix_sum - min_sum)
            min_sum = min(min_sum, prefix_sum)
            
            right_maxsum[i] = max_sum
        
        # 隔板法
        max_sum = -sys.maxsize - 1
        for i in range(n - 1):
            max_sum = max(max_sum, left_maxsum[i] + right_maxsum[i + 1])
        
        return max_sum




# 43 · Maximum Subarray III
"""
Given an array of integers and a number k, find k non-overlapping subarrays which have the largest sum.
The number in each subarray should be contiguous.Return the largest sum.

Input:  nums =  [-1,4,-2,3,-2,3]   k = 2
Output: 8
"""

# 复杂度分析
# 假设数组大小为n，划分为k个不重叠的子数组。因为用到了两个数组记录局部最优解和全局最优解，如果记录当前位置使用二维数组，
# 则空间复杂度为O(nk)；若不记录当前位置使用一维数组，则空间复杂度为O(k)。
# 递推过程中依次推出到位置i分为j个部分的最优解，时间复杂度为O(nk)。


# Method. 1     二维
# 用两个二维数组 local 和 global 分别来记录局部最优解和全局最优解，局部最优解就是必须选取当前元素的最优解，
# 全局最优解就是不一定选取当前元素的最优解。local[i][j]表示整数数组nums的前i个元素被分成j个不重叠子数组时的最大值
# （必须选取元素nums[i]）。global[i][j]表示整数数组nums的前i个元素被分成j个不重叠子数组时的最大值（不一定选取元素nums[i]）。
# 易得当i<j时不可能存在可行解，且边界值为i==j时，每个元素各自为一组，答案就是nums的前i项之和。
# 所以我们从边界值往回递推（j从k递推回1），状态转移方程（此时i >= j）
class Solution:
    """
    @param nums: A list of integers
    @param k: An integer denote to find k non-overlapping subarrays
    @return: An integer denote the sum of max k non-overlapping subarrays
    """
    def maxSubArray(self, nums, k):
        n = len(nums)
        if n < k:
            return 0
        local = [[0] * (k + 3) for _ in range(n + 3)]
        globa = [[0] * (k + 3) for _ in range(n + 3)]
        for i in range(1, n + 1):
            # 从边界值往前递推
            for j in range(min(i, k), 0, -1):
                if i == j:
                    local[i][j] = local[i - 1][j - 1] + nums[i - 1]
                    globa[i][j] = globa[i - 1][j - 1] + nums[i - 1]
                else:
                    # local[i-1][j]表示nums[i]加入上一个子数组成为一部分
                    # global[i-1][j-1]表示nums[i]重新开始一个新的子数组
                    local[i][j] = max(local[i - 1][j], globa[i - 1][j - 1]) + nums[i - 1]
                    globa[i][j] = max(globa[i - 1][j], local[i][j])
        
        return globa[n][k]


# Method.2     一维（优化）
class Solution:
    """
    @param nums: A list of integers
    @param k: An integer denote to find k non-overlapping subarrays
    @return: An integer denote the sum of max k non-overlapping subarrays
    """
    def maxSubArray(self, nums, k):
        n = len(nums)
        if n < k:
            return 0
        local, globa = [0] * (k + 5), [0] * (k + 5)
        for i in range(1, n + 1):
            # 从边界值往前递推
            for j in range(min(i, k), 0, -1):
                if i == j:
                    local[j] = local[j - 1] + nums[i - 1]
                    globa[j] = globa[j - 1] + nums[i - 1]
                else:
                    # local[i-1][j]表示nums[i]加入上一个子数组成为一部分
                    # global[i-1][j-1]表示nums[i]重新开始一个新的子数组
                    local[j] = max(local[j], globa[j - 1]) + nums[i - 1]
                    globa[j] = max(globa[j], local[j])

        return globa[k]




# 45 · Maximum Subarray Difference
"""
Given an array with integers. Find two non-overlapping subarrays A and B, which 
|SUM(A) - SUM(B)| is the largest. Return the largest difference.

Input:  array = [1, 2, -3, 1]
Output: 6
Explanation:    The subarray are [1,2] and [-3].So the answer is 6.
"""
class Solution:
    """
    @param nums: A list of integers
    @return: An integer indicate the value of maximum difference between two substrings
    """
    def maxDiffSubArrays(self, nums):
        if not nums or len(nums) == 0:
            return -1
            
        n = len(nums)
        left_max, right_max = [0] * n, [0] * n
        left_min, right_min = [0] * n, [0] * n
        
        # get left max_sum
        prefix_sum = 0
        min_sum, max_sum = 0, -sys.maxsize - 1
        for i in range(n):
            prefix_sum += nums[i]
            max_sum = max(max_sum, prefix_sum - min_sum)
            min_sum = min(min_sum, prefix_sum)
            
            left_max[i] = max_sum
        
        # get right max_sum
        prefix_sum = 0
        min_sum, max_sum = 0, -sys.maxsize - 1
        for i in range(n - 1, -1, -1):
            prefix_sum += nums[i]
            max_sum = max(max_sum, prefix_sum - min_sum)
            min_sum = min(min_sum, prefix_sum)
            
            right_max[i] = max_sum
        
        # get left min_sum
        prefix_sum = 0
        min_sum, max_sum = sys.maxsize, 0
        for i in range(n):
            prefix_sum += nums[i]
            min_sum = min(min_sum, prefix_sum - max_sum)
            max_sum = max(max_sum, prefix_sum)
            
            left_min[i] = min_sum
        
        # get right min_sum
        prefix_sum = 0
        min_sum, max_sum = sys.maxsize, 0
        for i in range(n - 1, -1, -1):
            prefix_sum += nums[i]
            min_sum = min(min_sum, prefix_sum - max_sum)
            max_sum = max(max_sum, prefix_sum)
            
            right_min[i] = min_sum
        
        res = 0
        for i in range(n - 1):
            res = max(res, abs(left_max[i] - right_min[i + 1]), abs(left_min[i] - right_max[i + 1]))
        
        return res





# 620 · Maximum Subarray IV
"""
Given an integer arrays, find a contiguous subarray which has the largest sum and length should be 
greater or equal to given length k. Return the largest sum, return 0 if there are fewer than k 
elements in the array.

Input:  [-2,2,-3,4,-1,2,1,-5,3]     5
Output: 5
Explanation:    [2,-3,4,-1,2,1]     sum=5
"""

# 一维dp即可。维护一个前缀和。再维护一个前缀合法最小和，转移方程如下：
# result = Math.max(result, sum[i] - min_prefix)
# min_prefix = Math.min(min_prefix, sum[i - k + 1])

class Solution:
    """
    @param nums: an array of integer
    @param k: an integer
    @return: the largest sum
    """
    def maxSubarray4(self, nums, k):
        if not nums or len(nums) == 0 or len(nums) < k:
            return 0

        #rightSum表示当前指向位置的前缀和
        #leftSum表示当前指向位置左侧k个位置的前缀和
        #minLeftSum表示左侧最小的前缀和
        leftSum, rightSum = 0, 0
        minLeftSum = 0

        for i in range(k):
            rightSum += nums[i]

        result = rightSum
        for i in range(k, len(nums)):
            #右端点右移，更新数据
            rightSum += nums[i]
            leftSum += nums[i - k]
            minLeftSum = min(minLeftSum, leftSum)
            result = max(result, rightSum - minLeftSum)

        return result




# 621 · Maximum Subarray V
"""
Given an integer arrays, find a contiguous subarray which has the largest sum and length 
should be between k1 and k2 (include k1 and k2).
Return the largest sum, return 0 if there are fewer than k1 elements in the array.

Input:  [-2,2,-3,4,-1,2,1,-5,3]     2   4
Output: 6
Explanatipn:
 the contiguous subarray `[4,-1,2,1]` has the largest sum = `6`.
"""
from collections import deque

# 滑动窗口系列题，deque维护子数组，每一次窗口滑动则：前缀和 - 队头，最后得出最大值。
class Solution:
    """
    @param nums: an array of integers
    @param k1: An integer
    @param k2: An integer
    @return: the largest sum
    """
    def maxSubarray5(self, nums, k1, k2):
        n = len(nums)
        if n < k1:
            return 0

        result = -sys.maxsize
        queue = deque()
        prefix_sum = [0 for _ in range(n + 1)]

        for i in range(1, n + 1):
            prefix_sum[i] = prefix_sum[i - 1] + nums[i - 1]
            if len(queue) and queue[0] < i - k2:
                queue.popleft()

            # 求maxsubarry只需要現在的prefix減去最小值，因此可以用一個while loop, 把所有比要加上去的數值
            # (prefix[i-k1])小的數都pop掉，最後如果deque裡有值, 就比較目前的最大值及prefix[i] - deque[0]得到答案
            if i >= k1:
                while len(queue) and prefix_sum[queue[-1]] > prefix_sum[i - k1]:
                    queue.pop()     # 这里注意不是popleft
                queue.append(i - k1)

            # [i - k2, i - k1]
            if len(queue) and prefix_sum[i] - prefix_sum[queue[0]] > result:
                result = prefix_sum[i] - prefix_sum[queue[0]]
    
        return result




# 722 · Maximum Subarray VI     (Skipped)
"""
Given an array of integers. find the maximum XOR subarray value in given array.

Input: [8, 1, 2, 12, 7, 6]
Output: 15
Explanation:
The subarray [1, 2, 12] has maximum XOR value
"""





# 1850 · Pick Apples
"""
There are N apple trees in the orchard. Alice is planning to collect all the apples from K 
consecutive trees and Bob is planning to collect all the apples from L consecutive trees.
They want to choose to disjoint segements so not to disturb each other. you should return 
the maximum number of apples that they can collect.

input:  A = [6, 1, 4, 6, 3, 2, 7, 4]    K = 3   L = 2
Output:     24
Explanation: 
beacuse Alice can choose tree 3 to 5 and collect 4 + 6 + 3 = 13 apples, and Bob can choose 
trees 7 to 8 and collect 7 + 4 = 11 apples.Thus, they will collect 13 + 11 = 24.
"""
from typing import (
    List,
)

# 区间和首先想到前缀和快速求解, 因为互相不干涉，所以考虑隔板，分割成左右两部分
# 因为L和K的区间连续，所以右移枚举每一个分割点，分成 [i-L:i]和[i:i+L] 两部分。
# max(max(左窗口sum) + 当前右窗口sum) 即为结果
# K和L的长度不同，可以分成的左右两个部分的区间也会不同, 所以需要考虑左边K，右边L和左边L，右边K。
# 所以基本同样的code，copy过来K和L对掉。
# 时间：O(n)
# 空间：O(n)
class Solution:
    """
    @param a: a list of integer
    @param k: a integer
    @param l: a integer
    @return: return the maximum number of apples that they can collect.
    """
    def pick_apples(self, A: List[int], K: int, L: int) -> int:
        prefix_sum = [0]
        for n in A:
            prefix_sum.append(prefix_sum[-1] + n)

        result = -1

        max_left = 0
        for i in range(K, len(prefix_sum) - L):
            max_left = max(prefix_sum[i] - prefix_sum[i-K], max_left)
            result = max(max_left + prefix_sum[i+L] - prefix_sum[i], result)

        max_left = 0
        for i in range(L, len(prefix_sum) - K):
            max_left = max(prefix_sum[i] - prefix_sum[i-L], max_left)
            result = max(max_left + prefix_sum[i+K] - prefix_sum[i], result)

        return result