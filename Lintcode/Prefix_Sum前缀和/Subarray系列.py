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