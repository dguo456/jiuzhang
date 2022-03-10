# 76 · Longest Increasing Subsequence
# 1093 · Number of Longest Increasing Subsequence
# 397 · Longest Continuous Increasing Subsequence
# 398 · Longest Continuous Increasing Subsequence II

#########################################################################################################



# 76 · Longest Increasing Subsequence
"""
Given a sequence of integers, find the longest increasing subsequence (LIS).
You code should return the length of the LIS.

Input:
nums = [4,2,4,5,3,7]
Output: 4
Explanation:    LIS is [2,4,5,7]
"""
class Solution:
    """
    @param nums: An integer array
    @return: The length of LIS (longest increasing subsequence)
    """
    # Method.1      标准DP      因为所求为子序列，很容易想到一种线性动态规划。
    # 对于求最长上升子序列，上升我们肯定需要知道当前阶段最后一个元素为多少，最长我们还要知道当前我们的序列有多长。
    # 那么我们就可以这样定义状态：设 dp[i] 表示以 nums[i] 为结尾的最长上升子序列的长度，为了保证元素单调递增肯定
    # 只能从 i 前面且末尾元素比 nums[i] 小的状态转移过来，每个位置初始值为 dp[i]=1（将自己作为一个序列）
    # 答案可以是任何阶段中只要长度最长的那一个, 空间复杂度：O(N)    时间复杂度：O(N^2)
    def longestIncreasingSubsequence(self, nums):
        if not nums or len(nums) == 0:
            return 0
        
        # dp[i]表示以nums[i]为结尾的最长上升子序列的长度，初始值为1
        dp = [1] * len(nums)

        for right in range(len(nums)):
            for left in range(right):
                # 若nums[left] < nums[right]那么可以接在该序列后，更新状态
                if nums[left] < nums[right]:
                    dp[right] = max(dp[right], dp[left] + 1)

        # 答案为所有状态中的最大值
        return max(dp)



    # Method.2      优化算法：动态规划(dp)+二分优化     设 dp[i] 表示长度为 i 的最长上升子序列的末尾元素的最小值，
    # 显然这个数组的权值一定单调不降。于是我们按顺序枚举数组nums，每一次对dp数组二分查找，
    # 找到小于nums[i]的最大的 dp[j]，并更新 dp[j+1]。
    def longestIncreasingSubsequence(self, nums):
        if not nums:
            return 0
        
        dp = [float('inf')] * (len(nums) + 1)
        dp[0] = -float('inf')
        
        longest = 0
        for num in nums:
            index = self.first_gte(dp, num)
            dp[index] = num
            longest = max(longest, index)
        
        return longest
        
    # find first index that the number greater than or equal to num
    def first_gte(self, nums, target):
        start, end = 0, len(nums) - 1
        while start + 1 < end:
            mid = (start + end) // 2
            if nums[mid] <= target:
                start = mid
            else:
                end = mid
                
        if nums[start] >= target:
            return start
        return end



    # Method.3      使用动态规划计算 Longest Increasing Subsequence，并同时打印具体的方案
    def longestIncreasingSubsequence(self, nums):
        if nums is None or not nums:
            return 0
        
        # state: dp[i] 表示从左到右跳到i的最长sequence 的长度
        
        # initialization: dp[0..n-1] = 1
        dp = [1] * len(nums)
        
        # prev[i] 代表 dp[i] 的最优值是从哪个 dp[j] 算过来的
        prev = [-1] * len(nums)
        
        # function dp[i] = max{dp[j] + 1},  j < i and nums[j] < nums[i]
        for i in range(len(nums)):
            for j in range(i):
                if nums[j] < nums[i] and dp[i] < dp[j] + 1:
                    dp[i] = dp[j] + 1
                    prev[i] = j
        
        # answer: max(dp[0..n-1])
        longest, last = 0, -1
        for i in range(len(nums)):
            if dp[i] > longest:
                longest = dp[i]
                last = i
        
        path = []
        while last != -1:
            path.append(nums[last])
            last = prev[last]
        print(path[::-1])
        
        return longest





# 1093 · Number of Longest Increasing Subsequence
"""
Given an unsorted array of integers, find the number of longest increasing subsequence.
Input: [2,2,2,2,2]
Output: 5
Explanation: there are 5 subsequences' length is 1, so output 5.
"""
class Solution:
    """
    @param nums: an array
    @return: the number of longest increasing subsequence
    """
    # 对每个元素，寻找其前面比它小的元素，比较LIS。length用来存LIS的长度，
    # total用来存length所对应的个数。时间复杂度O(N^2)
    def find_number_of_l_i_s(self, nums) -> int:
        n = len(nums)
        if n <= 1:
            return n

        length = [1 for _ in range(n)]
        total = [1 for _ in range(n)]

        for right in range(1, n):
            for left in range(right):
                if nums[left] < nums[right]:
                    if length[right] < length[left] + 1:
                        length[right] = length[left] + 1
                        total[right] = total[left]
                    elif length[right] == length[left] + 1:
                        total[right] += total[left]

        max_length = max(length)
        result = 0

        for i in range(n):
            if length[i] == max_length:
                result += total[i]

        return result





# 397 · Longest Continuous Increasing Subsequence
"""
Give an integer array, find the longest increasing continuous subsequence in this array.
An increasing continuous subsequence:
Can be from right to left or from left to right.
Indices of the integers in the subsequence should be continuous.

Input: [5, 4, 2, 1, 3]
Output: 4
Explanation:
For [5, 4, 2, 1, 3], the LICS  is [5, 4, 2, 1], return 4.
"""
class Solution:
    """
    @param A: An array of Integer
    @return: an integer
    """
    def longestIncreasingContinuousSubsequence(self, A):
        if not A:
            return 0
            
        longest, incr, desc = 1, 1, 1
        
        for i in range(1, len(A)):
            if A[i] > A[i-1]:
                incr += 1
                desc = 1
            elif A[i] < A[i-1]:
                desc += 1
                incr = 1
            else:
                incr = 1
                desc = 1
                
            longest = max(longest, max(incr, desc))
            
        return longest





# 398 · Longest Continuous Increasing Subsequence II
"""
Given an integer matrix. Find the longest increasing continuous subsequence in this matrix 
and return the length of it.
The longest increasing continuous subsequence here can start at any position and go up/down/left/right.
Input: 
    [
      [1, 2, 3, 4, 5],
      [16,17,24,23,6],
      [15,18,25,22,7],
      [14,19,20,21,8],
      [13,12,11,10,9]
    ]
Output: 25
Explanation: 1 -> 2 -> 3 -> 4 -> 5 -> ... -> 25 (Spiral from outside to inside.)
"""
DIRECTIONS = [(-1, 0), (1, 0), (0, -1), (0, 1)]
class Solution:
    """
    @param matrix: A 2D-array of integers
    @return: an integer
    """
    # Method.1      DFS + 记忆化搜索。memo 里记录了从 r,c 出发的最长上升序列的长度。
    def longest_continuous_increasing_subsequence2(self, matrix) -> int:
        if not matrix or not matrix[0]:
            return 0

        memo = {}
        result = 0

        for r in range(len(matrix)):
            for c in range(len(matrix[0])):
                result = max(result, self.dfs(matrix, memo, r, c))

        return result

    def dfs(self, matrix, memo, x, y):
        if (x, y) in memo:
            return memo[(x, y)]

        longest = 1
        for dx, dy in DIRECTIONS:
            next_x, next_y = x + dx, y + dy

            if not (0 <= next_x < len(matrix) and 0 <= next_y < len(matrix[0])):
                continue
            if matrix[next_x][next_y] <= matrix[x][y]:
                continue

            longest = max(longest, self.dfs(matrix, memo, next_x, next_y) + 1)

        memo[(x, y)] = longest
        return longest
        


    # Method.2      序列性动态规划      把二维矩阵打散成为一维数组，数组中每个元素记录二维矩阵中的坐标和高度。
    # 然后把一维数组按照高度排序。f[i] 表示第 i 个点结束的最长序列，得到公式：
    # f[i] = max{f[j] + 1 |  j < i && j 这个点比 i 要低，且i和j两个点相邻}
    # 类似的题还有 longest increasing subsequence.
    def longest_continuous_increasing_subsequence2(self, matrix) -> int:
        if not matrix or not matrix[0]:
            return 0

        m, n = len(matrix), len(matrix[0])
        points = []
        for r in range(m):
            for c in range(n):
                points.append((matrix[r][c], r, c))

        points.sort()

        longest_hash = {}
        for i in range(len(points)):
            key, val = (points[i][1], points[i][2]), points[i][0]
            longest_hash[key] = 1

            for dx, dy in DIRECTIONS:
                next_x, next_y = points[i][1] + dx, points[i][2] + dy
                next_key = (next_x, next_y)

                if not (0 <= next_x < m and 0 <= next_y < n):
                    continue
                if (next_x, next_y) in longest_hash and matrix[next_x][next_y] < val:
                    longest_hash[key] = max(longest_hash[key], longest_hash[next_key]+1)

        return max(longest_hash.values())
