# 135 · Combination Sum
"""
The same repeated number may be chosen from candidates unlimited number of times.

复杂度分析:

时间复杂度: O(n^{target/min}),拷贝过程视作O(1),n为集合中数字个数，min为集合中最小的数字
每个位置可以取集合中的任意数字，最多有target/min个数字。

空间复杂度: O(n^{target/min}),n为集合中数字个数，min为集合中最小的数字
对于用来保存答案的列表，最多有n^{target/min}种组合
"""

# Method.1      以target作为递归变量
class Solution:
    # @param candidates, a list of integers
    # @param target, integer
    # @return a list of lists of integers
    def combinationSum(self, candidates, target):
        # 用set去重,因为可以多次重复用同一个元素，所以每个元素只留一个
        candidates = sorted(list(set(candidates)))
        results = []
        self.dfs(candidates, target, 0, [], results)
        return results

    # 递归的定义：在candidates[start ... n-1] 中找到所有的组合，他们的和为 target
    # 和前半部分的 combination 拼起来放到 results 里
    # 找到所有以 combination 开头的满足条件的组合，放到 results
    def dfs(self, candidates, target, start_index, combination, results):
        # 递归的出口：target <= 0
        if target < 0:
            return
        
        if target == 0:
            # deepcooy
            results.append(list(combination))
            return
            
        # 递归的拆解：挑一个数放到 combination 里
        for i in range(start_index, len(candidates)):
            # [2] => [2,2]
            combination.append(candidates[i])
            self.dfs(candidates, target - candidates[i], i, combination, results)
            # [2,2] => [2]
            combination.pop()  # backtracking


    # Method.2      以path作为递归变量
    def dfs(self, candidates, target, start_index, combination, results):
        # if target < 0:
        if sum(combination) > target:
            return

        # if target == 0:
        if sum(combination) == target:
            results.append(combination[:])
            return
        
        for i in range(start_index, len(candidates)):
            # combination.append(candidates[i])
            # self.dfs(candidates, target-candidates[i], combination, i)
            self.dfs(candidates, target, combination + [candidates[i]], i)
            # combination.pop()



# 153 · Combination Sum II
"""
1.  Each number in num can only be used once in one combination.
2.  All numbers (including target) will be positive integers.
3.  Numbers in a combination a1, a2, … , ak must be in non-descending order. (ie, a1 ≤ a2 ≤ … ≤ ak)
4.  Different combinations can be in any order.
5.  The solution set must not contain duplicate combinations.
"""
class Solution:
    """
    @param num: Given the candidate numbers
    @param target: Given the target number
    @return: All the combinations that sum to target
    """
    def combinationSum2(self, num, target):
        num = sorted(num)
        results = []
        self.dfs(num, target, 0, [], results)
        return results

    def dfs(self, num, target, start_index, combination, results):
        if target < 0:
            return

        if target == 0:
            results.append(combination[:])
            return

        for i in range(start_index, len(num)):

            # 这里需要跳过相邻元素重复的情况-> [1, 2, 2, ...]
            if i > start_index and num[i] == num[i-1]:
                continue

            combination.append(num[i])
            # 同一个元素只能用一次，所以 index+1
            self.dfs(num, target - num[i], i + 1, combination, results)
            combination.pop()



# 1321 · Combination Sum III
"""
Find all possible combinations of k numbers that add up to a number n, 
given that only numbers from 1 to 9 can be used and each combination should be a unique set of numbers.

Example 1:
Input: k = 3, n = 7
Output: [[1,2,4]]

Example 2:
Input: k = 3, n = 9
Output: [[1,2,6], [1,3,5], [2,3,4]]
"""
class Solution:
    """
    @param k: an integer
    @param n: an integer
    @return: return a List[List[int]]
    """
    def combinationSum3(self, k, n):
        self.results = []
        self.dfs(k, n, [], 1)
        return self.results

    def dfs(self, k, target, path, start_index):
        if k == 0 and target == 0:
            self.results.append(path[:])
            return
            
        for i in range(start_index, 10):
            path.append(i)
            if target - i >= 0:
                self.dfs(k-1, target - i, path, i+1)
            path.pop()



# 564 · Combination Sum IV
"""Given an integer array nums with all positive numbers and no duplicates, 
find the number of possible combinations that add up to a positive integer target.

Input: nums = [1, 2, 4], and target = 4
Output: 6
Explanation:
The possible combination ways are:
[1, 1, 1, 1], [1, 1, 2], [1, 2, 1], [2, 1, 1], [2, 2], [4]
"""

# DFS, Time Limit Exceed easily
class Solution:
    """
    @param nums: an integer array and all positive numbers, no duplicates
    @param target: An integer
    @return: An integer
    """
    def backPackVI(self, nums, target):
        nums = sorted(nums)
        results = []
        self.dfs(nums, target, 0, [], results)
        print(results)
        return len(results)

    def dfs(self, nums, target, start_index, path, results):
        if target < 0:
            return

        if target == 0:
            results.append(path[:])
            return

        for i in range(start_index, len(nums)):
            path.append(nums[i])
            # 这里的start_index不传i，而是都传入start_index同一个值
            self.dfs(nums, target - nums[i], start_index, path, results)
            path.pop()

# DP, 参考backpack4
"""backPack VI 和combination sum IV是同一题
本质上是背包问题，但是和BP IV又不一样，因为同一组数可以组成不同的组合
比如[1, 1, 2]和[1, 2, 1]在本题是两个解，所以j循环要放在外面，使得相同的一组数
有可能出现在不同的结果中"""
class Solution:
    
    def backPackVI(self, nums, target):
        if not nums:
            return 0
        
        dp = [0] * (target + 1)
        dp[0] = 1
        
        for j in range(1, target + 1):
            for num in nums:
                if num > j:
                    continue
                dp[j] += dp[j - num]
        
        return dp[target]