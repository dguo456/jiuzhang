# 15 · Permutations

class Solution:
    """
    @param: nums: A list of integers.
    @return: A list of permutations.
    """
    # Method.1      标准DFS模板
    def permute(self, nums):
        if not nums:
            return [[]]
            
        permutations = []
        self.dfs(nums, [], set(), permutations)
        return permutations
        
    def dfs(self, nums, permutation, visited, permutations):
        if len(nums) == len(permutation):
            permutations.append(list(permutation))
            return
        
        for num in nums:
            if num in visited:
                continue
            permutation.append(num)
            visited.add(num)
            self.dfs(nums, permutation, visited, permutations)
            visited.remove(num)
            permutation.pop()

    # Method.2      BFS
    def permute(self, nums):
        
        if not nums:
            return [[]]
            
        stack = [[n] for n in nums]
        results = []
        
        while stack:
            last = stack.pop()
            if len(last) == len(nums):
                results.append(last)
                continue
            
            for i in range(len(nums)):
                if nums[i] not in last:
                    stack.append(last + [nums[i]])
                    
        return results



# 16 · Permutations II
"""Given a list of numbers with duplicate numbers in it. Find all unique permutations of it."""
class Solution:
    """
    @param: :  A list of integers
    @return: A list of unique permutations
    """

    def permuteUnique(self, nums):
        
        nums = sorted(nums)
        visited = {i: False for i in range(len(nums))}
        permutations = []
        self.dfs(nums, visited, [], permutations)
        return permutations
        
    def dfs(self, nums, visited, permutation, permutations):
        if len(permutation) == len(nums):
            permutations.append(permutation[:])
            return
        
        for i in range(len(nums)):
            if visited[i]:
                continue
            
            if i > 0 and nums[i] == nums[i-1] and visited[i-1]:
                continue
            
            visited[i] = True
            self.dfs(nums, visited, permutation + [nums[i]], permutations)
            visited[i] = False