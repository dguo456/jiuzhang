################################# Combination / Permutation / Subset ##############################
# 39. Combination Sum
# 40. Combination Sum II
# 77. Combinations
# 46. Permutations
# 47. Permutations II
# 78. Subsets
# 90. Subsets II
################################## Merge Sort / Quick Sort / Quick Select #################################
# Merge Sort / Quick Sort / Quick Select Template
# 21. Merge Two Sorted Lists
# 23. Merge k Sorted Lists
# 88. Merge Sorted Array
# 75. Sort Colors
# 148. Sort List
# 280. Wiggle Sort

############################################################################################################
import heapq
import random
from typing import List, Optional, Callable
from collections import Counter, deque, defaultdict, OrderedDict
from itertools import accumulate, combinations, permutations


# 39. Combination Sum
"""
Given an array of distinct integers candidates and a target integer target, return all unique combinations of candidates where the chosen numbers sum to target. 
You may return the combinations in any order. The same number may be chosen from candidates an unlimited number of times.
Input: candidates = [2,3,6,7], target = 7
Output: [[2,2,3],[7]]
"""
class Solution:
    
    def combinationSum(self, candidates: List[int], target: int) -> List[List[int]]:
        combinations = []
        self.dfs(candidates, target, 0, [], combinations)
        return combinations
    
    def dfs(self, candidates, target, start_index, combination, combinations):
        if target < 0:
            return
        
        if target == 0:
            combinations.append(combination[:])
            return
        
        for i in range(start_index, len(candidates)):
            combination.append(candidates[i])
            self.dfs(candidates, target - candidates[i], i, combination, combinations)
            combination.pop()





# 40. Combination Sum II
"""
Each number in candidates may only be used once in the combination. The solution set must not contain duplicate combinations.
Input: candidates = [2,5,2,1,2], target = 5
Output: [[1,2,2], [5]]
"""
class Solution:
    
    def combinationSum2(self, candidates: List[int], target: int) -> List[List[int]]:
        combinations = []
        self.dfs(sorted(candidates), target, 0, [], combinations)       # 需要有序
        return combinations
    
    def dfs(self, candidates, target, start_index, combination, combinations):
        if target < 0:
            return
        
        if target == 0:
            combinations.append(combination[:])
            return
        
        for i in range(start_index, len(candidates)):
            if i > start_index and candidates[i] == candidates[i-1]:
                continue
                
            combination.append(candidates[i])
            self.dfs(candidates, target - candidates[i], i + 1, combination, combinations)      # i + 1
            combination.pop()





# 77. Combinations
"""
Input: n = 4, k = 2
Output: [[2,4], [3,4], [2,3], [1,2], [1,3], [1,4]]
"""
class Solution:
    
    def combine(self, n: int, k: int) -> List[List[int]]:
        combinations = []
        self.dfs(n, k, 1, [], combinations)
        return combinations
    
    def dfs(self, n, k, start_index, combination, combinations):
        if len(combination) == k:
            combinations.append(combination[:])
            return
        
        for i in range(start_index, n + 1):
            combination.append(i)
            self.dfs(n, k, i + 1, combination, combinations)
            combination.pop()





# 46. Permutations
"""
Input: nums = [1,2,3]
Output: [[1,2,3],[1,3,2],[2,1,3],[2,3,1],[3,1,2],[3,2,1]]
"""
class Solution:
    def permute(self, nums: List[int]) -> List[List[int]]:
        if not nums or len(nums) == 0:
            return []
        
        permutations = []
        visited = set()
        self.dfs(nums, visited, [], permutations)
        return permutations
    
    def dfs(self, nums, visited, permutation, permutations):
        if len(permutation) == len(nums):
            permutations.append(permutation[:])
            return
        for i in range(len(nums)):
            if i in visited:
                continue
            
            # permutation.append(nums[i])
            visited.add(i)
            self.dfs(nums, visited, permutation + [nums[i]], permutations)
            visited.remove(i)
            # permutation.pop()





# 47. Permutations II
"""
Input: nums = [1,1,2]
Output: [[1,1,2], [1,2,1], [2,1,1]]
"""
class Solution:
    def permuteUnique(self, nums: List[int]) -> List[List[int]]:
        if not nums or len(nums) == 0:
            return []
        
        permutations = []
        visited = set()
        self.dfs(sorted(nums), visited, [], permutations)
        return permutations
    
    def dfs(self, nums, visited, permutation, permutations):
        if len(permutation) == len(nums):
            permutations.append(permutation[:])
            return
        
        for i in range(len(nums)):
            if i in visited:
                continue
            if i > 0 and nums[i-1] == nums[i] and (i-1) in visited:
                continue
            
            # permutation.append(nums[i])
            visited.add(i)
            self.dfs(nums, visited, permutation + [nums[i]], permutations)
            visited.remove(i)
            # permutation.pop()





# 78. Subsets
"""
Input: nums = [1,2,3]
Output: [[],[1],[2],[1,2],[3],[1,3],[2,3],[1,2,3]]
"""
class Solution:
    def subsets(self, nums: List[int]) -> List[List[int]]:
        if not nums or len(nums) == 0:
            return []
        self.results = []
        self.search(sorted(nums), [], 0)
        return self.results
    
    # Approach 1
    def search(self, nums, S, index):
        
        self.results.append(S[:])
        
        for i in range(index, len(nums)):
            self.search(nums, S + [nums[i]], i + 1)
            
    # Approach 2
    def search(self, nums, S, index):
        if index == len(nums):
            self.results.append(S[:])
            return
        
        self.search(nums, S + [nums[index]], index + 1)
        self.search(nums, S, index + 1)
        
    # Approach 3
    def subsets(self, nums: List[int]) -> List[List[int]]:
        n = len(nums)
        output = [[]]
        
        for num in nums:
            for j in range(len(output)):
                output.append(output[j] + [num])
        # for num in nums:
        #     output += [curr + [num] for curr in output]
        
        return output





# 90. Subsets II
"""
Input: nums = [1,2,2]
Output: [[],[1],[1,2],[1,2,2],[2],[2,2]]
"""
class Solution:
    def subsetsWithDup(self, nums: List[int]) -> List[List[int]]:
        self.results = []
        self.dfs(sorted(nums), 0, [])
        return self.results
    
    def dfs(self, nums, start_index, subset):
        self.results.append(subset[:])
        
        for i in range(start_index, len(nums)):
            if i > start_index and nums[i] == nums[i - 1]:
                continue
            subset.append(nums[i])
            self.dfs(nums, i + 1, subset)
            subset.pop()





################################## Merge Sort / Quick Sort / Quick Select #################################

# Standard Merge Sort Template      Time: O(nlogn)       Space: O(nlogn)      先局部有序，再整体有序
class Solution:
    def sortArray(self, nums: List[int]) -> List[int]:
        if not nums:
            return nums
        self.mergeSort(0, len(nums) - 1, nums)
        return nums
    
    def mergeSort(self, start, end, nums):
        if start >= end:
            return

        middle = (start + end) // 2
        self.mergeSort(start, middle, nums)
        self.mergeSort(middle + 1, end, nums)
        self.merge(start, end, nums)
    
    def merge(self, start, end, nums):
        tmp = [0] * (end - start + 1)
        middle = (start + end) // 2
        left, right = start, middle + 1
        index = 0
        
        while left <= middle and right <= end:
            if nums[left] <= nums[right]:
                tmp[index] = nums[left]
                left += 1
                index += 1
            else:
                tmp[index] = nums[right]
                right += 1
                index += 1
        
        while left <= middle:
            tmp[index] = nums[left]
            left += 1
            index += 1
        
        while right <= end:
            tmp[index] = nums[right]
            right += 1
            index += 1
        
        for i in range(end - start + 1):
            nums[i + start] = tmp[i]



# Standard Quick Sort Tempate   Time: on average O(nlogn), worst case O(n^2)     Space: O(logn), on average the recursion stack       先整体有序，再局部有序
class Solution:
    def sortArray(self, nums: List[int]) -> List[int]:
        if not nums:
            return nums
        self.quickSort(0, len(nums) - 1, nums)
        return nums
    
    def quickSort(self, start, end, nums):
        if start >= end:
            return  
        left, right = start, end
        pivot = nums[(left + right) // 2]
        
        while left <= right:
            while left <= right and nums[left] < pivot:
                left += 1
            while left <= right and nums[right] > pivot:
                right -= 1
            if left <= right:
                nums[left], nums[right] = nums[right], nums[left]
                left += 1
                right -= 1

        self.quickSort(start, right, nums)
        self.quickSort(left, end, nums)



# Standard Quick Select Tempate   Time: Worst O(n^2), Avg and Best O(n). Down from O(nlogn) of QuickSort        Space: O(log n)
class Solution:
    def findKthLargest(self, nums, k):
        return self.quickSelect(nums, 0, len(nums)-1 ,k-1)
        
    def quickSelect(self, nums, start, end, k):
        if start == end:
            return nums[start]
        
        pivot_point = self.give_pivot(start, end)
        pivot_index = self.partitian(nums, start, end, pivot_point)
        
        if pivot_index == k:
            return nums[k]
        elif k < pivot_index:
            return self.quickSelect(nums, start, pivot_index - 1, k)
        else: 
            return self.quickSelect(nums, pivot_index + 1, end, k)
    
    def partitian(self, nums, start, end, pivot):
        pivot_val = nums[pivot]
        nums[pivot], nums[end] = nums[end], nums[pivot]
        i = start - 1
        for j in range(start, end):
            if nums[j] > pivot_val:
                i += 1
                nums[j], nums[i] = nums[i], nums[j]
            
        nums[i+1], nums[end] = nums[end], nums[i+1]
        return i + 1
    
    def give_pivot(self, start, end):
        return random.randint(start, end)




# Definition for singly-linked list.
class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next

# 21. Merge Two Sorted Lists
"""
Input: list1 = [1,2,4], list2 = [1,3,4]
Output: [1,1,2,3,4,4]
"""
class Solution:
    def mergeTwoLists(self, list1: Optional[ListNode], list2: Optional[ListNode]) -> Optional[ListNode]:
        if not list1:
            return list2
        if not list2:
            return list1
        
        dummy = ListNode(-1)
        head = dummy
        while list1 and list2:
            if list1.val < list2.val:
                head.next = list1
                list1 = list1.next
            else:
                head.next = list2
                list2 = list2.next
            head = head.next
            
        head.next = list1 if list1 is not None else list2
        return dummy.next

    for _ in range(10): pass         # 为了好看





# 23. Merge k Sorted Lists
"""
Input: lists = [[1,4,5],[1,3,4],[2,6]]
Output: [1,1,2,3,4,4,5,6]
"""
class Solution:
    def mergeKLists(self, lists: List[Optional[ListNode]]) -> Optional[ListNode]:
        if not lists:
            return None
        
        order = 0
        dummy = head = ListNode(-1)
        heap = []
        
        for list_head in lists:
            if list_head:
                order += 1
                heapq.heappush(heap, (list_head.val, order, list_head))
                
        while heap:
            _, _, node = heapq.heappop(heap)
            head.next = node
            head = head.next
            
            if head.next:
                order += 1
                heapq.heappush(heap, (head.next.val, order, head.next))
                
        return dummy.next

    for _ in range(10): pass         # 为了好看





# 88. Merge Sorted Array
"""
Input: nums1 = [1,2,3,0,0,0], m = 3, nums2 = [2,5,6], n = 3
Output: [1,2,2,3,5,6]
"""
class Solution:     # Do not return anything, modify nums1 in-place instead.
    def merge(self, nums1: List[int], m: int, nums2: List[int], n: int) -> None:
        if not nums1 or len(nums1) == 0 or not nums2 or len(nums2) == 0:
            return nums1
        
        p1, p2, p = m - 1, n - 1, m + n - 1
        while p1 >= 0 and p2 >= 0:
            if nums1[p1] > nums2[p2]:
                nums1[p] = nums1[p1]
                p1 -= 1
            else:
                nums1[p] = nums2[p2]
                p2 -= 1
            p -= 1
            
        while p1 >= 0:
            nums1[p] = nums1[p1]
            p1 -= 1
            p -= 1
            
        while p2 >= 0:
            nums1[p] = nums2[p2]
            p2 -= 1
            p -= 1

    for _ in range(10): pass         # 为了好看





# 75. Sort Colors
"""
Given an array nums with n objects colored red, white, or blue, sort them in-place so that objects of the same color are adjacent, with the colors in the order red, white, and blue.
We will use the integers 0, 1, and 2 to represent the color red, white, and blue, respectively. You must solve this problem without using the library's sort function.
Input: nums = [2,0,2,1,1,0]
Output: [0,0,1,1,2,2]
"""
class Solution:     # 三指针
    def sortColors(self, nums: List[int]) -> None:
        left, curr, right = 0, 0, len(nums) - 1
        while curr <= right:
            if nums[curr] == 0:
                nums[left], nums[curr] = nums[curr], nums[left]
                left += 1
                curr += 1
            elif nums[curr] == 2:
                nums[curr], nums[right] = nums[right], nums[curr]
                right -= 1
            else:
                curr += 1

    for _ in range(10): pass         # 为了好看





# 148. Sort List
"""
Given the head of a linked list, return the list after sorting it in ascending order.
Input: head = [4,2,1,3]
Output: [1,2,3,4]
"""
class Solution:         # Merge Sort
    def sortList(self, head: Optional[ListNode]) -> Optional[ListNode]:
        if head is None:
            return None
        if head.next is None:
            return head
        
        slow = fast = head
        while fast.next and fast.next.next:
            slow = slow.next
            fast = fast.next.next
        
        mid = slow.next
        slow.next = None
        
        list1 = self.sortList(head)
        list2 = self.sortList(mid)
        results = self.merge(list1, list2)
        return results
    
    def merge(self, list1, list2):
        if not list1:
            return list2
        if not list2:
            return list1
        
        dummy = ListNode(-1)
        head = dummy
        while list1 and list2:
            if list1.val < list2.val:
                head.next = list1
                list1 = list1.next
            else:
                head.next = list2
                list2 = list2.next
            head = head.next
            
        head.next = list1 if list1 is not None else list2
        return dummy.next





# 280. Wiggle Sort
"""
Given an integer array nums, reorder it such that nums[0] <= nums[1] >= nums[2] <= nums[3]....
Input: nums = [3,5,2,1,6,4]
Output: [3,5,1,6,2,4]
Explanation: [1,6,2,5,3,4] is also accepted.
"""
class Solution:
    # Approach 1
    def wiggleSort(self, nums: List[int]) -> None:
        nums.sort()
        for i in range(1, len(nums) - 1, 2):
            nums[i], nums[i+1] = nums[i+1], nums[i]
    
    # Approach 2
    def wiggleSort(self, nums: List[int]) -> None:
        for i in range(len(nums) - 1):
            if (i % 2 == 0 and nums[i] > nums[i + 1]) or (i % 2 == 1 and nums[i] < nums[i + 1]):
                nums[i], nums[i + 1] = nums[i + 1], nums[i]