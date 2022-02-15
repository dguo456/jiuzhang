# 56 · Two Sum
"""Given an array of integers, find two numbers such that they add up to a specific target number.
    The function twoSum should return indices of the two numbers such that they add up to the target, 
    where index1 must be less than index2. Assume each input would have exactly one solution"""

# Example:
# Input:
# numbers = [0,7,-11,0]
# target = 0
# Output: [0,3]


# Method.1      Hashmap
class Solution:
    """
    @param numbers: An array of Integer
    @param target: target = numbers[index1] + numbers[index2]
    @return: [index1, index2] (index1 < index2)
    """
    def twoSum(self, numbers, target):
        
        hash = {}
        
        for i in range(len(numbers)):
            if target - numbers[i] in hash:
                return [min(i, hash[target - numbers[i]]), max(i, hash[target - numbers[i]])]
            hash[numbers[i]] = i

        return [-1, -1]


# Method.2      使用双指针算法，时间复杂度 O(nlogn)，空间复杂度 O(n)
class Solution:
    """
    @param numbers: An array of Integer
    @param target: target = numbers[index1] + numbers[index2]
    @return: [index1, index2] (index1 < index2)
    """
    def twoSum(self, numbers, target):
        if not numbers:
            return [-1, -1]
        
        # transform numbers to a sorted array with index
        nums = [(number, index) for index, number in enumerate(numbers)]
        nums = sorted(nums)
        
        left, right = 0, len(nums) - 1
        while left < right:
            if nums[left][0] + nums[right][0] > target:
                right -= 1
            elif nums[left][0] + nums[right][0] < target:
                left += 1
            else:
                return sorted([nums[left][1], nums[right][1]])
        
        return [-1, -1]




# 443 · Two Sum - Greater than target
"""
Given an array of integers, find how many pairs in the array such that their sum is bigger than 
a specific target number. Please return the number of pairs.

Example 1:
Input: [2, 7, 11, 15], target = 24
Output: 1

Example 2:
Input: [1, 1, 1, 1], target = 1
Output: 6

Challenge
Do it in O(1) extra space and O(nlogn) time.
"""
class Solution:
    """
    @param nums: an array of integer
    @param target: An integer
    @return: an integer
    """
    def twoSum2(self, nums, target):
        if nums is None or len(nums) < 2:
            return 0
            
        nums.sort()
        left, right = 0, len(nums) - 1
        count = 0
        
        while left < right:
            if nums[left] + nums[right] > target:
                count += right - left
                right -= 1
                
            else:
                left += 1
                
        return count




# 609 · Two Sum - Less than or equal to target
"""
Given an array of integers, find how many pairs in the array such that their sum is less than or equal 
to a specific target number. Please return the number of pairs.

Example:
Input: nums = [2, 7, 11, 15], target = 24. 
Output: 5. 
"""
class Solution:
    """
    @param nums: an array of integer
    @param target: an integer
    @return: an integer
    """
    def twoSum5(self, nums, target):
        if nums is None or len(nums) < 2:
            return 0
            
        nums.sort()
        left, right = 0, len(nums) - 1
        count = 0
        
        while left < right:
            if nums[left] + nums[right] <= target:
                count += right - left
                left += 1
                
            else:
                right -= 1
                
        return count





# 533 · Two Sum - Closest to target
"""
Given an array nums of n integers, find two integers in nums such that the sum is 
closest to a given number, target. Return the absolute value of difference between 
the sum of the two numbers and the target.

Example
Input:  nums = [-1, 2, 1, -4] and target = 4
Output: 1
The minimum difference is 1. (4 - (2 + 1) = 1).
"""
import sys

class Solution:
    """
    @param nums: an integer array
    @param target: An integer
    @return: the difference between the sum and the target
    """
    def twoSumClosest(self, nums, target):
        if not nums or len(nums) < 2:
            return None

        nums.sort()
        left, right = 0, len(nums) - 1
        result = sys.maxsize

        while left < right:

            result = min(abs(target - (nums[left] + nums[right])), result)

            if nums[left] + nums[right] == target:
                return 0
            elif nums[left] + nums[right] < target:
                left += 1
            else:
                right -= 1

        return result



# 587 · Two Sum - Unique pairs
"""
Given an array of integers, find how many unique pairs in the array such that 
their sum is equal to a specific target number. Please return the number of pairs.

Input: nums = [1,1,2,45,46,46], target = 47 
Output: 2
"""
class Solution:
    """
    @param nums: an array of integer
    @param target: An integer
    @return: An integer
    """
    def twoSum6(self, nums, target):
        if not nums or len(nums) < 2:
            return 0

        nums.sort()
        
        left, right = 0, len(nums) - 1
        last_pair = [None, None]
        count = 0

        while left < right:
            if nums[left] + nums[right] == target:
                if [nums[left], nums[right]] != last_pair:
                    count += 1
                last_pair = [nums[left], nums[right]]
                # 这里左移右移之后一共就三种情况，要么一边是重复的，要么两边都重复，要么都不重复
                # 干脆就同时左移又移，可仔细揣摩
                left, right = left + 1, right - 1
            elif nums[left] + nums[right] < target:
                left += 1
            else:
                right -= 1

        return count




# 610 · Two Sum - Difference equals to target
"""
Given an sorted array of integers, find two numbers that their difference equals to a target value.
Return a list with two number like [num1, num2] that the difference of num1 and num2 equals to 
target value, and num1 is less than num2.

It's guaranteed there is only one available solution.
Note: Requires O(1) space complexity to comple

Input: nums = [2, 7, 15, 24], target = 5 
Output: [2, 7] 
"""
class Solution:
    """
    @param nums: an array of Integer
    @param target: an integer
    @return: [num1, num2] (num1 < num2)
    """
    def twoSum7(self, nums, target):
        if not nums or len(nums) < 2:
            return []

        n = len(nums)
        if target < 0:
            target = -target
        j = 0
        
        for i in range(n):
            if i == j:
                j += 1
            while j < n and nums[j] - nums[i] < target:
                j += 1
            if j < n and nums[j] - nums[i] == target:
                return [nums[i],nums[j]]




# 608 · Two Sum II - Input array is sorted
"""Please note that your returned answers (both index1 and index2) are not zero-based."""

# Method.1      双指针
class Solution:
    """
    @param nums: an array of Integer
    @param target: target = nums[index1] + nums[index2]
    @return: [index1 + 1, index2 + 1] (index1 < index2)
    """
    def twoSum(self, nums, target):
        if not nums or len(nums) < 2:
            return []

        left, right = 0, len(nums) - 1
        while left < right:
            if nums[left] + nums[right] == target:
                return [left+1, right+1]
            elif nums[left] + nums[right] < target:
                left += 1
            else:
                right -= 1

        return [-1, -1]


# Method.2      Hashmap
class Solution:
    """
    @param nums: an array of Integer
    @param target: target = nums[index1] + nums[index2]
    @return: [index1 + 1, index2 + 1] (index1 < index2)
    """
    def twoSum(self, nums, target):
        if not nums or len(nums) < 2:
            return []

        hashmap = {}
        for i in range(len(nums)):
            if target - nums[i] in hashmap:
                return [hashmap[target-nums[i]]+1, i+1]
            hashmap[nums[i]] = i




# 607 · Two Sum III - Data structure design
"""
Design and implement a TwoSum class. It should support the following operations: add and find.

add - Add the number to an internal data structure.
find - Find if there exists any pair of numbers which sum is equal to the value.

Example
add(1); add(3); add(5);
find(4) // return true
find(7) // return false
"""
class TwoSum:

    def __init__(self):
        self.values = []

    """
    @param number: An integer
    @return: nothing
    """
    def add(self, number):
        self.values.append(number)

    """
    @param value: An integer
    @return: Find if there exists any pair of numbers which sum is equal to the value.
    """
    def find(self, value):
        if len(self.values) < 2:
            return False

        self.values.sort()

        left, right = 0, len(self.values) - 1
        while left < right:
            if self.values[left] + self.values[right] == value:
                return True
            elif self.values[left] + self.values[right] < value:
                left += 1
            else:
                right -= 1

        return False




# 57 · 3Sum
"""Given an array S of n integers, are there elements a, b, c in S such that a + b + c = 0? 
Find all unique triplets in the array which gives the sum of zero."""

class Solution:
    """
    @param numbers: Give an array numbers of n integer
    @return: Find all unique triplets in the array which gives the sum of zero.
    """
    def threeSum(self, numbers):
        if numbers is None or len(numbers) < 3:
            return None
            
        numbers.sort()
        results = []
        
        for i in range(len(numbers) - 2):
            if i > 0 and numbers[i] == numbers[i-1]:
                continue
            # target = -numbers[i]
            left, right = i+1, len(numbers)-1
            
            while left < right:
                if numbers[i] + numbers[left] + numbers[right] == 0:
                    results.append([numbers[i], numbers[left], numbers[right]])
                    left += 1
                    right -= 1
                    while left < right and numbers[left] == numbers[left-1]:
                        left += 1
                    while left < right and numbers[right] == numbers[right+1]:
                        right -= 1
    
                elif numbers[i] + numbers[left] + numbers[right] < 0:
                    left += 1
                else:
                    right -= 1
            
        return results




# 918 · 3Sum Smaller
"""
Given an array of n integers nums and a target, find the number of index triplets i, j, k with 
0 <= i < j < k < n that satisfy the condition nums[i] + nums[j] + nums[k] < target.

Input:  nums = [-2,0,1,3], target = 2
Output: 2
Explanation:
Because there are two triplets which sums are less than 2:
[-2, 0, 1]
[-2, 0, 3]
"""
class Solution:
    """
    @param nums:  an array of n integers
    @param target: a target
    @return: the number of index triplets satisfy the condition nums[i] + nums[j] + nums[k] < target
    """
    def threeSumSmaller(self, nums, target):
        if not nums and len(nums) < 3:
            return 0

        nums.sort()
        result = 0

        for i in range(len(nums) - 2):
            left, right = i + 1, len(nums) - 1
            # t = target - nums[i]

            while left < right:
                if nums[i] + nums[left] + nums[right] < target:
                    result += right - left      # 这里注意要算上所有小于t的组合
                    left += 1
                else:
                    right -= 1

        return result




# 59 · 3Sum Closest
"""
Given an array S of n integers, find three integers in S such that the sum is closest to a given number, 
target. Return the sum of the three integers.
Input:
numbers = [2,7,11,15]       target = 3
Output:
20
Explanation:
2+7+11=20
"""
class Solution:
    """
    @param numbers: Give an array numbers of n integer
    @param target: An integer
    @return: return the sum of the three integers, the sum closest target.
    """
    def threeSumClosest(self, numbers, target):
        if not numbers or len(numbers) < 3:
            return
        
        numbers.sort()
        result = None
        
        for i in range(len(numbers) - 2):
            left, right = i + 1, len(numbers) - 1

            while left < right:
                temp_sum = numbers[left] + numbers[right] + numbers[i]
                if result == None or abs(temp_sum - target) < abs(result - target):
                    result = temp_sum
                    
                elif temp_sum <= target:
                    left += 1
                else:
                    right -= 1
                    
        return result




# 58 · 4Sum
"""Given an array S of n integers, are there elements a, b, c, and d in S such that a + b + c + d = target?"""
class Solution:
    """
    @param numbers: Give an array
    @param target: An integer
    @return: Find all unique quadruplets in the array which gives the sum of zero
    """
    def fourSum(self, numbers, target):
        if not numbers or len(numbers) < 4:
            return []

        numbers.sort()
        results = []

        for i in range(len(numbers) - 3):
            # 去重
            if i > 0 and numbers[i] == numbers[i - 1]:
                continue
            for j in range(i + 1, len(numbers) - 2):
                # 去重
                if j > i + 1 and numbers[j] == numbers[j - 1]:
                        continue
                left, right = j + 1, len(numbers) - 1
                two_sum = target - (numbers[i] + numbers[j])

                while left < right:
                    if numbers[left] + numbers[right] == two_sum:
                        results.append([numbers[i], numbers[j], numbers[left], numbers[right]])
                        left += 1
                        right -= 1

                        # 去重
                        while left < right and numbers[left] == numbers[left - 1]:
                            left += 1
                        while left < right and numbers[left] == numbers[right + 1]:
                            right -= 1

                    elif numbers[left] + numbers[right] < two_sum:
                        left += 1
                    else:
                        right -= 1

        return results
