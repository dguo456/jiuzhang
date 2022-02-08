# 31 · Partition Array
"""
Given an array nums of integers and an int k, partition the array (i.e move the elements in "nums") 
such that:
All elements < k are moved to the left
All elements >= k are moved to the right
Return the partitioning index, i.e the first index i nums[i] >= k.
"""
class Solution:
    """
    @param nums: The integer array you should partition
    @param k: An integer
    @return: The index after partition
    """
    def partitionArray(self, nums, k):
        if nums is None or len(nums) == 0:
            return 0
            
        left, right = 0, len(nums) - 1
        # pivot = nums[(left + right) // 2]
        
        while left <= right:    # 注意是小于等于
            while left <= right and nums[left] < k:
                left += 1
            while left <= right and nums[right] >= k:
                right -= 1
                
            if left <= right:
                nums[left], nums[right] = nums[right], nums[left]
                left += 1
                right -= 1
                
        return left




# 625 · Partition Array II
"""
Partition an unsorted integer array into three parts:
The front part < low
The middle part >= low & <= high
The tail part > high
Return any of the possible solutions.
"""
class Solution:
    """
    @param nums: an integer array
    @param low: An integer
    @param high: An integer
    @return: nothing
    """
    def partition2(self, nums, low, high):
        if not nums or len(nums) == 0:
            return nums
            
        left, right = 0, len(nums) - 1
        index = len(nums) - 1           # 注意这里index初始值要跟right一样
        
        while index >= left:
            if nums[index] > high:
                nums[right], nums[index] = nums[index], nums[right]
                right -= 1
                index -= 1      # 注意这里只在挪动右边的时候index减一
            elif nums[index] < low:
                nums[left], nums[index] = nums[index], nums[left]
                left += 1
            else:
                index -= 1




# 373 · Partition Array by Odd and Even
class Solution:
    """
    @param: nums: an array of integers
    @return: nothing
    """
    def partitionArray(self, nums):
        if nums is None and len(nums) == 0:
            return None
            
        left, right = 0, len(nums) - 1
        
        while left <= right:
            while left <= right and nums[left] % 2 == 1:
                left += 1
            while left <= right and nums[right] % 2 == 0:
                right -= 1
                
            if left <= right:
                nums[left], nums[right] = nums[right], nums[left]
                left += 1
                right -= 1




# 96 · Partition List
"""
list = 1->4->3->2->5->2->null
x = 3
Output: 1->2->2->4->3->5->null
"""
# Definition of ListNode
class ListNode(object):
    def __init__(self, val, next=None):
        self.val = val
        self.next = next

class Solution:
    """
    @param head: The first node of linked list
    @param x: An integer
    @return: A ListNode
    """
    def partition(self, head, x):
        
        if head is None:
            return head
            
        # 新构建两个linked list ： a 和 b， 分别存放小于X的部分和大于X的部分
        head_a, head_b = ListNode(0), ListNode(0)
        tail_a, tail_b = head_a, head_b

        while head:
            if head.val < x:
                tail_a.next = head
                tail_a = tail_a.next
            else:
                tail_b.next = head
                tail_b = tail_b.next
            head = head.next
            
        tail_a.next = head_b.next
        tail_b.next = None          # 根据题意，tail最后要指向Null
        
        return head_a.next