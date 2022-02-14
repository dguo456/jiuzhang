# 1833 · pen box
"""
Given you an array boxes and a target. Boxes[i] means that there are boxes[i] pens in the ith box. 
Subarray [i, j] is valid if sum(boxes[i] + boxes[i+1] + ... + boxes[j]) == target. Please find two 
not overlapped valid subarrays and let the total length of the two subarrays minimum. 
Return the minimum length. If you can not find such two subarrays, return -1.

Input:
boxes = [1,2,2,1,1,1],
target = 3
Output: 4
"""
import sys

class Solution:
    """
    @param boxes: number of pens for each box
    @param target: the target number
    @return: the minimum boxes
    """
    # APP1:
    # find one min valid subsarrays: sliding window
    # find two: iterate through boxes, find left and find right
    # Time: O(n^2), Space: O(1)
    # Result: TLE 
    def minimumBoxes(self, boxes, target):
        if not boxes or len(boxes) < 2 or not target or target < 0:
            return -1

        result = sys.maxsize

        for i in range(1, len(boxes)):
            min_left = self.find_min_len(boxes, 0, i, target)
            min_right = self.find_min_len(boxes, i, len(boxes), target)
            if min_left != sys.maxsize and min_right != sys.maxsize:
                result = min(min_left + min_right, result)

        if result == sys.maxsize:
            return -1
        return result
        
    def find_min_len(self, boxes, start, end, target):
        min_length = sys.maxsize 
        left = start
        temp_sum = 0

        for right in range(start, end):
            temp_sum += boxes[right]
            while temp_sum > target:
                temp_sum -= boxes[left]
                left += 1
            if temp_sum == target:
                min_length = min(min_length, right - left + 1)
                
        return min_length



# Method.2      优化，用空间换时间
# Optimize APP1 by pre-processing left_min, right_min array 
# Time: O(n), Space: O(n)
class Solution:
    """
    @param boxes: number of pens for each box
    @param target: the target number
    @return: the minimum boxes
    """
    def minimumBoxes(self, boxes, target):
        if not boxes or len(boxes) < 2 or not target or target < 0:
            return -1 

        left_min = self.get_min_len(boxes, target)
        new_boxes = boxes[::-1]

        right_min = self.get_min_len(new_boxes, target)
        right_min = right_min[::-1]

        result = sys.maxsize
        for i in range(len(boxes) - 1):
            if left_min[i] == sys.maxsize or right_min[i + 1] == sys.maxsize:
                continue
            result = min(left_min[i] + right_min[i + 1], result)

        if result == sys.maxsize:
            return -1
        return result 
    
    def get_min_len(self, boxes, target):
        result = []
        left = temp_sum = 0
        min_length = sys.maxsize

        for right in range(len(boxes)):
            temp_sum += boxes[right]
            while temp_sum > target:
                temp_sum -= boxes[left]
                left += 1 
            if temp_sum == target:
                min_length = min(right - left + 1, min_length)
            result.append(min_length)

        return result