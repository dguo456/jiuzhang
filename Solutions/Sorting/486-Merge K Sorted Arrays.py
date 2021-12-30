# Method.1  分治法，先自顶向下的分，再自底向上两两合并
class Solution:
    """
    @param arrays: k sorted integer arrays
    @return: a sorted array
    """
    def mergekSortedArrays(self, arrays):
        return self.divide_and_conquer(arrays, 0, len(arrays) - 1)

    def divide_and_conquer(self, arrays, start, end):
        if start == end:
            return arrays[start]

        mid = (start + end) // 2
        left = self.divide_and_conquer(arrays, start, mid)
        right = self.divide_and_conquer(arrays, mid+1, end)
        return self.merge_two_arrays(left, right)

    # 参考merge sort的merge步骤
    def merge_two_arrays(self, arr1, arr2):
        i, j = 0, 0
        results = []

        while i < len(arr1) and j < len(arr2):
            if arr1[i] < arr2[j]:
                results.append(arr1[i])
                i += 1
            else:
                results.append(arr2[j])
                j += 1

        while i < len(arr1):
            results.append(arr1[i])
            i += 1

        while j < len(arr2):
            results.append(arr2[j])
            j += 1

        return results


# Method.2  从左到右两两归并
class Solution:
    """
    @param arrays: k sorted integer arrays
    @return: a sorted array
    """
    def mergekSortedArrays(self, arrays):
        while len(arrays) > 1:
            next_arrays = []
            for i in range(0, len(arrays), 2):
                if i + 1 < len(arrays):
                    array = self.merge_two_arrays(arrays[i], arrays[i+1])
                else:
                    array = arrays[i]
                next_arrays.append(array)
            arrays = next_arrays

        return arrays[0]

    def merge_two_arrays(self, arr1, arr2):
        i, j = 0, 0
        results = []

        while i < len(arr1) and j < len(arr2):
            if arr1[i] < arr2[j]:
                results.append(arr1[i])
                i += 1
            else:
                results.append(arr2[j])
                j += 1

        while i < len(arr1):
            results.append(arr1[i])
            i += 1

        while j < len(arr2):
            results.append(arr2[j])
            j += 1

        return results


# Method.3  使用 Heapq 的方法最快，因为不需要创建额外空间。时间复杂度和其他的算法一致，都是 O(NlogK) N 是所有元素个数
import heapq

class Solution:
    """
    @param arrays: k sorted integer arrays
    @return: a sorted array
    """
    def mergekSortedArrays(self, arrays):
        result = []
        heap = []
        for index, array in enumerate(arrays):
            if len(array) == 0:
                continue
            heapq.heappush(heap, (array[0], index, 0))
             
        while len(heap):
            val, x, y = heap[0]
            heapq.heappop(heap)
            result.append(val)
            if y + 1 < len(arrays[x]):
                heapq.heappush(heap, (arrays[x][y + 1], x, y + 1))
            
        return result


# 104 Merge K Sorted Linked Lists (对比)
class ListNode(object):

    def __init__(self, val, next=None):
        self.val = val
        self.next = next

import heapq

class Solution:
    """
    @param lists: a list of ListNode
    @return: The head of one sorted list.
    """
    def mergeKLists(self, lists):
        sequence = 0
        
        if not lists:
            return None
        
        dummy = tail = ListNode(-1)
        heap = []
        for l in lists:
            if l:
                sequence += 1
                heapq.heappush(heap, (l.val, sequence, l))
                
        while heap:
            node = heapq.heappop(heap)[2]
            tail.next = node
            tail = tail.next

            if tail.next:
                sequence += 1
                heapq.heappush(heap, (tail.next.val, sequence, tail.next))
                
        return dummy.next
