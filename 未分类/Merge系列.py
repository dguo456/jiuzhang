"""
Merge系列的题基本属于必须熟练掌握的程度,又可以分为 merge 2/K sorted (array/linked-list),  and区间型的合并merge

Part.1
Merge 2/K sorted (array/linked-list)系列:
6  /486 - Merge Two/K Sorted Arrays
165/104 - Merge Two/K Sorted Lists

Part.2
Merge Interval系列:
156     - Merge Intervals  (920/919 - Meeting Rooms I&II)
295     - Intersection
839/577 - Merge Two Sorted Interval Lists I&II
"""

############################################################################################
###################################### PART ONE ############################################
############################################################################################


# 6 · Merge Two Sorted Arrays
"""
Merge two given sorted ascending integer array A and B into a new sorted integer array.
Input:
A = [1,2,3,4]
B = [2,4,5,6]
Output:
[1,2,2,3,4,4,5,6]
"""
class Solution:
    """
    @param A: sorted integer array A
    @param B: sorted integer array B
    @return: A new sorted integer array
    """
    def mergeSortedArray(self, A, B):
        
        index_A, index_B = 0, 0
        results = []

        while index_A < len(A) and index_B < len(B):
            if A[index_A] < B[index_B]:
                results.append(A[index_A])
                index_A += 1
            else:
                results.append(B[index_B])
                index_B += 1

        while index_A < len(A):
            results.append(A[index_A])
            index_A += 1

        while index_B < len(B):
            results.append(B[index_B])
            index_B += 1

        return results




# 486 · Merge K Sorted Arrays
"""
Given k sorted integer arrays, merge them into one sorted array.
Input: 
  [
    [1, 3, 5, 7],
    [2, 4, 6],
    [0, 8, 9, 10, 11]
  ]
Output: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
"""
# Method.1 使用 Heapq 的方法
# 最快，因为不需要创建额外空间。
# 时间复杂度和其他的算法一致，都是 O(NlogK),  N是所有元素个数
class Solution:
    """
    @param arrays: k sorted integer arrays
    @return: a sorted array
    """
    def mergekSortedArrays(self, arrays):
        results = []
        heap = []
        for index, array in enumerate(arrays):
            if not array:
                continue
            heapq.heappush(heap, (array[0], index, 0))

        while heap:
            val, index_arrays, index_val = heapq.heappop(heap)
            results.append(val)

            if index_val + 1 < len(arrays[index_arrays]):
                heapq.heappush(heap, (arrays[index_arrays][index_val+1], index_arrays, index_val+1))

        return results



# Method.2  分治法，先自顶向下的分，再自底向上两两合并
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


# Method.3  从左到右两两归并
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

    # Merge Two Sorted Arrays
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





# 165 · Merge Two Sorted Lists
"""
Merge two sorted (ascending) linked lists and return it as a new sorted list. The new sorted list 
should be made by splicing together the nodes of the two lists and sorted in ascending order.

Input:  list1 = 1->3->8->11->15->null, list2 = 2->null
Output: 1->2->3->8->11->15->null
"""

# Definition of ListNode
class ListNode(object):
    def __init__(self, val, next=None):
        self.val = val
        self.next = next

class Solution:
    """
    @param l1: ListNode l1 is the head of the linked list
    @param l2: ListNode l2 is the head of the linked list
    @return: ListNode head of linked list
    """
    def mergeTwoLists(self, l1, l2):
        if not l1:
            return l2
        if not l2:
            return l1

        dummy = ListNode(-1)
        temp = dummy

        while l1 and l2:
            if l1.val < l2.val:
                temp.next = l1
                l1 = l1.next
            else:
                temp.next = l2
                l2 = l2.next

            temp = temp.next

        while l1:
            temp.next = l1
            l1 = l1.next
            temp = temp.next

        while l2:
            temp.next = l2
            l2 = l2.next
            temp = temp.next

        return dummy.next




# 104 · Merge K Sorted Linked Lists
"""
Merge k sorted linked lists and return it as one sorted list.
Input:      lists = [2->4->null, null, -1->null]
Output:     2->5->6->7->null
"""
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

        for list_node in lists:
            if list_node:
                sequence += 1
                # 注意！heap里存的tuple顺序不能变！因为如果第一位相等，比较第二位，必须要是sequence，ListNode无法比大小！
                heapq.heappush(heap, (list_node.val, sequence, list_node))

        while heap:
            _, _, node = heapq.heappop(heap)
            tail.next = node
            tail = tail.next
            
            if tail.next:
                sequence += 1
                heapq.heappush(heap, (tail.next.val, sequence, tail.next))

        return dummy.next





############################################################################################
###################################### PART TWO ############################################
############################################################################################


# 156 · Merge Intervals
"""
Given a collection of intervals, merge all overlapping intervals.

Input:  [(1,3),(2,6),(8,10),(15,18)]
Output: [(1,6),(8,10),(15,18)]
"""

# Definition of Interval.
class Interval(object):
    def __init__(self, start, end):
        self.start = start
        self.end = end

class Solution:
    """
    @param intervals: interval list.
    @return: A new interval list.
    """
    def merge(self, intervals):
        intervals = sorted(intervals, key = lambda x: x.start)
        results = []
        
        for interval in intervals:
            if len(results) == 0 or results[-1].end < interval.start:
                results.append(interval)
            else:
                results[-1].end = max(interval.end, results[-1].end)
                
        return results




# 920 · Meeting Rooms
"""
Given an array of meeting time intervals consisting of start and end times [[s1,e1],[s2,e2],...] (si < ei), 
determine if a person could attend all meetings.
Input: intervals = [(0,30),(5,10),(15,20)]
Output: false
Explanation: 
(0,30), (5,10) and (0,30),(15,20) will conflict
"""

class Solution:
    """
    @param intervals: an array of meeting time intervals
    @return: if a person could attend all meetings
    """
    def canAttendMeetings(self, intervals):
        intervals = sorted(intervals, key = lambda x: x.start)
        max_end = -1

        for interval in intervals:
            if interval.start < max_end:
                return False
            max_end = max(interval.end, max_end)

        return True



# 919 · Meeting Rooms II
"""
Given an array of meeting time intervals consisting of start and end times [[s1,e1],[s2,e2],...] (si < ei), 
find the minimum number of conference rooms required.

Input: intervals = [(0,30),(5,10),(15,20)]
Output: 2
Explanation:
We need two meeting rooms
room1: (0,30)
room2: (5,10),(15,20)
"""

# 使用九章算法强化班中讲到的扫描线算法
class Solution:
    """
    @param intervals: an array of meeting time intervals
    @return: the minimum number of conference rooms required
    """
    def minMeetingRooms(self, intervals):
        if not intervals or len(intervals) == 0:
            return 0
            
        points = []
        for interval in intervals:
            points.append((interval.start, 1))
            points.append((interval.end, -1))
        # [(0, 1), (30, -1), (5, 1), (10, -1), (15, 1), (20, -1)]
            
        meeting_rooms = 0
        ongoing_meetings = 0
        # [(0, 1), (5, 1), (10, -1), (15, 1), (20, -1), (30, -1)]
        for _, delta in sorted(points):
            ongoing_meetings += delta
            meeting_rooms = max(meeting_rooms, ongoing_meetings)
            
        return meeting_rooms





# 295 · Intersection
"""
Given two sorted interval sequences, each interval in the sequence does not intersect each other, 
and returns the index of the interval where the two sequences intersect
input: a = [[0,3], [7,10]] , b = [[-1,1],[2,8]]
output: ans = [[0,0],[0,1],[1,1]]
"""

class Solution:
    """
    @param a: first sequence
    @param b: second sequence
    @return: return ans
    """
    def Intersection(self, a, b):
        index_a, index_b = 0, 0
        results = []

        while index_a < len(a) and index_b < len(b):
            if a[index_a][1] < b[index_b][0]:
                index_a += 1
            elif a[index_a][0] > b[index_b][1]:
                index_b += 1
            else:
                results.append([index_a, index_b])
                if a[index_a][1] < b[index_b][1]:
                    index_a += 1
                else:
                    index_b += 1

        return results




# 839 · Merge Two Sorted Interval Lists
"""
Merge two sorted (ascending) lists of interval and return it as a new sorted list. 
The new sorted list should be made by splicing together the intervals of the two lists and 
sorted in ascending order.

Input: list1 = [(1,2),(3,4)] and list2 = [(2,3),(5,6)]
Output: [(1,4),(5,6)]
"""

class Solution:
    """
    @param list1: one of the given list
    @param list2: another list
    @return: the new sorted list of interval
    """
    def mergeTwoInterval(self, list1, list2):
        i, j = 0, 0
        intervals = []
        while i < len(list1) and j < len(list2):
            if list1[i].start < list2[j].start:
                self.merge(intervals, list1[i])
                i += 1
            else:
                self.merge(intervals, list2[j])
                j += 1
        while i < len(list1):
            self.merge(intervals, list1[i])
            i += 1
        while j < len(list2):
            self.merge(intervals, list2[j])
            j += 1
        
        return intervals
        
    def merge(self, intervals, interval):
        if not intervals or intervals[-1].end < interval.start:
            intervals.append(interval)
            return
        
        intervals[-1].end = max(intervals[-1].end, interval.end)




# 577 · Merge K Sorted Interval Lists
"""
Merge K sorted interval lists into one sorted interval list. You need to merge overlapping intervals too.
Input: [
  [(1,3),(4,7),(6,8)],
  [(1,2),(9,10)]
]
Output: [(1,3),(4,8),(9,10)]
"""
# Method.1      先展开成一个大的区间列表，然后进行merge
class Solution:
    """
    @param intervals: the given k sorted interval lists
    @return:  the new sorted interval list
    """
    def mergeKSortedIntervalLists(self, intervals):
        if not intervals or len(intervals) == 0:
            return []

        flattened_intervals = []
        for interval_list in intervals:
            for interval in interval_list:
                flattened_intervals.append(interval)

        # 展平之后直接用 156 - Merge Intervals 去做
        flattened_intervals = sorted(flattened_intervals, key = lambda x: x.start)

        results = []
        for i in range(len(flattened_intervals)):
            if not results or results[-1].end < flattened_intervals[i].start:
                results.append(flattened_intervals[i])
            else:
                results[-1].end = max(flattened_intervals[i].end, results[-1].end)

        return results


# Method.2      使用 heap 来解决
import heapq

class Solution:
    """
    @param intervals: the given k sorted interval lists
    @return:  the new sorted interval list
    """
    def mergeKSortedIntervalLists(self, intervals):
        if not intervals or len(intervals) == 0:
            return []

        heap = []
        results = []
        for index, lists in enumerate(intervals):
            if not lists:
                continue
            heapq.heappush(heap, (lists[0].start, lists[0].end, index, 0))

        while heap:
            start, end, list_index, heap_index = heapq.heappop(heap)
            # 每次heappop出来的 interval(start, end)，merge到results里，注意merge的是区间，不能是Tuple
            self.merge(results, Interval(start, end))

            if heap_index + 1 < len(intervals[list_index]):
                next_interval = intervals[list_index][heap_index + 1]
                heapq.heappush(heap, (next_interval.start, next_interval.end, list_index, heap_index + 1))

        return results
        
    def merge(self, intervals, interval):
        if not intervals or intervals[-1].end < interval.start:
            intervals.append(interval)
            return
        
        intervals[-1].end = max(interval.end, intervals[-1].end)