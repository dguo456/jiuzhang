# 174 · Remove Nth Node From End of List
"""
Given a linked list, remove the nth node from the end of list and return its head.
For example, Given linked list: 1->2->3->4->5, and n = 2.
After removing the second node from the end, the linked list becomes 1->2->3->5.
先让快指针走n步，然后快慢一起走，走到头的时候慢指针指向的点删掉。
"""

# Definition of ListNode
class ListNode(object):
    def __init__(self, val, next=None):
        self.val = val
        self.next = next

class Solution:
    """
    @param head: The first node of linked list.
    @param n: An integer
    @return: The head of linked list.
    """
    def removeNthFromEnd(self, head, n):
        if not head or n < 0:
            return None

        dummy = ListNode(0)
        dummy.next = head
        slow, fast = dummy, head

        for _ in range(n):
            fast = fast.next

        while fast:
            fast = fast.next
            slow = slow.next

        slow.next = slow.next.next

        return dummy.next