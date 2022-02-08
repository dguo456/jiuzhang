# 102 · Linked List Cycle
"""
Given a linked list, determine if it has a cycle in it.
快慢指针的经典题。快指针每次走两步，慢指针一次走一步。
在慢指针进入环之后，快慢指针之间的距离每次缩小1，所以最终能相遇。
"""

# Definition of ListNode
class ListNode(object):
    def __init__(self, val, next=None):
        self.val = val
        self.next = next

class Solution:
    """
    @param head: The first node of linked list.
    @return: True if it has a cycle, or false
    """
    def hasCycle(self, head):
        if not head or not head.next:
            return False

        slow, fast = head, head
        while fast and fast.next:
            slow = slow.next
            fast = fast.next.next

            if slow == fast:
                return True

        return False




# 103 · Linked List Cycle II
"""
Given a linked list, return the node where the cycle begins.
考点:   双指针链表判环
题解:
使用双指针判断链表中是否有环，慢指针每次走一步，快指针每次走两步，若链表中有环，则两指针必定相遇。
假设环的长度为l，环上入口距离链表头距离为a，两指针第一次相遇处距离环入口为b，则另一段环的长度为c=l-b，
由于快指针走过的距离是慢指针的两倍，则有a+l+b=2*(a+b),又有l=b+c，可得a=c，故当判断有环时(slow==fast)时，
从头移动慢指针，同时移动快指针，两指针相遇处即为环的入口。
"""
class Solution:
    """
    @param head: The first node of linked list.
    @return: The node where the cycle begins. if there is no cycle, return null
    """
    def detectCycle(self, head):
        if head is None or head.next is None:
            return None

        fast = slow = head
        while fast and fast.next:
            fast = fast.next.next
            slow = slow.next
            if slow == fast:
                break
            
        if slow == fast:
            slow = head
            while slow != fast:
                slow = slow.next
                fast = fast.next
            return slow
            
        return None