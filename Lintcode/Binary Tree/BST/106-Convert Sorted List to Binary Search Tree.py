# 106 · Convert Sorted List to Binary Search Tree
"""
在链表上使用分治的方法。
这个算法的时间复杂度是 O(n) 的。
要诀在于，先把整个链表的长度取一次，然后拿着这个长度和链表头一起作为参数来进行递归构造。
convert(head, length) 表示把从 head 开头的长度为 length 的那么多个节点，转化为 bst。
return 两个值，一个是转化后的 bst 根节点，一个是链表上从 head 开始第 length + 1 个节点是谁。
***这样做的好处是，你不需要用 O(n) 的时间去找链表的中点了，直接 O(1) 从 return 里得到。***
"""
# Definition of ListNode
class ListNode(object):

    def __init__(self, val, next=None):
        self.val = val
        self.next = next

# Definition of TreeNode:
class TreeNode:
    def __init__(self, val):
        self.val = val
        self.left, self.right = None, None


class Solution:
    """
    @param: head: The first node of linked list.
    @return: a tree node
    """
    def sortedListToBST(self, head):
        length = self.get_linked_list_length(head)
        root, tail = self.convert(head, length)
        return root

    def get_linked_list_length(self, head):
        length = 0
        while head:
            length += 1
            head = head.next
        return length

    def convert(self, head, length):
        # 这里注意分治函数convert里分的是length，而不是root，所以谁一分为二谁作为判断条件
        if length == 0:
            return None, head   # 返回值个数要match

        # 以middle作为整个tree的根节点, 用分治构建左右子树
        left_root, middle = self.convert(head, length // 2)
        right_root, tail = self.convert(middle.next, length - length // 2 -1)

        root = TreeNode(middle.val)
        root.left = left_root
        root.right = right_root

        return root, tail   # (TreeNode, ListNode)



# Method.2      分治，不推荐，因为每次递归都需要用 O(n) 的时间去找链表的中点
class Solution:
    """
    @param: head: The first node of linked list.
    @return: a tree node
    """
    def sortedListToBST(self, head):
        if not head:
            return head

        root = self.dfs(head)
        return root

    def dfs(self, head):
        if head is None:
            return

        if head.next is None:
            return TreeNode(head.val)

        dummy = ListNode(0)
        dummy.next = head
        slow, fast = dummy, head

        while fast and fast.next:
            fast = fast.next.next
            slow = slow.next

        middle = slow.next
        slow.next = None   # ***如果不每次断开链表，递归深度会溢出***
        root = TreeNode(middle.val)

        root.left = self.dfs(head)
        root.right = self.dfs(middle.next)

        return root





# 177 · Convert Sorted Array to Binary Search Tree With Minimal Height

# Similar with above when build the tree
class Solution:
    """
    @param: A: an integer array
    @return: A tree node
    """
    def sortedArrayToBST(self, A):
        return self.convert(A, 0, len(A) - 1)
        
    def convert(self, A, start, end):
        if start > end:
            return None
        if start == end:
            return TreeNode(A[start])
        
        mid = (start + end) // 2
        root = TreeNode(A[mid])
        root.left = self.convert(A, start, mid - 1)
        root.right = self.convert(A, mid + 1, end)
        return root