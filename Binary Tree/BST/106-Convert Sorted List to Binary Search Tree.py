# 106 · Convert Sorted List to Binary Search Tree
"""
在链表上使用分治的方法。
这个算法的时间复杂度是 O(n) 的。
要诀在于, 先把整个链表的长度取一次, 然后拿着这个长度和链表头一起作为参数来进行递归构造。
convert(head, length) 表示把从 head 开头的长度为 length 的那么多个节点, 转化为 bst。
return 两个值, 一个是转化后的 bst 根节点, 一个是链表上从 head 开始第 length + 1 个节点是谁。
***这样做的好处是, 你不需要用 O(n) 的时间去找链表的中点了, 直接 O(1) 从 return 里得到。***
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
        # 不能是length // 2 + 1, 有testcase过不了
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





# 242 · Convert Binary Tree to Linked Lists by Depth
"""
Given a binary tree, design an algorithm which creates a linked list of all the nodes at each depth 
(e.g., if you have a tree with depth D, you'll have D linked lists).

Input: {1,2,3,4}
Output: [1->null,2->3->null,4->null]
Explanation: 
        1
       / \
      2   3
     /
    4
"""
from collections import deque
class Solution:
    # @param {TreeNode} root the root of binary tree
    # @return {ListNode[]} a lists of linked list
    def binaryTreeToLists(self, root):
        if root is None:
            return []
            
        queue = deque([root])
        results = []
        
        dummy = ListNode(0)
        lastNode = None
        
        while queue:
            lastNode = dummy
            
            for _ in range(len(queue)):
                node = queue.popleft()
                
                lastNode.next = ListNode(node.val)
                lastNode = lastNode.next
                
                if node.left:
                    queue.append(node.left)
                if node.right:
                    queue.append(node.right)
                    
            results.append(dummy.next)
            
        return results





# 378 · Convert Binary Tree to Doubly Linked List
"""
Convert a binary tree to doubly linked list with in-order traversal.
Input:
	    3
	   / \
	  4   1
Output:4<->3<->1
"""

class DoublyListNode(object):
    def __init__(self, val, next=None):
        self.val = val
        self.next, self.prev = next, next

class Solution:
    """
    @param root: The root of tree
    @return: the head of doubly list node
    """
    # Method.1      Recursion
    def bstToDoublyList(self, root):
        if not root:
            return None

        dummy = DoublyListNode(0)
        tail = self.traverse(root, dummy)
        return dummy.next
    
    def traverse(self, root, head):
        if root is None:
            return head     # 注意这里必须要有返回值
        
        left_tail = self.traverse(root.left, head)

        curr_node = DoublyListNode(root.val)
        left_tail.next = curr_node
        curr_node.prev = left_tail
        
        right_tail = self.traverse(root.right, curr_node)
        return right_tail


    # Method.2      Divide&Conquer
    def bstToDoublyList(self, root):
        if not root:
            return None

        head, tail = self.divide_conquer(root)
        return head

    def divide_conquer(self, root):
        if root is None:
            return None, None

        left_head, left_tail = self.divide_conquer(root.left)
        right_head, right_tail = self.divide_conquer(root.right)

        curr_node = DoublyListNode(root.val)
        head = tail = curr_node

        if left_head or left_tail:
            head = left_head
            left_tail.next = curr_node
            curr_node.prev = left_tail
        if right_head or right_tail:
            tail = right_tail
            curr_node.next = right_head
            right_head.prev = curr_node

        return head, tail


    # Method.3      Iterative
    def bstToDoublyList(self, root):
        if not root:
            return None

        stack = []
        dummy = previous = DoublyListNode(0)

        while stack or root:
            if root:
                stack.append(root)
                root = root.left
                continue

            root = stack.pop()
            previous.next, root.prev = DoublyListNode(root.val), previous
            previous, root = previous.next, root.right

        dummy = dummy.next
        dummy.prev = None
        
        return dummy


    # Method.4      另一种思路，干脆先中序遍历存到一个stack里，然后再一点点建list
    def bstToDoublyList(self, root):
        self.stack = []
        self.dfs(root)
        if len(self.stack) == 0:
            return None
        
        head = prev = None
        for val in self.stack:
            node = DoublyListNode(val)
            if head is None:
                head = node
            else:
                prev.next = node
            node.prev = prev
            prev = node
            
        return head
        
        
    def dfs(self, root):
        if root is None:
            return None
            
        self.dfs(root.left)
        self.stack.append(root.val)
        self.dfs(root.right)