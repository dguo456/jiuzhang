# 1534 · Convert Binary Search Tree to Sorted Doubly Linked List
"""We want to transform this BST into a circular doubly linked list. 
    Each node in a doubly linked list has a predecessor and successor. 
    For a circular doubly linked list, the predecessor of the first element is the last element, 
    and the successor of the last element is the first element."""


# Definition of TreeNode:
class TreeNode:
    def __init__(self, val):
        self.val = val
        self.left, self.right = None, None

# Method.1      Inorder Traversal
class Solution:
    """
    @param root: root of a tree
    @return: head node of a doubly linked list
    """
    def treeToDoublyList(self, root):
        if not root:
            return None

        self.dummy = None
        self.prev = None

        self.dfs(root)
        self.dummy.left = self.prev
        self.prev.right = self.dummy

        return self.dummy

    def dfs(self, node):
        if node is None:
            return

        self.dfs(node.left)

        if self.dummy is None:
            self.dummy = node
        if self.prev:
            self.prev.right = node
            node.left = self.prev
        self.prev = node

        self.dfs(node.right)


# Method.2      标准分治
class Solution:
    """
    @param root: root of a tree
    @return: head node of a doubly linked list
    """
    def treeToDoublyList(self, root):
        if root is None:
            return None
        
        head, tail = self.dfs(root)
        
        # 因为题目要求是环，所以还要首尾相连
        tail.right = head
        head.left = tail
        
        return head
        
    def dfs(self, root):
        if root is None:
            return None, None
            
        left_head, left_tail = self.dfs(root.left)
        right_head, right_tail = self.dfs(root.right)
        
        # left tail <-> root
        if left_tail:
            left_tail.right = root
            root.left = left_tail
        # root <-> right head
        if right_head:
            root.right = right_head
            right_head.left = root
        
        # 其实 root 肯定是有的，第三个 or 只是为了好看
        head = left_head or root or right_head
        tail = right_tail or root or left_tail
        
        return head, tail