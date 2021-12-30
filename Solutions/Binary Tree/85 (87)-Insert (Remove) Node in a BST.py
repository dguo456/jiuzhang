# 85 Insert Node in a Binary Search Tree

# Definition of TreeNode:
class TreeNode:
    def __init__(self, val):
        self.val = val
        self.left, self.right = None, None

class Solution:
    """
    @param: root: The root of the binary search tree.
    @param: node: insert this node into the binary search tree
    @return: The root of the new binary search tree.
    """
    # Method.1  DFS 分治
    def insertNode(self, root, node):
        return self.__helper(root, node)
    
     # helper函数定义成私有属性   
    def __helper(self, root, node):     
        if root is None:
            return node
        if node.val < root.val:
            root.left = self.__helper(root.left, node)
        else:
            root.right = self.__helper(root.right, node)
        return root

    # Method.2  在树上定位要插入节点的位置。***注意在BST上插入node一定会是在leaf node下面插入，不可能在中间***
    #           1. 如果它大于当前根节点，则应该在右子树中，如果没有右子树则将该点作为右儿子插入；若存在右子树则在右子树中继续定位。
    #           2. 如果它小于当前根节点，则应该在左子树中，处理同上。（二叉查找树中保证不插入已经存在的值）
    def insertNode(self, root, node):
        if root is None:
            return node
        
        curr = root
        while curr != node:
            if node.val < curr.val:
                if curr.left is None:
                    curr.left = node
                curr = curr.left
            else:
                if curr.right is None:
                    curr.right = node
                curr = curr.right
                
        return root



# 87 Remove Node in a Binary Search Tree
class TreeNode:
    def __init__(self, val):
        self.val = val
        self.left, self.right = None, None

# Method.1  推荐使用的算法，更直观
class Solution:
    """
    @param: root: The root of the binary search tree.
    @param: value: Remove the node with given value.
    @return: The root of the binary search tree after removal.
    """
    ans = []
    
    def removeNode(self, root, value):
        self.inorder(root, value)
        return self.build(0, len(self.ans)-1)
        
    def inorder(self, root, value):
        if root is None:
            return

        self.inorder(root.left, value)
        if root.val != value:
            self.ans.append(root.val)
        self.inorder(root.right, value)
    
    # 分治模板
    def build(self, left, right):
        if left == right:
            return TreeNode(self.ans[left])
        if left > right:
            return None

        mid = (left + right) // 2
        node = TreeNode(self.ans[mid])
        node.left = self.build(left, mid-1)
        node.right = self.build(mid+1, right)
        return node

# Method.2  使用分治法: O(h) time, O(h) space, h: height of tree
# Divide & Conquer Solution
class Solution:
    """
    @param: root: The root of the binary search tree.
    @param: value: Remove the node with given value.
    @return: The root of the binary search tree after removal.
    """
    def removeNode(self, root, value):
        # null case
        if root is None:
            return root

        # check if node to delete is in left/right subtree 定位要删除的node
        if value < root.val:
            root.left = self.removeNode(root.left, value)
        elif value > root.val:
            root.right = self.removeNode(root.right, value)
        else:
            # if root has: 2 childs/only one child/leaf node 定位了之后，分情况讨论
            if root.left and root.right:
                max = self.findMax(root)
                root.val = max.val
                root.left = self.removeNode(root.left, max.val)
            elif root.left:
                root = root.left
            elif root.right:
                root = root.right
            else:
                root = None

        return root

    # find max node in left subtree of root
    def findMax(self, root):
        node = root.left
        while node.right:
            node = node.right
        return node