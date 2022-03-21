# 448 · Inorder Successor in BST
"""Given a binary search tree and a node in it, find the in-order successor of that node in the BST.
If the given node has no in-order successor in the tree, return null."""


# Definition for a binary tree node.
class TreeNode(object):
    def __init__(self, x):
        self.val = x
        self.left = None
        self.right = None


# Method.1      DFS
"""
首先要确定中序遍历的后继:
如果该节点有右子节点, 那么后继是其右子节点的子树中最左端的节点
如果该节点没有右子节点, 那么后继是离它最近的祖先, 该节点在这个祖先的左子树内.

使用循环实现:
查找该节点, 并在该过程中维护上述性质的祖先节点
查找到后, 如果该节点有右子节点, 则后继在其右子树内; 否则后继就是维护的那个祖先节点

使用递归实现:
如果根节点小于或等于要查找的节点, 直接进入右子树递归
如果根节点大于要查找的节点, 则暂存左子树递归查找的结果, 如果是 null, 则直接返回当前根节点; 反之返回左子树递归查找的结果.
在递归实现中, 暂存左子树递归查找的结果就相当于循环实现中维护的祖先节点.
"""
class Solution:
    """
    @param: root: The root of the BST.
    @param: p: You need find the successor node of p.
    @return: Successor of p.
    """
    def inorderSuccessor(self, root, p):
        if root is None:
            return None

        if root.val <= p.val:
            node_in_right_tree = self.inorderSuccessor(root.right, p)
            return node_in_right_tree        # 注意这里要return
        else:
            node_in_left_tree = self.inorderSuccessor(root.left, p)
            if node_in_left_tree is not None:
                return node_in_left_tree
            else:
                return root


# Method.2      DFS Inorder Traversal
"""
如果p>=现在的root的话，那p的后继承一定是在整个右子树（因为inorder的顺序就是“左中右”)
相反，如果p<现在的root的话，那他的下一项有两个可能
1)如果left子树是空的话，那只有一个可能就是p的root
2)如果left子树不是空的话，答案就在整个左子树里
并且在recursion里面也已经考虑了p的右子树
（等于找寻大于p的树然后最小的数字）
"""
class Solution:
    """
    @param: root: The root of the BST.
    @param: p: You need find the successor node of p.
    @return: Successor of p.
    """
    def inorderSuccessor(self, root, p):
        if root is None:
            return None

        self.prev_node = None
        self.result = None

        self.dfs(root, p)
        return self.result

    def dfs(self, root, target_node):
        if root is None:
            return

        self.dfs(root.left, target_node)

        if self.prev_node and self.prev_node.val == target_node.val:
            self.result = root
        self.prev_node = root

        self.dfs(root.right, target_node)

# Another Method
class Solution:
    """
    @param: root: The root of the BST.
    @param: p: You need find the successor node of p.
    @return: Successor of p.
    """
    def inorderSuccessor(self, root, p):
        if root is None:
            return None

        self.successor = None
        self.result = None
        self.dfs(root, p)
        return self.result

    def dfs(self, root, p):
        if root is None:
            return

        # 这里我们要有一个变量指针不断的指向root的后面一个节点，那么就先遍历右子树，再遍历左子树，就可以指向root的后面
        self.dfs(root.right, p)
        if root == p:
            self.result = self.successor
            return
        self.successor = root
        self.dfs(root.left, p)


# Method.3      Non-Recursive   O(logn) or O(h)
class Solution:
    """
    @param: root: The root of the BST.
    @param: p: You need find the successor node of p.
    @return: Successor of p.
    """
    def inorderSuccessor(self, root, p):
        if root is None:
            return None

        temp, successor = root, None

        while temp:
            if temp.val <= p.val:
                temp = temp.right
            else:
                successor = temp
                temp = temp.left

        return successor


# Method.4      Non_Recursive  O(n) time, O(n) space, the worst
import sys
class Solution:
    """
    @param: root: The root of the BST.
    @param: p: You need find the successor node of p.
    @return: Successor of p.
    """
    def inorderSuccessor(self, root, p):
        if root is None:
            return None

        dummy = TreeNode(-sys.maxsize)
        dummy.right = root
        stack = [dummy]
        prev, result = None, None

        while stack:
            node = stack.pop()
            prev = node     # 每次pop出当前node要复制到prev

            if node.right:
                node = node.right
                while node:
                    stack.append(node)
                    node = node.left

            # 只有等当前层循环进栈完毕之后，才能保证栈顶node一定是prev的后继
            # 这里还要判断当前node是否是树的最后一个node，pop出来之后stack即为空
            if stack and prev.val == p.val:
                result = stack[-1]

        return result





# 915 · Inorder Predecessor in BST

# Method.1      DFS
class Solution:
    """
    @param root: the given BST
    @param p: the given node
    @return: the in-order predecessor of the given node in the BST
    """
    def inorderPredecessor(self, root, p):
        if root is None:
            return None

        if root.val >= p.val:
            return self.inorderPredecessor(root.left, p)
        else:
            right = self.inorderPredecessor(root.right, p)
            if right:
                return right
            else:
                return root


# Method.2      DFS  Inorder Traversal
class Solution:
    """
    @param root: the given BST
    @param p: the given node
    @return: the in-order predecessor of the given node in the BST
    """
    def inorderPredecessor(self, root, p):
        if root is None:
            return None
        
        self.predecessor, self.result = None, None
        self.dfs(root, p)
        return self.result

    def dfs(self, root, p):
        if root is None:
            return

        self.dfs(root.left, p)

        if root.val == p.val:
            self.result = self.predecessor
        self.predecessor = root
        self.dfs(root.right, p)



# Method.3      None-Recuresion
class Solution:
    """
    @param root: the given BST
    @param p: the given node
    @return: the in-order predecessor of the given node in the BST
    """
    def inorderPredecessor(self, root, p):
        if root is None:
            return None

        temp, predecessor = root, None

        while temp:
            if temp.val >= p.val:
                temp = temp.left
            else:
                predecessor = temp
                temp = temp.right

        return predecessor



# Method.4      Non-Recursive  O(n) time, O(n) space, the worst
class Solution:
    """
    @param root: the given BST
    @param p: the given node
    @return: the in-order predecessor of the given node in the BST
    """
    def inorderPredecessor(self, root, p):
        if root is None:
            return None

        dummy = TreeNode(-sys.maxsize)
        dummy.right = root
        stack = [dummy]
        prev, result = None, None

        while stack:
            node = stack.pop()

            if node.val == p.val:
                # 注意prev有可能是dummy，要单独讨论这种情况
                if prev.val == dummy.val:
                    return result
                else:
                    result = prev
            prev = node

            if node.right:
                node = node.right
                while node:
                    stack.append(node)
                    node = node.left

        return result