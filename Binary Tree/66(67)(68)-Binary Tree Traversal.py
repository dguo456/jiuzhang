
# Binary Tree PreOrder - InOrder - PostOrder Traversal
"""Definition of TreeNode:"""
class TreeNode:
    def __init__(self, val):
        self.val = val
        self.left, self.right = None, None

# Version 0: Recursion 标准解法
class Solution:
    """
    @param root: The root of binary tree.
    @return: Preorder in ArrayList which contains node values.
    """
    def preorderTraversal(self, root):
        self.results = []
        self.traverse(root)
        return self.results
        
    def traverse(self, root):
        if root is None:
            return
        self.results.append(root.val)   # 此行在： 上(前序遍历) 中(中序遍历) 下(后序遍历)
        self.traverse(root.left)
        self.traverse(root.right)

    # Recursion 非标准解法
    def preorderTraversalII(self, root):
        if root is None:
            return []

        left = self.preorderTraversalII(root.left)
        right = self.preorderTraversalII(root.right)

        return [root.val] + left + right


# Version 1: Non-Recursion

# Pre-Order
class Solution:
    """
    @param root: The root of binary tree.
    @return: Preorder in list which contains node values.
    """
    def preorderTraversal(self, root):
        if root is None:
            return []
        stack = [root]  # 注意这里用stack不用queue
        preorder = []

        while stack:
            node = stack.pop()
            preorder.append(node.val)
            if node.right:
                stack.append(node.right)
            if node.left:
                stack.append(node.left)
        return preorder

    # Post-Order
    """
    后续遍历是先左子树，再右子树再到父结点，倒过来看就是先父结点，再右子树再左子树。
    是前序遍历改变左右子树访问顺序。 再将输出的结果倒序输出一遍就是后序遍历。
    """
    def postorderTraversal(self, root):
        # write your code here
        if root is None:
            return []
        stack = [root]
        postorder = []

        while stack:
            node = stack.pop()
            postorder.append(node.val)
            if node.left is not None:
                stack.append(node.left)
            if node.right is not None:
                stack.append(node.right)
        return postorder[::-1]

    # Another Post-Order Solution
    def postorderTraversal(self, root):
        if not root:
            return []
        ans, stack, cur = [], [], root
        while cur:
            stack.append(cur)
            if cur.left:
                cur = cur.left
            else:
                cur = cur.right
            
        while stack:
            cur = stack.pop()
            ans.append(cur.val)
            if stack and stack[-1].left == cur:
                cur = stack[-1].right
                while cur:
                    stack.append(cur)
                    if cur.left:
                        cur = cur.left
                    else:
                        cur = cur.right
        
        return ans

    # In-Order
    
    # Method.1      标准模板 Binary Tree Iterator
    def inorderTraversal(self, root):
        if root is None:
            return []
            
        # 创建一个 dummy node，右指针指向 root 并放到 stack 里
        dummy = TreeNode(0)
        dummy.right = root
        stack = [dummy]
        inorder = []
        
        # 每次将 iterator 挪到下一个点, 也就是调整 stack 使得栈顶到下一个点
        while stack:
            node = stack.pop()
            if node.right:
                node = node.right
                while node:
                    stack.append(node)
                    node = node.left
            if stack:
                inorder.append(stack[-1].val)
                
        return inorder

        # 第二种写法
        while stack:
            curr_node = stack.pop()
            inorder.append(curr_node.val)

            if curr_node.right:
                curr_node = curr_node.right
                while curr_node:
                    stack.append(curr_node)
                    curr_node = curr_node.left

        return inorder[1:]

    # Method.2      
    def inorderTraversal(self, root):
        # write your code here
        if root is None:
            return []
        stack = []    # 注意这里用stack不用queue
        inorder = []
        
        while root:
            stack.append(root)
            root = root.left
        
        while stack:
            curNode = stack.pop()
            inorder.append(curNode.val)
            
            if curNode.right:
                curNode = curNode.right
                while curNode:
                    stack.append(curNode)
                    curNode = curNode.left
        
        return inorder

    