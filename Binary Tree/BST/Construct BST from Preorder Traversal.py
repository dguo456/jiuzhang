# 1008. Construct Binary Search Tree from Preorder Traversal (leetcode)
"""
Given an array of integers preorder, which represents the preorder traversal of a BST,
construct the tree and return its root. 
It is guaranteed that there is always possible to find a BST with the given requirements.

A binary search tree is a binary tree where for every node, any descendant of Node.left has a value 
strictly less than Node.val, and any descendant of Node.right has a value strictly greater than Node.val.
"""

from typing import List, Optional

# Definition for a binary tree node.
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

# Method.1      Time Complexity: O(n2)
class Solution1:

    preIndex = 0

    #  Main Function
    def constructTree(self, preorder: List[int]) -> Optional[TreeNode]:
        if not preorder or len(preorder) == 0:
            return None

        size = len(preorder)
        return self.constructTreeUtil(preorder, 0, size-1)
    
    # A recurseive function to construct Full from pre[].
    # preIndex is used to keep track of index in pre[[].
    def constructTreeUtil(self, preorder, low, high):

        # Base Case
        if(low > high):
            return None
    
        # The first node in preorder traversal is root. So take the node at preIndex 
        # from pre[] and make it root, and increment preIndex
        root = TreeNode(preorder[self.preIndex])
        self.preIndex += 1
    
        # If the current subarray has onlye one element, no need to recur
        if low == high:
            return root
    
        r_root = -1
    
        # Search for the first element greater than root
        for i in range(low, high+1):
            if (preorder[i] > root.val):
                r_root = i
                break
    
        # If no elements are greater than the current root, all elements are left children
        if r_root == -1:
            r_root = self.preIndex + (high - low)
    
        # Use the index of element found in preorder to divide preorder array in two parts. 
        # Left subtree and right subtree
        root.left = self.constructTreeUtil(preorder, self.preIndex, r_root-1)
        root.right = self.constructTreeUtil(preorder, r_root, high)
    
        return root
        
    
    def printInorder(self, root):
        if root is None:
            return
        self.printInorder(root.left)
        print(root.val)
        self.printInorder(root.right)


# Driver code for Solution1
preorder = [10, 5, 1, 7, 40, 50]

s1  = Solution1()
root = s1.constructTree(preorder)
print("Inorder traversal of the constructed tree: ")
s1.printInorder(root)





# Method.2       O(n) time complexity, set a range {min .. max} for every node
INT_MIN = float("-infinity")
INT_MAX = float("infinity")

class Solution2:

    preIndex = 0

    #  Main Function
    def constructTree(self, preorder: List[int]) -> Optional[TreeNode]:
        if not preorder or len(preorder) == 0:
            return None

        size = len(preorder)
        return self.constructTreeUtil(preorder, preorder[0], INT_MIN, INT_MAX, size)
    

    def constructTreeUtil(self, preorder, node_val, min, max, size):

        # Base Case
        if self.preIndex >= size:
            return None
    
        root = None
        
        # If current element of pre[] is in range, then only it is part of current subtree
        if node_val > min and node_val < max:
            
            # Allocate memory for root of this subtree and increment constructTreeUtil.preIndex
            root = TreeNode(node_val)
            self.preIndex += 1

            if self.preIndex < size:

                # Construct the subtree under root
                root.left = self.constructTreeUtil(preorder, preorder[self.preIndex], min, node_val, size)
                root.right = self.constructTreeUtil(preorder, preorder[self.preIndex], node_val, max, size)
    
        return root
        
    
    def printInorder(self, root):
        if root is None:
            return
        self.printInorder(root.left)
        print(root.val)
        self.printInorder(root.right)


s2  = Solution2()
root = s2.constructTree(preorder)
print("Inorder traversal of the constructed tree: ")
s2.printInorder(root)






# Method.3
"""1. Create an empty stack.
    2. Make the first value as root. Push it to the stack.
    3. Keep on popping while the stack is not empty and the next value is greater than stack's top value. 
        Make this value as the right child of the last popped node. Push the new node to the stack.
    4. If the next value is less than the stack's top value, make this value as the left child of the 
        stack's top node. Push the new node to the stack.
    5. Repeat steps 2 and 3 until there are no items remaining in pre[]. """

# A binary tree node
class TreeNode:
    def __init__(self, val = 0):
        self.val = val
        self.left = None
        self.right = None
 
class BinaryTree :
 
    # The main function that constructs BST from preorder[]
    def constructTree(self, preorder, size):
 
        # The first element of preorder[] is always root
        root = TreeNode(preorder[0])
 
        stack = []
        stack.append(root)
 
        i = 1
 
        # Iterate through rest of the size-1 items of given preorder array
        while i < size:
            temp = None
 
            # Keep on poping while the next value is greater than stack's top value.
            while stack and preorder[i] > stack[-1].val:
                temp = stack.pop()
             
            # Make this greater value as the right child and append it to the stack
            if temp != None:
                temp.right = TreeNode(preorder[i])
                stack.append(temp.right)
             
            # If the next value is less than the stack's top value, make this value as 
            # the left child of the stack's top node. append the new node to stack
            else :
                temp = stack[-1]
                temp.left = TreeNode(preorder[i])
                stack.append(temp.left)
            i = i + 1
         
        return root
     
    # A utility function to print
    # inorder traversal of a Binary Tree
    def printInorder(self, node):
        if node is None:
            return
         
        self.printInorder(node.left)
        print(node.val)
        self.printInorder(node.right)
 
# Driver code
tree = BinaryTree()
size = len(preorder)
root = tree.constructTree(preorder, size)
print("Inorder traversal of the constructed tree is ")
tree.printInorder(root)