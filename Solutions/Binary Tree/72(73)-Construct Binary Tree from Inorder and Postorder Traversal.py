# 72 · Construct Binary Tree from Inorder and Postorder Traversal
"""
Given inorder and postorder traversal of a tree, construct the binary tree.
Input:
    inorder traversal = [1,2,3]
    postorder traversal = [1,3,2]
Output:
    {2,1,3}
"""

# Definition of TreeNode:
class TreeNode:
    def __init__(self, val):
        self.val = val
        self.left, self.right = None, None

class Solution:
    """
    @param inorder: A list of integers that inorder traversal of a tree
    @param postorder: A list of integers that postorder traversal of a tree
    @return: Root of a tree
    """
    def buildTree(self, inorder, postorder):
        
        if not inorder:
            return None
            
        root = TreeNode(postorder[-1])
        rootPos = inorder.index(postorder[-1])
        
        root.left = self.buildTree(inorder[:rootPos], postorder[:rootPos])
        root.right = self.buildTree(inorder[rootPos+1:], postorder[rootPos:-1])
        
        return root



# 73 · Construct Binary Tree from Preorder and Inorder Traversal
class Solution:
    """
    @param preorder : A list of integers that preorder traversal of a tree
    @param inorder : A list of integers that inorder traversal of a tree
    @return : Root of a tree
    """
    def buildTree(self, preorder, inorder):
        if preorder is None or len(preorder) == 0:
            return None
            
        root = TreeNode(preorder[0])
        rootPos = inorder.index(preorder[0])
        
        root.left = self.buildTree(preorder[1:rootPos+1], inorder[:rootPos])
        root.right = self.buildTree(preorder[rootPos+1:], inorder[rootPos+1:])
        
        return root