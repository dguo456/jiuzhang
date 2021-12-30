
# 88 · Lowest Common Ancestor of a Binary Tree
"""
Definition of TreeNode:
class TreeNode:
    def __init__(self, val):
        self.val = val
        self.left, self.right = None, None
"""

class Solution:
    """
    @param: root: The root of the binary search tree.
    @param: A: A TreeNode in a Binary.
    @param: B: A TreeNode in a Binary.
    @return: Return the least common ancestor(LCA) of the two nodes.
    """
    def lowestCommonAncestor(self, root, A, B):
        # 如果 A 和 B 都在，return  LCA
        # 如果只有 A 在，return A
        # 如果只有 B 在，return B
        # 如果 A, B 都不在，return None
        if root is None:
            return None
        
        if root == A or root == B:
            return root
        
        left = self.lowestCommonAncestor(root.left, A, B)
        right = self.lowestCommonAncestor(root.right, A, B)
        
        # A 和 B 一边一个
        if left and right: 
            return root
        
        # 左子树有一个点或者左子树有LCA
        if left:
            return left
        
        # 右子树有一个点或者右子树有LCA
        if right:
            return right
        
        # 左右子树啥都没有
        return None



# 474 · Lowest Common Ancestor II (has parent)
"""
Definition of ParentTreeNode:
class ParentTreeNode:
    def __init__(self, val):
        self.val = val
        self.parent, self.left, self.right = None, None, None
"""

class Solution:
    """
    @param: root: The root of the tree
    @param: A: node in the tree
    @param: B: node in the tree
    @return: The lowest common ancestor of A and B
    """
    def lowestCommonAncestorII(self, root, A, B):
        
        if root is None:
            return None
        
        d = {}
        
        while A is not root:
            d[A] = True
            A = A.parent
            
        while B is not root:
            if B in d:
                return B
            B = B.parent
            
        return root



# 578 · Lowest Common Ancestor III (A or B might not in tree)
class Solution:
    """
    @param: root: The root of the binary tree.
    @param: A: A TreeNode
    @param: B: A TreeNode
    @return: Return the LCA of the two nodes.
    """
    def lowestCommonAncestor3(self, root, A, B):
        # write your code here
        a, b, lca = self.helper(root, A, B)
        if a and b:
            return lca
        else:
            return None

    def helper(self, root, A, B):
        if root is None:
            return False, False, None
            
        left_a, left_b, left = self.helper(root.left, A, B)
        right_a, right_b, right = self.helper(root.right, A, B)
        
        # 判断A and B是否在树里
        a = left_a or right_a or root == A
        b = left_b or right_b or root == B
        
        if root == A or root == B:
            return a, b, root

        if left and right:
            return a, b, root
        if left:
            return a, b, left
        if right:
            return a, b, right

        return a, b, None



# 1311 · Lowest Common Ancestor of a Binary Search Tree
"""
解题思路
这道题与88. 最近公共祖先相似，都是求树内两个节点的最近公共祖先（LCA），我们本题也继续采用深度优先搜索（DFS）的方法。
不同的是，这道题明确指出这是二叉搜索树（BST），我们复习一下BST的性质:
对任意节点N，左子树上的所有节点的值都小于等于节点N的值
对任意节点N，右子树上的所有节点的值都大于等于节点N的值
BST的左子树和右子树也都是 BST
*** 我们如果在搜索时充分利用BST的性质，就能够有效剪枝。***
算法流程
从根节点root开始，遍历整棵树
如果root等于p或q，那么root即为p和q的LCA。
如果root同时大于p和q，说明p和q 都在左子树上，那么将root.left作为根节点，继续第一步的操作。
如果root同时小于p和q，说明p和q 都在右子树上，那么将root.right作为根节点，继续第一步的操作。
如果以上情况都不成立，说明p和q分别在两颗子树上，那么root就是p和q的LCA。
复杂度分析
时间复杂度: O(N)，其中 N 为 BST 中节点的个数。在最坏的情况下，BST退化成链表，我们可能访问 BST 中所有的节点。
空间复杂度: O(N)，其中 N 为 BST 中节点的个数。所需开辟的额外空间主要是递归栈产生的，在最坏的情况下，BST退化成链表，那么递归栈的深度就是BST的节点数目。
"""
class Solution:
    """
    @param root: root of the tree
    @param p: the node p
    @param q: the node q
    @return: find the LCA of p and q
    """
    def lowestCommonAncestor(self, root, p, q):
    	# root 等于 p或q
        if root == p or root == q:
            return root
        # p, q 都在左子树
        if p.val < root.val and q.val < root.val:
            return self.lowestCommonAncestor(root.left, p, q)
        # p, q 都在右子树
        if p.val > root.val and q.val > root.val:
            return self.lowestCommonAncestor(root.right, p, q)
        # p, q 分别在左右子树，那么root即为结果
        return root