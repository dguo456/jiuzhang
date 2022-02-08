# 880 · Construct Binary Tree from String
"""
You need to construct a binary tree from a string consisting of parenthesis and integers.
The whole input represents a binary tree. It contains an integer followed by zero, one or two 
pairs of parenthesis. The integer represents the root's value and a pair of parenthesis contains 
a child binary tree with the same structure.

You always start to construct the left child node of the parent first if it exists.

There will only be '(', ')', '-' and '0' ~ '9' in the input string.
An empty tree is represented by "" instead of "()".
"""

# Input: "-4(2(3)(1))(6(5))"
# Output: {-4,2,6,3,1,5}
# Explanation:
# The output is look like this:
#       -4
#      /  \
#     2    6
#    / \   / 
#   3   1 5   



# Definition of TreeNode:
class TreeNode:
    def __init__(self, val):
        self.val = val
        self.left, self.right = None, None

class Solution:
    """
    @param s: a string
    @return: a root of this tree
    """
    def str2tree(self, s):
        if not s:
            return None
        if "(" not in s:
            return TreeNode(int(s))

        index = 0
        stack = []

        while index < len(s):
            if s[index] == "(":
                index += 1
                continue
            if s[index] == ")":
                stack.pop()     # 如果发现是右括号就要stack.pop
                index += 1
                continue
            
            end_index = index
            # 这里考虑了负数的情况
            while s[end_index] != "(" and s[end_index] != ")":
                end_index += 1
            
            node = TreeNode(int(s[index:end_index]))
            # 链接当前根节点的左（右）儿子
            if stack:
                father = stack[-1]      #这里stack的最后一个元素一定是父节点，因为上面遇到)就pop出一个child
                if father.left is None:
                    father.left = node      #先要查看left，因为如果left为空则不可能轮到right，一定是先left
                else:
                    father.right = node
            stack.append(node)
            index = end_index       #注意index要一直往右走

        return stack[0]