# 1561 · BST Node Distance
"""
Given an integer array (unordered) and two node values, construct a BST from it
(you need to insert nodes one-by-one with the given order to get the BST) and 
find the distance between two given nodes.

If two nodes do not appear in the BST, return -1
We guarantee that there are no duplicate nodes in BST
The node distance means the number of edges between two nodes
"""

# Definition of TreeNode:
class TreeNode:
    def __init__(self, val):
        self.val = val
        self.left, self.right = None, None

class Solution:
    """
    @param numbers: the given list
    @param node1: the given node1
    @param node2: the given node2
    @return: the distance between two nodes
    """
    def bstDistance(self, numbers, node1, node2):
        # Write your code here
        if numbers is None or len(numbers) < 2:
            return -1
        
        if not self.check(numbers, node1, node2):
            return -1
        
        root = self.buildTree(numbers)

        while node1 > root.val and node2 > root.val or node1 < root.val and node2 < root.val:
            if node1 > root.val and node2 > root.val:
                root = root.right
            else:
                root = root.left
                
        return self.findDis(root, node1) + self.findDis(root, node2)

    def buildTree(self, numbers):
        root = TreeNode(numbers[0])
        length = len(numbers)
        for i in range(1, length):
            self.insert(root, numbers[i])
        
        return root
    
    def insert(self, root, node):
        if root is None:
            return TreeNode(node)
        
        if root.val > node:
            root.left = self.insert(root.left, node)
        elif root.val < node:
            root.right = self.insert(root.right, node)
        
        return root
    
    def check(self, numbers, node1, node2):
        Set = set()
        for i in range(0, len(numbers)):
            Set.add(numbers[i])

        if node1 in Set and node2 in Set:
            return True
        return False
        
    def findDis(self, root, node):
        dis = 0
        while root.val != node:
            dis += 1
            if root.val > node:
                root = root.left
            else:
                root = root.right
        return dis