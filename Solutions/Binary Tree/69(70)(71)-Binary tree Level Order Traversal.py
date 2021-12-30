# 69 - Binary Tree Level Order Traversal I&II
class TreeNode:
    def __init__(self, val):
        self.val = val
        self.left, self.right = None, None

from collections import deque

class Solution:
    """
    @param root: A Tree
    @return: Level order a list of lists of integer
    """
    def levelOrder(self, root):
        if root is None:
            return []
            
        queue = deque([root])
        results = []
        # i = 0
        
        while queue:
            level = []
            # i += 1
            
            for _ in range(len(queue)):
                node = queue.popleft()
                level.append(node.val)

                if node.left:
                    queue.append(node.left)
                if node.right:
                    queue.append(node.right)
                    
            results.append(level)
            # Zigzag Level Order Traversal
            # if i % 2 == 0:
            #     results.append(list(reversed(level)))
            # else:
            #     results.append(level)

        return results
        # return list(reversed(results))       II