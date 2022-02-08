# Definition of TreeNode:
class TreeNode:
    def __init__(self, val):
        self.val = val
        self.left, self.right = None, None

from collections import deque

class Solution:
    """
    @param root: An object of TreeNode, denote the root of the binary tree.
    This method will be invoked first, you should design your own algorithm 
    to serialize a binary tree which denote by a root node to a string which
    can be easily deserialized by your own "deserialize" method later.
    Input:
        tree = {3,9,20,#,#,15,7}
    Output:
            "3 9 20 # # 15 7"
    """
    def serialize(self, root):
        if root is None:
            return ""
            
        queue = deque([root])
        order = []
        
        while queue:
            node = queue.popleft()
            order.append(str(node.val) if node is not None else '#') # str(node.val), not list comprehension but similar for 'append'
            
            # no need to check root.left & root.right since None == '#'
            if node:
                queue.append(node.left)
                queue.append(node.right)
                
        return ' '.join(order)

    """
    @param data: A string serialized by your serialize method.
    This method will be invoked second, the argument data is what exactly
    you serialized at method "serialize", that means the data is not given by
    system, it's given by your own serialize method. So the format of data is
    designed by yourself, and deserialize it here as you serialize it in 
    "serialize" method.
    """
    def deserialize(self, data):
        if not data:
            return None
            
        order = [TreeNode(int(val)) if val != '#' else None for val in data.split()]   # 记住
        
        root = order[0]
        tree = [root]
        slow, fast = 0, 1
        
        while slow < len(tree):
            node = tree[slow]
            slow += 1
            node.left = order[fast]
            node.right = order[fast + 1]
            fast = 2 * slow + 1  # 记住
            
            if node.left:
                tree.append(node.left)
            if node.right:
                tree.append(node.right)
            
        return root