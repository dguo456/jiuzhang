# 7 · Serialize and Deserialize Binary Tree
# 1235 · Serialize and Deserialize BST          答案一样
# 1108 · Find Duplicate Subtrees
# 1137 · Construct String from Binary Tree
# 1195 · Find Largest Value in Each Tree Row
# 1533 · N-ary Tree Level Order Traversal

##############################################################################################


# Definition of TreeNode:
class TreeNode:
    def __init__(self, val):
        self.val = val
        self.left, self.right = None, None

from collections import Counter, defaultdict, deque

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
    # tree:   {3,9,20,#,#,15,7}
    # output: [3 9 20 # # 15 7 # # # #]
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
    # order: [3, 9, 20, '#', '#', 15, 7, '#', '#', '#', '#']
    # tree:  [3, 9, 20, 15, 7]
    # Method.1
    def deserialize(self, data):
        if not data:
            return None
        
        bfs_order = data.split()   
        root = TreeNode(int(bfs_order[0])) 
        
        queue = [root]   # 二叉树的原型
        isLeftChild = True
        index = 0
        
        for val in bfs_order[1:]:   # 第一位已经交给root了
            if val is not "#":
                node = TreeNode(int(val))
                if isLeftChild:
                    queue[index].left = node
                else: 
                    queue[index].right = node
                queue.append(node)
            
            if not isLeftChild:
                index += 1
            isLeftChild = not isLeftChild
            
        return root


    # Method.2      有点乱，更推荐用第一种方法
    def deserialize(self, data):
        if not data:
            return None
            
        order = [TreeNode(int(val)) if val != '#' else None for val in data.split()]   # 记住
        
        root = order[0]
        tree = [root]           # 这里是为了防止index溢出
        slow, fast = 0, 1
        
        while slow < len(tree): # 必须用tree的长度作为判断，防止index溢出
            node = tree[slow]   # 必须用tree，不能用order
            node.left = order[fast]
            node.right = order[fast + 1]
            slow += 1
            fast = 2 * slow + 1  # 记住
            
            if node.left:
                tree.append(node.left)
            if node.right:
                tree.append(node.right)
            
        return root






# 1108 · Find Duplicate Subtrees
"""
Given a binary tree, return all duplicate subtrees. For each kind of duplicate subtrees, 
you only need to return the root node of any one of them.
Two trees are duplicate if they have the same structure with same node values.
"""
class Solution(object):

    # Method.1      将一棵二叉树的所有结点作为根节点进行序列化，记录该前序序列化字符串出现的次数。
    #                   1、如果出现的次数大于1，那么就说明该序列重复出现。
    #                   2、如果等于1，说明在这之前遇到过一次节点。
    #               最后统计完重复的后，返回结果；如果是空结点的话，返回一个任意非空字符串。
    #               时间复杂度：O(N), 空间复杂度：O(N)
    def findDuplicateSubtrees(self, root):
        if not root:
            return []

        counter = Counter()
        results = []

        self.dfs(root, counter, results)
        return results

    def dfs(self, root, counter, results):
        if not root: 
            return "#"

        serial = "{},{},{}".format(
            root.val, 
            self.dfs(root.left, counter, results), 
            self.dfs(root.right, counter, results)
        )
        counter[serial] += 1
        if counter[serial] == 2:
            results.append(root)

        return serial


    # Method.2      DFS
    def findDuplicateSubtrees(self, root):
        if not root:
            return []

        treeRecored = defaultdict(int)
        results = []
        
        self.dfs(root, treeRecored, results)
        return list(results)

    def dfs(self, node, treeRecored, results):
        if node == None:
            return []

        left = self.dfs(node.left, treeRecored, results)
        right = self.dfs(node.right, treeRecored, results)

        treeOrders = left + [node.val] + right
        if treeRecored[tuple(treeOrders)] == 1:                
            results.append(node)
        treeRecored[tuple(treeOrders)] += 1

        return treeOrders






# 1137 · Construct String from Binary Tree
"""
You need to construct a string consists of parenthesis and integers from a binary tree 
with the preorder traversing way.
The null node needs to be represented by empty parenthesis pair "()". And you need to omit all the 
empty parenthesis pairs that don't affect the one-to-one mapping relationship between the string and 
the original binary tree.

Input: Binary tree: [1,2,3,4]
       1
     /   \
    2     3
   /    
  4     

Output: "1(2(4))(3)"

Explanation: Originallay it needs to be "1(2(4)())(3()())", 
but you need to omit all the unnecessary empty parenthesis pairs. 
And it will be "1(2(4))(3)".
"""
class Solution:
    """
    @param t: the root of tree
    @return: return a string
    """
    # Method.1      类似于树的前序遍历，只需在遍历时在左子树和右子树最外面加一对括号即可。
    # 注意如果右子树为空，则右子树不需要加括号；若左子树为空而右子树非空，则需要在右子树前加一对空括号表示左子树。
    def tree2str(self, t):
        if t is None:
            return ""

        s = str(t.val)
        hasLeft = False

        if t.left is not None:
            hasLeft = True
            s += '(' + self.tree2str(t.left) + ')'
        if t.right is not None:
            if not hasLeft:
                s += '()'
            s += '(' + self.tree2str(t.right) + ')'

        return s


    # Method.2      只用检查两种情况。是否同时没有左分支和右分支。是否没有右分支。
    def tree2str(self, t):
        if t is None:
            return ""
        
        right = "(" + self.tree2str(t.right) + ")"
        left = "(" + self.tree2str(t.left) + ")"
        
        s = str(t.val)
        
        if left == "()" and right == "()":
            return s
        if right == "()":
            return s + left
        
        return s + left + right


    # Method.3      Time: O(N), Space: O(1), 分四種case: 1.兩邊都有, 2.只有左邊, 3.只有右邊, 4.葉子節點
    def tree2str(self, root):
        return self.helper(root)

    def helper(self, root):
        if not root:
            return ''
        left = self.helper(root.left)
        right = self.helper(root.right)

        if left and right:
            return str(root.val) + '(' + left + ')' + '(' + right + ')'
        if left:
            return str(root.val) + '(' + left + ')'
        if right:
            return str(root.val) + '()' + '(' + right + ')'
        # 4.葉子節點
        return str(root.val)


    # Method.4      
    def tree2str(self, t):
        if not t:
            return ""
        if not t.left and not t.right:
            return str(t.val)
        if not t.right:
            return str(t.val) + "(" + self.tree2str(t.left) + ")"

        return str(t.val) + "(" + self.tree2str(t.left) + ")(" + self.tree2str(t.right) + ")"






# 1195 · Find Largest Value in Each Tree Row
"""
You need to find the largest value in each row of a binary tree.
Input:
{1,3,2,5,3,#,9}
Output:
[1,3,9]
"""
class Solution:
    """
    @param root: a root of integer
    @return: return a list of integer
    """
    def largest_values(self, root: TreeNode):
        if not root:
            return []

        queue = deque([root])
        results = []

        while queue:
            level = []
            len_q = len(queue)

            for _ in range(len_q):
                node = queue.popleft()
                level.append(node.val)

                if node.left:
                    queue.append(node.left)
                if node.right:
                    queue.append(node.right)

            results.append(max(level))

        return results






# 1533 · N-ary Tree Level Order Traversal
"""
Given an n-ary tree, return the level order traversal of its nodes' values. 
(ie, from left to right, level by level).
Input:      {1,3,2,4#2#3,5,6#4#5#6}
Output:     [[1],[3,2,4],[5,6]]
"""

# Definition for a UndirectedGraphNode:
class UndirectedGraphNode:
    def __init__(self, label):
        self.label = label
        self.neighbors = []


class Solution:
    """
    @param root: the tree root
    @return: the order level of this tree
    """
    def level_order(self, root: UndirectedGraphNode):
        if not root:
            return []

        queue = deque([root])
        results = []

        while queue:
            level = []
            len_q = len(queue)

            for _ in range(len_q):
                node = queue.popleft()
                level.append(node.label)

                for neighbor in node.neighbors:
                    queue.append(neighbor)

            results.append(level[:])

        return results






