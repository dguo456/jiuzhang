# 1862 · Time to Flower Tree
"""
There is a tree of n nodes with Num.0 to n-1, where node 0 is the root node and the parent node of 
node i is father[i]. Now to water the tree, sprinkle water on the root node, the water will flow down 
every edge, from the father of node i to node i costs time[i], how long will it take to flow water 
to all nodes?

Input:
[-1,0,0]
[-1,3,5]
Output: 
5
Explanation:
The tree is look like this:
   0
 3/\5
1    2
From 0 to 1 need time 3 and from 0 to 2 need time 5. So total we need time 5.
"""

from collections import deque

class Solution:
    """
    @param father: the father of every node
    @param time: the time from father[i] to node i
    @return: time to flower tree
    """
    # Method.1      BFS 先处理一下father，变成map，这样查找起来 O(1), 然后用BFS逐层遍历
    def time_to_flower_tree(self, father, time) -> int:
        if not father:
            return -1

        graph = {i: set() for i in range(len(father))}

        for node, parent in enumerate(father):
            if node != 0:
                graph[parent].add(node)

        queue = deque([(0, 0)])
        max_length = -float('inf')

        while queue:
            node, length = queue.popleft()

            if len(graph[node]) == 0:
                max_length = max(length, max_length)
            else:
                for next_node in graph[node]:
                    queue.append((next_node, length + time[next_node]))

        return max_length



    # Method.2      DFS + Memo
    def timeToFlowerTree(self, father, time):
        max_time = 0
        memo = {}
        
        for i in range(1, len(father)):
            time_spent = self.dfs(father, time, memo, i)
            max_time = max(max_time, time_spent)

        return max_time
    
    def dfs(self, father, time, memo, root):
        if root == 0:
            return 0

        if root in memo:
            return memo[root]
        
        memo[root] = time[root] + self.dfs(father, time, memo, father[root])
        return memo[root]



    # Method.3      greedy search
    def timeToFlowerTree(self, father, time):
        timeToRoot, maxTime = {0:0}, 0

        for i in range(1, len(father)):
            if i not in timeToRoot:
                k, t = i, 0
                while k not in timeToRoot:
                    k, t = father[k], t + time[k]
                timeToRoot[i] = t + timeToRoot[k]
                maxTime = max(maxTime, timeToRoot[i])

        return maxTime