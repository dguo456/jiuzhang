# 814 · Shortest Path in Undirected Graph
"""
Given an undirected graph in which each edge's length is 1, and two nodes from the graph. 
Return the length of the shortest path between two given nodes.
"""
from collections import deque

class UndirectedGraphNode:
     def __init__(self, x):
         self.label = x
         self.neighbors = []

class Solution:
    """
    @param graph: a list of Undirected graph node
    @param A: nodeA
    @param B: nodeB
    @return:  the length of the shortest path
    """
    def shortestPath(self, graph, A, B):
        if A == B:
            return 0

        queue = deque([A])
        visited = set([A])
        path = 0

        while queue:
            path += 1

            for _ in range(len(queue)):
                node = queue.popleft()
                visited.add(node)

                for neighbor in node.neighbors:
                    if neighbor == B:
                        return path
                    if neighbor not in visited:
                        visited.add(neighbor)
                        queue.append(neighbor)

        return -1

    # 另一种优化方法，建立distance用来存储所有中间路径，同时去重
    def shortestPath(self, graph, A, B):
        if A == B:
            return 0

        queue = deque([A])
        distance = {A : 0}

        while queue:
            node = queue.popleft()
            if node == B:
                return distance[node]
            for neighbor in node.neighbors:
                if neighbor in distance:
                    continue
                distance[neighbor] = distance[node] + 1
                queue.append(neighbor)

        return -1