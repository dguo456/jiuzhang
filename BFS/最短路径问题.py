# 814 · Shortest Path in Undirected Graph
# 1565 · Modern Ludo I
#############################################################################################################

import heapq
from collections import deque



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





# 1565 · Modern Ludo I
"""
There is a one-dimensional board with a starting point on the far left side of the board and an end point 
on the far right side of the board. There are several positions on the board that are connected to other positions,
ie if A is connected to B, then when chess falls at position A, you can choose whether to give up to throw dice 
and move the chess from A to B directly. And the connection is one way, which means that the chess cannot move 
from B to A. Now given the length and connections of the board, and you have a six-sided dice(1-6), output the 
minimum steps to reach the end point.

Input: length = 15 and connections = [[2, 8],
[6, 9]]
Output: 2
Explanation: 
1->6 (dice)
6->9 (for free)
9->15(dice)
"""
class Solution:
    """
    @param length: the length of board
    @param connections: the connections of the positions
    @return: the minimum steps to reach the end
    """
    # Method.1      两个队列交替，因为不用popleft，所以每层需要一个nextQueue来进行替换当前层的queue，
    #               因为不用每次popleft，不会有元素左移一位而增加时间复杂度的问题。如按照标准的popleft操作，则要用deque
    def modern_ludo(self, length, connections) -> int:
        graph = self.build_graph(length, connections)

        # queue = collections.deque([1])
        queue = [1]
        distance = {1: 0}

        while queue:
            next_queue = []

            for node in queue:
                for connected_node in graph[node]:
                    if connected_node in distance:
                        continue
                    queue.append(connected_node)
                    distance[connected_node] = distance[node]

            for node in queue:
                for next_node in range(node + 1, min(node + 7, length + 1)):
                    if next_node in distance:
                        continue
                    next_queue.append(next_node)
                    distance[next_node] = distance[node] + 1

            queue = next_queue

        return distance[length]


    # Method.2      SPFA解法，shortest path faster algorithm
    def modernLudo(self, length, connections):
        from collections import deque
        graph = self.build_graph(length, connections)
        
        queue = deque([1])
        distance = {
            i: float('inf')
            for i in range(1, length + 1)
        }
        distance[1] = 0

        while queue:
            node = queue.popleft()

            for next_node in graph[node]:
                if distance[next_node] > distance[node]:
                    distance[next_node] = distance[node]
                    queue.append(next_node)

            for next_node in range(node + 1, min(node + 7, length + 1)):
                if distance[next_node] > distance[node] + 1:
                    distance[next_node] = distance[node] + 1
                    queue.append(next_node)

        return distance[length]


    # Method.3      SPFA + 堆优化
    def modernLudo(self, length, connections):
        graph = self.build_graph(length, connections)
        queue = [(0, 1)]
        distance = {
            i: float('inf')
            for i in range(1, length + 1)
        }
        distance[1] = 0

        while queue:
            dist, node = heapq.heappop(queue)

            for next_node in graph[node]:
                if distance[next_node] > dist:
                    distance[next_node] = dist
                    heapq.heappush(queue, (dist, next_node))

            for next_node in range(node + 1, min(node + 7, length + 1)):
                if distance[next_node] > dist + 1:
                    distance[next_node] = dist + 1
                    heapq.heappush(queue, (dist + 1, next_node))
                    
        return distance[length]


    def build_graph(self, length, connections):
        graph = {i: set() for i in range(1, length + 1)}

        for x, y in connections:
            graph[x].add(y)

        return graph