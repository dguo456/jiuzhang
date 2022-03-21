# 717 · Tree Longest Path With Same Value

"""
Description
We consider an undirected tree with N nodes, numbered from 1 to N, 
Additionally, each node also has a label attached to it and the label is an integer value. 
Note that different nodes can have identical labels. You need to write a function , that
given a zero-indexed array A of length N, where A[J] is the label value of the (J + 1)-th node in the tree, 
and a zero-indexed array E of length K = (N - 1) * 2 in which the edges of the tree are described 
(for every 0 <= j <= N -2 values E[2 * J] and E[2 * J + 1] represents and edge connecting node E[2 * J] 
with node E[2 * J + 1])， returns the length of the longest path such that all the nodes on that path 
have the same label. Then length of a path if defined as the number of edges in that path.
"""
# Input: A = [1, 1, 1 ,2, 2] and E = [1, 2, 1, 3, 2, 4, 2, 5]
# Output: 2
# Explanation: 
# described tree appears as follows:

#                    1 （value = 1）
#                  /   \
#     (value = 1) 2     3 (value = 1)
#                /  \
#  (value = 2)  4     5 (value = 2)

# The longest path (in which all nodes have the save value of 1) is (2 -> 1 -> 3). 
# The number of edges on this path is 2, thus, that is the answer.

class Solution:
    """
    @param A: as indicated in the description
    @param E: as indicated in the description
    @return: Return the number of edges on the longest path with same value.
    """
    def LongestPathWithSameValue(self, A, E):
        n = len(A)
        graph = {i: set() for i in range(1, n+1)}

        for i in range(0, len(E), 2):
            node1, node2 = E[i], E[i+1]
            graph[node1].add(node2)
            graph[node2].add(node1)

        results = [0]
        self.dfs(1, -1, A, graph, results)
        return results[0] - 1

    def dfs(self, node, father, A, graph, results):
        v = [0, 0]
        for neighbor in graph[node]:
            if neighbor == father:
                continue
            if A[node - 1] == A[neighbor - 1]:
                v.append(self.dfs(neighbor, node, A, graph, results))
            else:
                self.dfs(neighbor, node, A, graph, results)

        v.sort(reverse=True)
        results[0] = max(results[0], v[0] + v[1] + 1)
        return v[0] + 1