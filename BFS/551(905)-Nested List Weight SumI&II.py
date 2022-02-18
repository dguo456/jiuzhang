# 551 · Nested List Weight Sum
"""
Given a nested list of integers, return the sum of all integers in the list weighted by their depth. 
Each element is either an integer, or a list -- whose elements may also be integers or other lists.

Input: the list [[1,1],2,[1,1]], 
Output: 10. 
Explanation:
four 1's at depth 2, one 2 at depth 1, 4 * 1 * 2 + 1 * 2 * 1 = 10
"""

"""
This is the interface that allows for creating nested lists.
You should not implement it, or speculate about its implementation

class NestedInteger(object):
    def isInteger(self):
        # @return {boolean} True if this NestedInteger holds a single integer,
        # rather than a nested list.

    def getInteger(self):
        # @return {int} the single integer that this NestedInteger holds,
        # if it holds a single integer
        # Return None if this NestedInteger holds a nested list

    def getList(self):
        # @return {NestedInteger[]} the nested list that this NestedInteger holds,
        # if it holds a nested list
        # Return None if this NestedInteger holds a single integer
"""


class Solution(object):
    # @param {NestedInteger[]} nestedList a list of NestedInteger Object
    # @return {int} an integer
    def depthSum(self, nestedList):
        if len(nestedList) == 0:
            return []
        sum = 0
        stack = []
        
        for n in nestedList:
            stack.append((n, 1))

        while stack:
            temp, d = stack.pop(0)
            if temp.isInteger():
                sum += d * temp.getInteger()
            else:
                # stack.append((temp.getList(), d+1))
                for i in temp.getList():
                    stack.append((i, d+1))

        return sum





# 905 · Nested List Weight Sum II
"""
Given a nested list of integers, return the sum of all integers in the list weighted by their depth.
Each element is either an integer, or a list -- whose elements may also be integers or other lists.
Different from the previous question where weight is increasing from root to leaf, now the weight is 
defined from bottom up. i.e., the leaf level integers have weight 1, and the root level integers have 
the largest weight.

Example 1:
Input: nestedList = [[1,1],2,[1,1]]
Output: 8
Explanation:
four 1's at depth 1, one 2 at depth 2

Example 2:
Input: nestedList = [1,[4,[6]]]
Output: 17
Explanation:
one 1 at depth 3, one 4 at depth 2, and one 6 at depth 1; 1*3 + 4*2 + 6*1 = 17
"""
class Solution:
    """
    @param nestedList: a list of NestedInteger
    @return: the sum
    """
    def depthSumInverse(self, nestedList):
        self.result, self.d = 0, 0
        self.find_depth(nestedList, 1)
        self.dfs(nestedList, self.d)
        return self.result

    def find_depth(self, nestedList, depth):
        self.d = max(depth, self.d)
        if len(nestedList) == 0:
            return
        for n in nestedList:
            if not n.isInteger():
                self.find_depth(n.getList(), depth + 1)

    def dfs(self, nestedList, depth):
        for n in nestedList:
            if not n.isInteger():
                self.dfs(n.getList(), depth - 1)
            else:
                self.result += depth * n.getInteger()