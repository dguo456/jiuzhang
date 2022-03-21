# 528 · Flatten Nested List Iterator
"""
Given a nested list of integers, implement an iterator to flatten it.
Each element of a list may be an integer or a list.

Input: list = [[1,1],2,[1,1]]
Output: [1,1,2,1,1]
"""


# This is the interface that allows for creating nested lists.
# You should not implement it, or speculate about its implementation

class NestedInteger(object):
    def isInteger(self):
        # @return {boolean} True if this NestedInteger holds a single integer,
        # rather than a nested list.
        ...

    def getInteger(self):
        # @return {int} the single integer that this NestedInteger holds,
        # if it holds a single integer
        # Return None if this NestedInteger holds a nested list
        ...

    def getList(self):
        # @return {NestedInteger[]} the nested list that this NestedInteger holds,
        # if it holds a nested list
        # Return None if this NestedInteger holds a single integer
        ...


class NestedIterator(object):

    def __init__(self, nestedList):
        self.next_ele = None
        self.stack = []
        for ele in reversed(nestedList):
            self.stack.append(ele)
            
    # @return {int} the next element in the iteration
    def next(self):
        if self.next_ele is None:
            self.hasNext()
            
        temp, self.next_ele = self.next_ele, None
        return temp
        
    # @return {boolean} true if the iteration has more element or false
    def hasNext(self):
        if self.next_ele:
            return True
            
        while self.stack:
            top = self.stack.pop()
            if top.isInteger():
                self.next_ele = top.getInteger()
                return True
            for ele in reversed(top.getList()):
                self.stack.append(ele)
                
        return False

# Your NestedIterator object will be instantiated and called as such:
# i, v = NestedIterator(nestedList), []
# while i.hasNext(): v.append(i.next())