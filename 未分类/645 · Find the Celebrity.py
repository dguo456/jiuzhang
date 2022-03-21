# 645 · Find the Celebrity
"""
Suppose you are at a party with n people (labeled from 0 to n - 1) and among them, there may exist one celebrity. 
The definition of a celebrity is that all the other n - 1 people know him/her but he/she does not know any of them.
Now you want to find out who the celebrity is or verify that there is not one. The only thing you are allowed to do 
is to ask questions like: "Hi, A. Do you know B?" to get information of whether A knows B. You need to find out the 
celebrity (or verify there is not one) by asking as few questions as possible (in the asymptotic sense).

You are given a helper function bool knows(a, b) which tells you whether A knows B. Implement a function 
int findCelebrity(n), your function should minimize the number of calls to knows.

Input: 2
0 knows 1
1 does not know 0
Output: 1
Explanation:
Everyone knows 1,and 1 knows no one.
"""
 
class Celebrity:
    def __init__(self,):
        pass
    
    def knows(self, a, b):
        ...

# Method.1      首先loop一遍找到一个人i使得对于所有j(j>=i)都不认识i。然后再loop一遍判断是否有人不认识i或者i认识某个人。
class Solution:
    # @param {int} n a party with n people
    # @return {int} the celebrity's label or -1
    def findCelebrity(self, n):
        
        celebrity = 0
        
        for i in range(1, n):
            if Celebrity.knows(celebrity, i):
                celebrity = i
                
        for i in range(n):
            if celebrity != i and Celebrity.knows(celebrity, i):
                return -1
            if celebrity != i and not Celebrity.knows(i, celebrity):
                return -1
                
        return celebrity



# Method.2      
# 第一种的优化，follow-up，假如knows这个api call很 expensive的话怎么办
# 就是用一个hashmap去空间换时间，把第一遍的结果存下来第二遍再用

class Solution:
    # @param {int} n a party with n people
    # @return {int} the celebrity's label or -1
    def findCelebrity(self, n):
        if not n or n < 0:
            return -1 
        celeb = 0
        memo = {}
        for i in range(1, n):
            if Celebrity.knows(celeb, i):
                memo[(celeb, i)] = True
                celeb = i
            else:
                memo[(celeb, i)] = False
        return self.is_celeb(celeb, n, memo)
        
    def is_celeb(self, celeb, n, memo):
        for i in range(n):
            if celeb == i:
                continue
            if (celeb, i) in memo and memo[(celeb, i)]:
                return -1
            if (i, celeb) in memo and not memo[(i, celeb)]:
                return -1
            if Celebrity.knows(celeb, i) or not Celebrity.knows(i, celeb):
                return -1 
        return celeb