# 1849 · Grumpy Bookstore Owner
"""
There is a bookstore. On the next n days, customer[i] will arrive on the i-th day and leave at the end of that day.
However, the bookstore owner's temper is sometimes good but sometimes bad. We use an array of grumpy to indicate 
his temper is good or bad every day. If grumpy[i] = 1, it means that the owner's temper is very bad on the day of i.
If grumpy[i] = 0, it means that the owner has a good temper on the first day.
If the owner of the bookstore has a bad temper one day, it will cause all customers who come on that day to give 
bad reviews to the bookstore. But if one day you have a good temper, then all customers will give the bookstore 
favorable comments on that day.
The boss wanted to increase the number of people who gave favorable comments to the bookstore as much as possible 
and came up with a way. He can keep a good temper for X days in a row. But this method can only be used once.
So how many people in this bookstore can give favorable comments to the bookstore when they leave on this nn day?

Input:
[1,0,1,2,1,1,7,5]
[0,1,0,1,0,1,0,1]
3
Output: 
16
Explanation: 
The bookstore owner keeps themselves not grumpy for the last 3 days. 
The maximum number of customers that can be satisfied = 1 + 1 + 1 + 1 + 7 + 5 = 16.
"""

class Solution:
    """
    @param customers: the number of customers
    @param grumpy: the owner's temper every day
    @param x: X days
    @return: calc the max satisfied customers
    """
    def max_satisfied(self, customers, grumpy, x):
        n = len(customers)
        satisfied_sum = 0
        
        # 先统计原本就给好评的人数
        for i in range(n):
            if grumpy[i] == 0:
                satisfied_sum += customers[i]
                
        # 记录最多的可能会变好评的人数
        max_become_satisfied = 0
        # 记录窗口中会变好评的人数
        become_satisfied = 0
        
        left = 0
        # 移动窗口的右端点
        for right in range(n):
            if grumpy[right] == 1:
                become_satisfied += customers[right]
                
            # 如果窗口中的个数过多，则移动左端点
            if right - left + 1 > x:
                if grumpy[left] == 1:
                    become_satisfied -= customers[left]
                left += 1
            
            max_become_satisfied = max(max_become_satisfied, become_satisfied)
        
        return satisfied_sum + max_become_satisfied

    
    # Method.2
    def maxSatisfied(self, customers, grumpy, X):
        # write your code here
        base = 0
        curr_window = 0
        bounus = 0

        for i in range(len(customers)):
            base += (1-grumpy[i])*customers[i]
            curr_window += grumpy[i]*customers[i]

            if i >= X: 
                curr_window -= grumpy[i-X]*customers[i-X]

            bounus = max(bounus, curr_window)

        return base + bounus