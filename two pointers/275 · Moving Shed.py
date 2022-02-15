# 275 · Moving Shed
"""
There are some cars parked. Given an array of integers 'stops', represents where each car stops. 
Given an integer 'k', now you're going to build a moving shed. When the shed is required to move 
between these cars (the front end of the shed does not exceed the car in front, and the back end 
of the shed does not exceed the car in the back),The mobile shed can successfully cover 'k' vehicles 
at any position. Ask for the minimum length of the shed that meets the requirements.

Input: stops=[7,3,6,1,8], k=3
Output: 6
Explanation: these 5 cars are in positions 1,3,6,7,8 respectively.The shed needs to cover at least 3 cars, 
with a minimum length of 6, as it can cover 3 or more cars in [1,6], [2,7], [3,8]. If the length is 5, 
it only covers 1 car when built in [1,5] and [2,6] , which does not meet the conditions.
"""

class Solution:
    """
    @param stops: An array represents where each car stops.
    @param k: The number of cars should be covered.
    @return: return the minimum length of the shed that meets the requirements.
    """
    def calculate(self, stops, k):
        if not stops or len(stops) == 0:
            return 0
        
        stops.sort()

        if (len(stops) == k):
            return stops[-1] - stops[0] + 1

        # [1,3,6,7,8], k = 3
        # 初始化为【0，k】然后【1，k+1】，【2，k+2】。。。时刻保持k辆车被cover
        shed = stops[k] - stops[0]
        left, right = 1, k + 1

        while right < len(stops):
            shed = max(stops[right] - stops[left], shed)
            left += 1
            right += 1

        return shed