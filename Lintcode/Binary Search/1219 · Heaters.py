# 1219 · Heaters
"""
Winter is coming! Your first job during the contest is to design a standard heater 
with fixed warm radius to warm all the houses. Now, you are given positions of houses and heaters 
on a horizontal line, find out minimum radius of heaters so that all houses could be covered by those heaters.
So, your input will be the positions of houses and heaters seperately, and your expected output 
will be the minimum radius standard of heaters.

Input: [1,2,3,4],[1,4]
Output: 1
Explanation: The two heater was placed in the position 1 and 4. We need to use radius 1 standard, 
then all the houses can be warmed.
"""

# 整体思路是用二分法找到离每一个house最近的heater，然后打擂台比较选出距离最长的一对house和heater的距离
class Solution:
    """
    @param houses: positions of houses
    @param heaters: positions of heaters
    @return: the minimum radius standard of heaters
    """
    def findRadius(self, houses, heaters):
        if not houses or not heaters:
            return None

        heaters.sort()
        result = 0

        for house in houses:
            result = max(self.find_nearest_heater(house, heaters), result)

        return result

    def find_nearest_heater(self, house, heaters):
        start, end = 0, len(heaters) - 1
        while start + 1 < end:
            mid = (start + end) // 2
            if heaters[mid] == house:
                return 0
            elif heaters[mid] < house:
                start = mid
            else:
                end = mid

        return min(abs(heaters[start] - house), abs(heaters[end] - house))