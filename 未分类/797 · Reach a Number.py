# 797 · Reach a Number
"""
You are standing at position 0 on an infinite number line. There is a goal at position target.
On each move, you can either go left or right. During the n-th move (starting from 1), you take n steps.
Return the minimum number of steps required to reach the destination.

Input: target = 2
Output: 3
Explanation:
On the first move we step from 0 to 1.
On the second move we step  from 1 to -1.
On the third move we step from -1 to 2.
"""

# 这个思路是这样的。 首先因为最小步数， 所以想要走的足够快， 就一直往右走。 正数负数是对称的， 所以直接绝对值来做就可以了。那么就分3个情况。
# 1. 正好走到， 那直接return 
# 2. 走多了， 但是多了偶数步a， 那么这个时候， 只要把之前走a/2的那一步反过来就可以了， 步数不会多的。比如说， 你目标是8， 结果走到了10，
#    那么差了2， 除以2就是1， 那么第一步就走到-1就可以了。 本来1+3+6 = 10现在-1+3+6 = 8
# 3. 走多了， 但是多了奇数步b。那么如果下一步要走的是奇数步， 那么多1步就变成了偶数步， 所以要多走一步。 
# 4. 如果多了奇数步， 但是下一步是偶数步， 奇数+偶数还是奇数， 所以要多走2步。
# 接下来就是每一步能走多远的问题。 稍微想一想， 就是1+2+3+...+n= n(n+1) / 2 >= t。那么要求出n， 除了很暴力的O(n)
# 还可以不那么暴力的用二分法， 或者是算到根号n就停的方法来做。
class Solution:
    """
    @param target: the destination
    @return: the minimum number of steps
    """
    def reach_number(self, target: int) -> int:
        target = abs(target)
        steps = self.get_min_steps(target)
        distance = steps * (steps + 1) // 2

        if (distance - target) % 2 == 0:
            return steps
        else:
            if steps % 2 == 0:
                return steps + 1
            else:
                return steps + 2

    def get_min_steps(self, target):
        step = 0
        while step * step + step < 2 * target:
            step += 1
        return step
