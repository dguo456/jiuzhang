# Color Sort I

"""
使用一次扫描的办法。
设立三根指针，left, index, right。定义如下规则:
left 的左侧都是 0（不含 left）
right 的右侧都是 2（不含 right）
index 从左到右扫描每个数，如果碰到 0 就丢给 left，碰到 2 就丢给 right。碰到 1 就跳过不管。
"""
class Solution:
    """
    @param nums: A list of integer which is 0, 1 or 2 
    @return: nothing
    """
    def sortColors(self, nums):
        if not nums or len(nums) < 2:
            return nums
        
        left, index, right = 0, 0, len(nums) - 1
        while index <= right:
            if nums[index] == 0:
                nums[left], nums[index] = nums[index], nums[left]
                left += 1
                index += 1
            elif nums[index] == 2:
                nums[right], nums[index] = nums[index], nums[right]
                right -= 1
            else:
                index += 1


# Color Sort II
"""
算法: 分治法
运使用rainbowSort，或者说是改动过的quickSort，运用的是分治的思想，不断将当前需要处理的序列分成两个更小的序列处理。

算法思路
思路与quickSort大致相同，每次选定一个中间的颜色，这个中间的颜色用给出的k来决定，将小于等于中间的颜色的就放到左边，大于中间颜色的就放到右边，然后分别再递归左右两半。

代码思路
递归函数设置四个参数，序列需要处理区间的左右端点和处理的颜色区间
根据给定的颜色区间决定中间的颜色
将小于等于中间的颜色的就放到左边，大于中间颜色的就放到右边
递归左右两半，直到颜色区间长度等于1

复杂度分析
NN为序列长度，KK为颜色数

空间复杂度: O(logK)O(logK)

时间复杂度: O(NlogK)O(NlogK)

  - 每次是对KK分成左右进行递归，因此有logKlogK层，每层递归遍历到整个序列，长度为NN
"""
class Solution:
    """
    @param colors: A list of integer
    @param k: An integer
    @return: nothing
    """
    def sortColors2(self, colors, k):
        self.sort(colors, 1, k, 0, len(colors)-1)
        
    def sort(self, colors, color_from, color_to, index_from, index_to):
        if color_from == color_to or index_from == index_to:
            return
        
        color = (color_from + color_to) // 2
        left, right = index_from, index_to
        
        while left <= right:
            while left<=right and colors[left]<=color:
                left += 1
            while left<=right and colors[right]>color:
                right -= 1
            if left <= right:
                colors[left], colors[right] = colors[right], colors[left]
                left += 1
                right -= 1
                
        self.sort(colors, color_from, color, index_from, right)
        self.sort(colors, color+1, color_to, left, index_to)

# Method. 2  Counting Sort
"""
算法: 计数排序（counting sort）
-  题目要求不使用额外的数组，一种方法是使用彩虹排序(rainbow sort)，但是这样虽然做到了没有使用额外的空间，但是代价是时间复杂度变成了O(N logK)，那么是否有方法做到时间和空间的双赢呢？
-  我们重新考虑计数排序(counting sort)，这里我们需要注意到颜色肯定是1-k，那么k一定小于n，我们是否可以用colors自己本身这个数组作为count数组呢？
-  下面我们介绍一种不占用大量额外空间的计数排序的写法。

算法思路
我们用负数代表数字出现的次数，例如colors[i]=-cnt表示数字i出现了cnt次
代码思路
我们从左往右遍历colors数组
  - 若colors[i] > 0且colors[colors[i]] < 0，那么colors[colors[i]] -= 1
  - 若colors[i] > 0且colors[colors[i]] > 0，那么先用临时变量temp存下colors[i],将colors[colors[i]]赋值给colors[i]，再将colors[temp] = -1
    > 注意此时i指针并不需要指向下一个位置，因为交换过来的值还未进行计数
  - 若colors[i] < 0，跳过

倒着输出每种颜色
另外注意数组下标是从0开始，为了避免n==k导致数组越界的情况，本题中colors[i]对应的计数位为colors[colors[i] - 1]

复杂度分析
NN表示colors数组长度

空间复杂度: O(1)O(1)

时间复杂度: O(N)O(N)
"""
class Solution:
    """
    @param colors: A list of integer
    @param k: An integer
    @return: nothing
    """
    def sortColors2(self, colors, k):
        size = len(colors)
        if (size <= 0):
            return
        
        index =  0
        while index < size:
            temp = colors[index] - 1
            #遇到计数位，跳过
            if colors[index] <= 0:
                index += 1
            else:
                #已经作为计数位
                if colors[temp] <= 0:
                    colors[temp] -= 1
                    colors[index] = 0
                    index += 1
                #还未被作为计数位使用
                else:
                    colors[index], colors[temp] = colors[temp], colors[index]
                    colors[temp] = -1
        #倒着输出
        i = size - 1
        while k > 0:
            for j in range(-colors[k - 1]):
                colors[i] = k
                i -= 1
            k -= 1