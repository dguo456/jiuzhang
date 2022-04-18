# 15 · Permutations
# 16 · Permutations II
# 51 · Previous Permutation
# 52 · Next Permutation
# 197 · Permutation Index
# 388 · Permutation Sequence

###################################################################################################


# 15 · Permutations
"""
Given a list of numbers, return all possible permutations of it.
"""
class Solution:
    """
    @param: nums: A list of integers.
    @return: A list of permutations.
    """
    # Method.1      标准DFS模板
    def permute(self, nums):
        if not nums:
            return [[]]
            
        permutations = []
        self.dfs(nums, [], set(), permutations)
        return permutations
        
    def dfs(self, nums, permutation, visited, permutations):
        if len(nums) == len(permutation):
            permutations.append(list(permutation))
            return
        
        for num in nums:
            if num in visited:
                continue
            permutation.append(num)
            visited.add(num)
            self.dfs(nums, permutation, visited, permutations)
            visited.remove(num)
            permutation.pop()

    # Method.2      BFS
    def permute(self, nums):
        
        if not nums:
            return [[]]
            
        stack = [[n] for n in nums]
        results = []
        
        while stack:
            last = stack.pop()
            if len(last) == len(nums):
                results.append(last)
                continue
            
            for i in range(len(nums)):
                if nums[i] not in last:
                    stack.append(last + [nums[i]])
                    
        return results






# 16 · Permutations II
"""Given a list of numbers with duplicate numbers in it. Find all unique permutations of it."""
class Solution:
    """
    @param: :  A list of integers
    @return: A list of unique permutations
    """

    def permuteUnique(self, nums):
        
        nums = sorted(nums)
        visited = {i: False for i in range(len(nums))}
        permutations = []
        self.dfs(nums, visited, [], permutations)
        return permutations
        
    def dfs(self, nums, visited, permutation, permutations):
        if len(permutation) == len(nums):
            permutations.append(permutation[:])
            return
        
        for i in range(len(nums)):
            if visited[i]:
                continue
            
            if i > 0 and nums[i] == nums[i-1] and visited[i-1]:
                continue
            
            visited[i] = True
            self.dfs(nums, visited, permutation + [nums[i]], permutations)
            visited[i] = False






# 51 · Previous Permutation
"""
Given a list of integers, which denote a permutation.
Find the previous permutation in ascending order.
Input:      [1,3,2,3]
Output:     [1,2,3,3]
"""
class Solution:
    """
    @param: nums: A list of integers
    @return: A list of integers that's previous permuation
    """
    # Method.1      观察原排列[1, 2, 4, 3, 5, 6]和上一个排列[1, 2, 3, 6, 5, 4]，可以看到原排列中，
    # 4是一个分界线。4是从末尾开始往前第一个非下降点，在4之前的元素位置不变，只需要改变4以及4之后的元素。
    # 可以将排列想象成一种特殊的进制，比如从"0099"到"0100"，要更新当前的百位，必然这个位置以后的十位和个位元素均已达到最大。
    # 排列中是完全倒序表示最大，所以，我们接下来对原排列4之后的元素进行降序排序。利用排列的性质，将这部分[3, 5, 6]直接翻转
    # 即变为降序。此时新的排列为[1, 2, 4, 6, 5, 3]，可以看到离最终的上一个排列结果已经比较接近。
    # 最后一步，我们将4和后面最小的元素进行交换，即排列变为[1, 2, 3, 6, 5, 4]，这个就是上一个排列。
    def previousPermuation(self, nums):
        if not nums or len(nums) == 0:
            return []
        
        length = len(nums)
        i = length - 1
        
        #找到离结尾最近的一个底点
        while i > 0 and nums[i] >= nums[i - 1]:
            i -= 1
        
        #这种情况说明list已经是完全递增，直接翻转就可以
        if i == 0:
            return list(reversed(nums))
        
        #找到底点右边比nums[i - 1]小的数字中最大的一个
        j = length - 1
        while nums[j] >= nums[i - 1]:
            j -= 1
        
        #交换两个数字，翻转底点之后的部分
        nums[i - 1], nums[j] = nums[j], nums[i - 1]
        return nums[:i] + list(reversed(nums[i:]))



    # method.2      从排列的最末尾开始，找到第一个下降点，下降点的意义为这个点之前的序列无需改动。
    #               然后,将后面的序列变为降序。从下降点开始扫描，找到第一个比她小的数字，交换即可。
    def previousPermuation(self, num):
        for i in range(len(num)-2, -1, -1):
            if num[i] > num[i+1]:
                break
        else:
            num.reverse()
            return num
        for j in range(len(num)-1, i, -1):
            if num[j] < num[i]:
                num[i], num[j] = num[j], num[i]
                break
        for j in range(0, (len(num) - i)//2):
            num[i+j+1], num[len(num)-j-1] = num[len(num)-j-1], num[i+j+1]
        return num



    # Method.3
    def previousPermuation(self, nums):
        # 寻找基准点
        datum = len(nums) - 1
        while datum > 0 and nums[datum] >= nums[datum-1]:
            datum -= 1

        # 将基准点后序列翻转，变为降序
        self.swapList(nums, datum, len(nums) - 1)

        # 找到基准点后元素最小值，与基准点元素交换
        if datum != 0:
            i = datum
            while nums[i] >= nums[datum-1]:
                i += 1

            nums[datum-1], nums[i] = nums[i], nums[datum-1]

        return nums

    # 翻转left-right之间的元素
    def swapList(self, nums, left, right):
        while (left < right):
            nums[left], nums[right] = nums[right], nums[left]
            left += 1
            right -= 1






# 52 · Next Permutation
"""
Given a list of integers, which denote a permutation.Find the next permutation in ascending order.
The list may contains duplicate integers.
Input:      [1,3,2,3]
Output:     [1,3,3,2]
"""
class Solution:
    """
    @param nums: A list of integers
    @return: A list of integers
    """
    # Method.1      观察原排列[1, 2, 3, 6, 5, 4]和下一个排列[1, 2, 4, 3, 5, 6]
    def next_permutation(self, nums):
        if not nums or len(nums) < 2:
            return nums
        i = len(nums) - 2
        while nums[i] >= nums[i+1]:
            i -= 1
            if i < 0:
                return nums[::-1]
        j = len(nums) - 1
        while nums[j] <= nums[i]:
            j -= 1
        nums[i], nums[j] = nums[j], nums[i]
        return nums[:i+1] + sorted(nums[i+1:])



    # Method.2      从最后一个位置开始，找到一个上升点，上升点之前的无需改动。 (有问题，过不了)
    #               然后，翻转上升点之后的降序。在降序里，找到第一个比上升点大的，交换位置。
    def nextPermutation(self, num):
        for i in range(len(num)-2, -1, -1):
            if num[i] < num[i+1]:
                break
        else:
            num.reverse()
            return num
        for j in range(len(num)-1, i, -1):
            if num[j] > num[i]:
                num[i], num[j] = num[j], num[i]
                break
        for j in range(0, (len(num) - i)//2):
            num[i+j+1], num[len(num)-j-1] = num[len(num)-j-1], num[i+j+1]
        return num



    # Method.3      从右往左遍历，找到一个可以替换的点，然后保证用点右边比它大的点中最小的那个和它替换。
    #               然后再保证点右边是升序的，就完成了下一个序列
    def nextPermutation(self, nums):
		# 倒序遍历
        for i in range(len(nums)-1, -1, -1):
            # 找到第一个数值变小的点，这样代表右边有大的可以和它换，而且可以保证是next permutation
            if i > 0 and nums[i] > nums[i-1]:
                # 找到后再次倒序遍历，找到第一个比刚才那个数值大的点，互相交换
                for j in range(len(nums)-1, i-1, -1):
                    if nums[j] > nums[i-1]:
                        nums[j], nums[i-1] = nums[i-1], nums[j]
                        # 因为之前保证了，右边这段数从右到左是一直变大的，所以直接双指针reverse
                        left, right = i, len(nums)-1
                        while left <= right:
                            nums[left], nums[right] = nums[right], nums[left]
                            left += 1 
                            right -= 1 
                        return nums
    	# 如果循环结束了，表示没找到能替换的数，表示序列已经是最大的了
        nums.reverse()
        return nums



    # Method.4      DFS + 剪枝. 先给数组排序, 然后用dfs按顺序寻找permutation, 只要发现prefix与nums不一样, 
    #               立即剪枝, 直到找到nums后, 停止剪枝, 下一个就是result, 找到result之后就剪掉剩下的所有枝. 
    #               特殊情况:如果没有下一个了, 那么说明nums是纯倒序, 则result是sorted(nums)
    def nextPermutation(self, nums):
        sortedNums = sorted(nums)
        self.used = [0] * len(nums)
        self.found = False
        self.result = None
        
        self.dfs(sortedNums, nums, [])
        
        if self.result is not None:
            return self.result
        else:
            return sortedNums
            
            
    def dfs(self, sortedNums, nums, pre):
        if self.result is not None:
            return
        
        if not self.found:
            n = len(pre)
            if pre[:n] != nums[:n]:
                return
            
        if pre == nums:
            self.found = True
            return
        
        if len(pre) == len(nums):
            self.result = pre[:]
            return
        
        for i in range(len(sortedNums)) :
            if self.used[i] == 0 and not (i > 0 and sortedNums[i-1] == sortedNums[i] and self.used[i-1] == 0):
                self.used[i] = 1
                pre.append(sortedNums[i])
                self.dfs(sortedNums, nums, pre)
                self.used[i] = 0
                pre.pop()






# 197 · Permutation Index
"""
Given a permutation which contains no repeated number, find its index in all the permutations 
of these numbers, which are ordered in lexicographical order. The index begins at 1.
Input:[1,2,4]       Output:1
Input:[3,2,1]       Output:6
"""
class Solution:
    """
    @param: A: An array of integers
    @return: A long integer
    """
    # 正序利用权值计算index，按照正常思维，从正向思维计算要比反向思维好想一些
    def permutationIndex(self, A):
        # 至少为1，初始的全排列为增序，index记录第几个全排列
        index = 1
        for i in range(len(A)):
            # count记录该数后面有几个比它小的数字
            count = 0
            # factor用来计算阶乘的权值
            factorial = 1
            for j in range(i + 1, len(A)):
                if A[j] < A[i]:
                    count += 1
            # if count > 0:
            #     for k in range(1, len(A) - i):
            #         factorial *= k
                factorial *= j - i
            index += factorial * count

        return index






# 388 · Permutation Sequence
"""
Given n and k, find the kth permutation of the dictionary order in the full permutation of n.
Input: n = 3, k = 4
Output: "231"
Explanation:
For n = 3, all permutations are listed as follows:
"123", "132", "213", "231", "312", "321"
"""
class Solution:
    """
    @param n: n
    @param k: the k th permutation
    @return: return the k-th permutation
    """
    # DFS 稍微加了点purning 若res已经到了k 没必要再找后面的了 直接跳出
    def getPermutation(self, n, k):
        results = []
        self.dfs(n, k, "", set(), results)
        return results[k - 1]
    
    def dfs(self, n, k, path, visited, res):
        if len(res) == k:
            return 
        if len(path) == n:
            res.append(path)
            return 
        
        for i in range(1, n + 1):
            if i in visited:
                continue
            
            visited.add(i)
            self.dfs(res, path + str(i), visited, n, k)
            visited.remove(i)