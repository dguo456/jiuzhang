# 15 · Permutations
# 16 · Permutations II
# 51 · Previous Permutation
# 52 · Next Permutation
# 197 · Permutation Index
# 198 · Permutation Index II
# 388 · Permutation Sequence
# 211 · String Permutation
# 10 · String Permutation II
# 916 · Palindrome Permutation
# 917 · Palindrome Permutation II


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
        if A is None or len(A) == 0:
            return 0

        index = 1
        for i in range(len(A)):
            count, factorial = 0, 1
            for j in range(i + 1, len(A)):
                if A[j] < A[i]:
                    count += 1
                factorial *= j - i
            index += factorial * count

        return index


    # Method.2      逆序
    def permutation_index(self, a) -> int:
        if a is None or len(a) == 0:
            return 0

        index = factorial = 1
        for i in range(len(a)-1, -1, -1):
            count = 0
            for j in range(i + 1, len(a)):
                if a[j] < a[i]:
                    count += 1
                
            index += factorial * count
            factorial *= len(a) - i

        return index







# 198 · Permutation Index II
"""
Given a permutation which may contain repeated numbers, find its index in all the permutations 
of these numbers, which are ordered in lexicographical order. The index begins at 1.

Input :[1,4,2,2]
Output:3

Input :[1,6,5,3,1]
Output:24
"""
class Solution:
    # 这道题和Permutation IndexI思想一样，计算每一位上数字是该位上第几个排列，再将每一位结果加和即可。
    # 只是这道题有重复元素，有无重复元素最大的区别在于原来的1!, 2!, 3!...等需要除以重复元素个数的阶乘。
    # 按照数字从低位到高位进行计算。每遇到一个重复的数字就更新重复元素个数的阶乘的值。
    # 从后往前遍历数组，用一个hashmap来记录重复元素个数。若新来的数不是重复元素，则加入hashmap
    def permutation_index_i_i(self, a) -> int:
        if not a or len(a) == 0:
            return 0

        index = factorial = multi_fact = 1
        counter = {}
        
        for i in range(len(a) - 1, -1, -1):
            counter[a[i]] = counter.get(a[i], 0) + 1
            multi_fact *= counter[a[i]]
            
            count = 0
            for j in range(i + 1, len(a)):
                if a[j] < a[i]:
                    count += 1

            index += count * factorial // multi_fact
            factorial *= len(a) - i

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






# 211 · String Permutation
"""Given two strings, write a method to decide if one is a permutation of the other."""
class Solution:
    """
    @param A: a string
    @param B: a string
    @return: a boolean
    """
    def Permutation(self, A, B):
        return sorted(A) == sorted(B)



# 10 · String Permutation II
"""Given a string, find all permutations of it without duplicates."""

class Solution:
    """
    @param str: A string
    @return: all permutations
    """
    def stringPermutation2(self, str):
        chars = sorted(list(str))
        visited = [False] * len(chars)
        permutations = []
        self.dfs(chars, visited, "", permutations)
        return permutations
        
    def dfs(self, chars, visited, permutation, permutations):
        if len(permutation) == len(chars):
            permutations.append(permutation)
            return
        
        for i in range(len(chars)):
            if visited[i]:
                continue
            
            if i > 0 and chars[i] == chars[i-1] and not visited[i-1]:
                continue
            
            visited[i] = True
            self.dfs(chars, visited, permutation + chars[i], permutations)
            visited[i] = False






# 916 · Palindrome Permutation
"""
Example:

Input: s = "aab"
Output: True
Explanation: 
"aab" --> "aba"
"""
from collections import Counter

class Solution:
    """
    @param s: the given string
    @return: if a permutation of the string could form a palindrome
    """
    def canPermutePalindrome(self, s):
        
        return sum(v % 2 for v in Counter(s).values()) < 2



# 917 · Palindrome Permutation II
"""
Given a string s, return all the palindromic permutations (without duplicates) of it. 
Return an empty list if no palindromic permutation could be form.

Example1
Input: s = "aabb"
Output: ["abba","baab"]

Example2
Input: "abc"
Output: []
"""
class Solution:
    """
    @param s: the given string
    @return: all the palindromic permutations (without duplicates) of it
    """
    def generatePalindromes(self, s):

        counter = {}
        # odds = filter(lambda x: x % 2, counter.values())
        for c in s:
            counter[c] = counter.get(c, 0) + 1
        odds =  [c for c in counter if counter[c] % 2 == 1]
        if len(odds) > 1:
            return []

        half_s = []
        for c in counter:
            half_s.extend([c] * (counter[c] // 2))  

        visited = [False] * len(half_s)
        permutations = []
        self.dfs(half_s, visited, "", permutations)

        results = []
        # 这里优化是因为正常的代码会超时，只用palindrome的一半传入做dfs
        for permutation in permutations:
            if odds:
                results.append(permutation + odds[0] + permutation[::-1])
            else:
                results.append(permutation + permutation[::-1])

        return results

    def dfs(self, chars, visited, permutation, permutations):
        if len(permutation) == len(chars):
            permutations.append(permutation)
            return

        for i in range(len(chars)):
            if visited[i]:
                continue

            if i > 0 and chars[i-1] == chars[i] and not visited[i-1]:
                continue

            visited[i] = True
            self.dfs(chars, visited, permutation + chars[i], permutations)
            visited[i] = False