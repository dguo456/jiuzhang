# 263 · Matching of parentheses
"""
Given a string containing just the characters '(', ')', determine if the input string is valid.
The brackets must close in the correct order, "()" and "()" are all valid but "(]" and ")(" are not.
"""
class Solution:
    """
    @param string: A string
    @return: whether the string is a valid parentheses
    """
    def matchParentheses(self, string):
        if not string or len(string) == 0:
            return True

        stack = []

        for s in string:
            if s == '(':
                stack.append(s)
            elif s == ')':
                if stack and stack[-1] == '(':
                    stack.pop()
                else:
                    stack.append(s)

        if stack:
            return False

        return True




# 1721 · Minimum Add to Make Parentheses Valid
"""
Given a string S of '(' and ')' parentheses, we add the minimum number of parentheses 
'(' or ')', and in any positions  so that the resulting parentheses string is valid.

Input: "())"      Output: 1
Input: "((("      Output: 3
Input: "()"       Output: 0
"""
class Solution:
    """
    @param S: the given string
    @return: the minimum number of parentheses we must add
    """
    def minAddToMakeValid(self, S):
        if not S or len(S) == 0:
            return 0

        stack = []
        
        for char in S:
            if char == '(':
                stack.append(char)
            elif char == ')':
                if stack and stack[-1] == '(':
                    stack.pop()
                else:
                    stack.append(char)

        return len(stack)




# 2506 · Remove the Invalid Parentheses
"""
You will get a string s which consisting of lowercase letters a-z, left parentheses '(' and right parentheses ')'.
Your task is to remove as few parentheses as you can so that the parentheses in s is valid.
You need to return a valid string.

Input:        s = "a(b(c(de)fgh)"
Output:            "a(b(cde)fgh)"
"""
class Solution:
    """
    @param s: A string with lowercase letters and parentheses
    @return: A string which has been removed invalid parentheses
    """
    def removeParentheses(self, s: str) -> str:
        if not s or len(s) == 0:
            return s

        stack = []

        for index, value in enumerate(s):
            if value == '(':
                stack.append((index, value))
            elif value == ')':
                if stack and stack[-1][1] == '(':
                    stack.pop()
                else:
                    stack.append((index, value))

        if stack:
            list_s = list(s)
            # 这里注意必须要从后往前delete，否则index会浮动
            for i, v in reversed(stack):
                del list_s[i]

            return "".join(list_s)
        return s




# 1089 · Valid Parenthesis String
"""
Given a string containing only three types of characters: '(', ')' and '*', 
write a function to check whether this string is valid. We define the validity of a string by these rules:
'*' could be treated as a single right parenthesis ')' or a single left parenthesis '(' or an empty string.
An empty string is also valid.

Example 2:
	Input: "(*)"
	Output:  true
	
	Explanation:
	'*' is empty.
	
Example 3:
	Input: "(*))"
	Output: true
	
	Explanation:
	use '*' as '('.
"""
# 有点脑筋急转弯，需要搞清楚如何处理*的情况
class Solution:
    """
    @param s: the given string
    @return: whether this string is valid
    """
    def checkValidString(self, s):
        if not s or len(s) == 0:
            return True

        n = len(s)
        left, cp = 0, 0
        for i in range(0, n):
            if s[i] == '(':
                left += 1
                cp += 1
            elif s[i] == '*':
                if left > 0:
                    left -= 1
                cp += 1
            else:
                if left > 0:
                    left -= 1
                cp -= 1
                if cp < 0:
                    return False
        
        if left > 0:
            return False
        else:
            return True