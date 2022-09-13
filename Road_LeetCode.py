# def isMatch(s: str, p: str):
#     if not p: return not s  # 结束条件
#
#     first_match = (len(s) > 0) and p[0] in {s[0], '.'}
#     # 先处理 `*`
#     if len(p) >= 2 and p[1] == '*':
#         # 匹配0个 | 多个
#         return isMatch(s, p[2:]) or (first_match and isMatch(s[1:], p))
#
#     # 处理 `.` ，匹配一个
#     return first_match and isMatch(s[1:], p[1:])
#
# print(isMatch("aabb","a.b*"))



# def letterCombinations(self, digits: str):
#     if not digits:
#         return list()
#
# phoneMap = {
#     "2": "abc",
#     "3": "def",
#     "4": "ghi",
#     "5": "jkl",
#     "6": "mno",
#     "7": "pqrs",
#     "8": "tuv",
#     "9": "wxyz",
# }
#
# def backtrack(index: int):
#     if index == len(digits):
#         combinations.append("".join(combination))
#     else:
#         digit = digits[index]
#         for letter in phoneMap[digit]:
#             combination.append(letter)
#             backtrack(index + 1)
#             combination.pop()
#
# digits = "234"
# combination = list()
# combinations = list()
# backtrack(0)
# print(combinations)



# import itertools
# def isValid(s):
#     """
#     :type s: str
#     :rtype: bool
#     """
#     dic = {'(': ')','?': '?'}
#     stack = ['?']
#     for c in s:
#         if c in dic:
#             stack.append(c)
#         elif dic[stack.pop()] != c:
#             return False
#     return len(stack) == 1#栈空时才能返回
#
# def generate_combinations(size):
#     combinations_brackets = []
#     for comb in list(itertools.combinations(range(1, 1 + 2 * size), size)):
#         tmp = ""
#         for i in range(1,2 * size+1):
#             if i in comb:
#                 tmp = tmp + "("
#             else:
#                 tmp = tmp + ")"
#         combinations_brackets.append(tmp)
#     return combinations_brackets
#
# #print(generate_combinations(3), len(generate_combinations(3)))
# parenthesis = []
# n=3
# for combination in generate_combinations(n):
#     if isValid(combination):
#         parenthesis.append(combination)
# print(parenthesis)



#32.最长有效括号
#o（n3）
# def isValid(s):
#     """
#     :type s: str
#     :rtype: bool
#     """
#     dic = {'(': ')','?': '?'}
#     stack = ['?']
#     for c in s:
#         if c in dic:
#             stack.append(c)
#         elif dic[stack.pop()] != c:
#             return False
#     return len(stack) == 1#栈空时才能返回
# s="(()(()()()(()())()(()()))()()())()(((()())((())(()()((()((((())(())()()(())()(()(()(())))))))(()()()))()()))))))(()())))((())())))()(((()(()))())((())))(())(((()()))))())))((()((()()(()))())(()))(())((())()(((()(((()))))()))()()())()()()(()(()(()()()(()(())(())))())()))())(())((())(()((((())((())((())(()()(((()))(()(((())())()(())))(()))))))(()(()(()))())(()()(()(((()()))()(())))(()()(())))))(()(()()())))()()())))))((())(()()(((()(()()))(())))(((()))())())())(((()((()((()())((()))(()()((()(())())(()))()())())))))()(()())))()()))(((()(()))((()((((())((())))((())()()))())()(((()()(((()))))))(((()))()(()(((())(())()()()))))()))()))))()(()))()()()))))()(()))()()(()())))(()))()())(())()())(())()()))(()())))((()())))())))))((()))())()()))))()))(((())(())()))()()((()))(((()))))((()((()))(())(((()))()()()())())())))(()(((())()())(())(((()()((())()))(()()(((())))((()(((()))(((((()(((())())))(())(()()((()(()(())())(((((())((()()))())(()())))()()()(()(((((((())))(()()()()((()(((())())())())))())())())))()((((())(((()()()())()))()()(()(()()))()))(())(()())))))()())()())))()()(())))))))((())()()(((()))()))())))))((()())((()())(((())()())()))(()(()()(((()(())()))()())()()(())()(()))))()))()()))))(())(()()(())((()))(()))((())))))(())))()))(()()(())))())()((())()))((()()(()())()()(()))())(((())()(((()((())()(()()()((()(()())(()())())((((())))())())))(()))(((())((()))))((()()(((())((())()()()))((()())()()())())))))((((()((()())))(())(())()()()(((((())())()()()(())())()((()(()())(((())((((()((()(((()))(()()))())()()(()(()(())))()))())))(()()(()))))))(()()())()()))()(())()("
# max_len = 0
# for i in range(0,len(s)+1):
#     for j in range(i+1,len(s)+1):
#         if isValid(s[i:j]):
#             max_len =  max(max_len,j-i)
# print(max_len)

#题解
# s = "((())(("
# if not s:
#     print(0)
# res = []
# stack = []
# for i in range(len(s)):
#     if stack and s[i] == ")":
#         res.append(stack.pop())
#         res.append(i)
#     if s[i] == "(":
#         stack.append(i)
# res.sort()
# #print(res)
# i = 0
# ans = 0
# n = len(res)
# while i < n:
#     j = i
#     while j < n - 1 and res[j + 1] == res[j] + 1:
#         j += 1
#     ans = max(ans, j - i + 1)
#     i = j + 1#计算下一个最长连续序列，当然从j之后开始
# print (ans)

# 作者：powcai
# 链接：https://leetcode-cn.com/problems/longest-valid-parentheses/solution/zui-chang-you-xiao-gua-hao-by-powcai/
# 来源：力扣（LeetCode）
# 著作权归作者所有。商业转载请联系作者获得授权，非商业转载请注明出处。


#39.组合总数

# def dfs(candidates, begin, size, path, res, target):
#     if target < 0:
#         return
#     if target == 0:
#         res.append(path)
#         return
#
#     for index in range(begin, size):
#         dfs(candidates, index, size, path + [candidates[index]],
#             res, target - candidates[index])
# candidates = [2,3,6,7]
# target = 7
# size = len(candidates)
# if size == 0:
#     print([])
# path = []
# res = []
# dfs(candidates, 0, size, path, res, target)
# print(res)

# 作者：liweiwei1419
# 链接：https://leetcode-cn.com/problems/combination-sum/solution/hui-su-suan-fa-jian-zhi-python-dai-ma-java-dai-m-2/
# 来源：力扣（LeetCode）
# 著作权归作者所有。商业转载请联系作者获得授权，非商业转载请注明出处。

#49. 字母异位词分组
#strs = ["","b",""]
#利用字典的键值对性质
# import collections
# strs = ["eat", "tea", "tan", "ate", "nat", "bat"]
# mp = collections.defaultdict(list)
# for st in strs:
#     key = "".join(sorted(st))
#     mp[key].append(st)
# print(list(mp.values()))

#55. 跳跃游戏
# nums=[2,5,0,0]
# max_length = 0
# while max_length < len(nums):
#     max_length = max(max_length, max_length + nums[max_length])
#     if max_length < len(nums) and nums[max_length] == 0:
#         print(False)
# print(True)

#56. 合并区间
#intervals = [[1,3],[2,6],[8,10],[15,18]]
#intervals = [[1,4],[4,5]]
#intervals = [[1,4],[0,4]]
# res = []
# res_res = []
# intervals.sort()
# def mergeable(numss,k):
#     num1,num2 = numss[k],numss[k+1]
#     if num1[-1] <= num2[-1] and num1[-1] >= num2[0]:
#         return True
#         # res_res = [num1[0],num2[-1]]
#     else: return False
# def mergetwo(numss,k):
#     num1,num2 = numss[k],numss[k+1]
#     res_res = [num1[0],num2[-1]]
#     del numss[k]
#     del numss[k]
#     numss.insert(0,res_res)
# i = 0
# while i < len(intervals)-1:
#     if mergeable(intervals,i):
#         mergetwo(intervals,i)
#         i = 0
#     else: i += 1
# print(intervals)

# intervals.sort(key=lambda x: x[0])
#
# merged = []
# for interval in intervals:
#     # 如果列表为空，或者当前区间与上一区间不重合，直接添加
#     if not merged or merged[-1][1] < interval[0]:
#         merged.append(interval)
#     else:
#         # 否则的话，我们就可以与上一区间进行合并
#         merged[-1][1] = max(merged[-1][1], interval[1])
#
# print(merged)

# 作者：LeetCode-Solution
# 链接：https://leetcode-cn.com/problems/merge-intervals/solution/he-bing-qu-jian-by-leetcode-solution/
# 来源：力扣（LeetCode）
# 著作权归作者所有。商业转载请联系作者获得授权，非商业转载请注明出处。

#64. 最小路径和
# grid = [[1,3,1],[1,5,1],[4,2,1]]
# f = grid
# m = len(f)
# n = len(f[0])
# s = [[f[0][0]] + [0] * (n - 1)] + [[0] * n for i in range(1,m)]
# for i in range(1,m):
#     s[i][0] = s[i-1][0] + f[i][0]
# for j in range(1,n):
#     s[0][j] = s[0][j-1] + f[0][j]
#
# for i in range(1,m):
#     for j in range(1,n):
#         s[i][j] = min(s[i-1][j] + f[i][j], s[i][j-1] + f[i][j])
# print(s[m-1][n-1])

#72. 编辑距离
# word1 = "horse"#source m i
# word2 = "ros"#target n j
#
# def substitudeCost(word1, word2, i, j):
#     print(i,j)
#     if word1[j - 1] == word2[i - 1]:
#         return 0
#     else:
#         return 1
# m = len(word2)
# n = len(word1)
# dp = [[m + 1 - i] + [0] * n for i in range(1, m + 1)] + [list(range(0, n + 1))]
# for k in range(1, m + 1):
#     i = m - k
#     for j in range(1, n + 1):
#         dp[i][j] = min(dp[i + 1][j] + 1, dp[i + 1][j - 1] + substitudeCost(word1, word2, k, j), dp[i][j - 1] + 1)
# print(dp)
# print(dp[0][n])

#75. 颜色分类
#冒泡排序
# nums = [2,0,2,1,1,0]
# for i in range(0, len(nums)):
#     for k in range(i, len(nums)):
#         j = i + len(nums) - k - 2
#         if nums[j] > nums[j + 1]:
#             temp = nums[j]
#             nums[j] = nums[j + 1]
#             nums[j + 1] = temp
# print(nums)

#76. 最小覆盖子串
#自己的算法逻辑可以，但最后一个示例超出时间限制，这种问题应当学会滑动窗口算法套路，用空间换时间。
# s = "ADOBECODEBANC"
# t = "ABC"
# s = "ab"
# t = "a"
# L = t
# target = ""
# length = 100000
# stack = []
# for i in range(0, len(s)):
#     L = t
#     new_stack = []
#     if s[i] in t:
#         for j in range(i, len(s)):
#             new_stack.append(s[j])
#             if s[j] in t:
#                 L = L.replace(s[j], "", 1)
#             if L == "" and len(new_stack) < length:
#                 stack = new_stack.copy()
#                 print(stack,length)
#                 length = len(stack)
# # print(stack,length)
# print("".join(stack))

"""滑动窗口法"""
#滑动窗口算法可以用以解决数组/字符串的子元素问题,它可以将嵌套的循环问题,转换为单循环问题,降低时间复杂度
# s = "ADOBECODEBANC"
# t = "ABC"
# from collections import defaultdict
# '''
# 如果hs哈希表中包含ht哈希表中的所有字符，并且对应的个数都不小于ht哈希表中各个字符的个数，那么说明当前的窗口是可行的，可行中的长度最短的滑动窗口就是答案。
# '''
# if len(s)<len(t):
#     print("")
# hs, ht = defaultdict(int), defaultdict(int)#初始化新加入key的value为0
# for char in t:
#     ht[char] += 1
# res = ""
# left, right = 0, 0 #滑动窗口
# cnt = 0 #当前窗口中满足ht的字符个数
# while right<len(s):
#     hs[s[right]] += 1
#     if hs[s[right]] <= ht[s[right]]: #必须加入的元素
#         cnt += 1 #遇到了一个新的字符先加进了hs，所以相等的情况cnt也+1
#     while left<=right and hs[s[left]] > ht[s[left]]:#窗口内元素都符合，开始压缩窗口
#         hs[s[left]] -= 1
#         left += 1
#     if cnt == len(t):
#         if not res or right-left+1<len(res): #res为空或者遇到了更短的长度
#             res = s[left:right+1]
#     right += 1
# print(res)
# 作者：lin-shen-shi-jian-lu-k
# 链接：https://leetcode-cn.com/problems/minimum-window-substring/solution/leetcode-76-zui-xiao-fu-gai-zi-chuan-cja-lmqz/
# 来源：力扣（LeetCode）
# 著作权归作者所有。商业转载请联系作者获得授权，非商业转载请注明出处。

"""78.子集"""
#回溯法
#nums = [1,2,3]
# result = []
# path = []
# k = 0
# def generate_subsets(path, start):
#     if start == len(nums):
#         result.append(path)
#         return
#     generate_subsets(path, start + 1)
#     generate_subsets(path + [nums[start]], start + 1)
# generate_subsets([], 0)
# print(result)

#迭代法
# res = [[]]
# for i in nums:
#     res = res + [[i] + num for num in res]
# print(res)


"""79.单词搜索"""
# def dfs(board, i, j, word):
#     if len(word) == 0: # 如果单词已经检查完毕
#         return True
#     if i < 0 or i >= len(board) or j < 0 or j >= len(board[0]) or word[0] != board[i][j]:  # 如果路径出界或者矩阵中的值不是word的首字母，返回False
#         return False
#     tmp = board[i][j]  # 如果找到了第一个字母，检查剩余的部分
#     board[i][j] = '0'
#     res = dfs(board,i+1,j,word[1:]) or dfs(board,i-1,j,word[1:]) or dfs(board,i,j+1,word[1:]) or dfs(board, i, j-1, word[1:]) # 上下左右四个方向搜索
#
#     board[i][j] = tmp  # 标记过的点恢复原状，以便进行下一次搜索
#     return res
#
# if __name__ == '__main__':
#     board = [["A","B","C","E"],["S","F","C","S"],["A","D","E","E"]]
#     word = "ABCCED"
#     if not board:   # 边界条件
#         print(False)
#     for i in range(len(board)):
#         for j in range(len(board[0])):
#             if dfs(board, i, j, word):
#                 print(True)
#                 exit()
#     print(False)

# 作者：z1m
# 链接：https://leetcode-cn.com/problems/word-search/solution/tu-jie-di-gui-shen-du-you-xian-sou-suo-by-z1m/
# 来源：力扣（LeetCode）
# 著作权归作者所有。商业转载请联系作者获得授权，非商业转载请注明出处。

"""84.柱状图中最大的矩形"""
#heights = [2,1,5,6,2,3]
#heights = [4,2,0,3,2,4,3,4]
#heights = [5,5,1,7,1,1,5,2,7,6]
#heights = [6,4,2,0,3,2,0,3,1,4,5,3,2,7,5,3,0,1,2,1,3,4,6,8,1,3]
'''用盛最多水的方法逻辑不完备，因为问题不一样，改变边界条件会破坏最终的最优解'''
# a, b = 0, len(heights) - 1
# res = min(heights) * (b + 1)
# while a < b:
#     if heights[a] <= heights[b]:
#         a += 1
#         res = max(min(heights[a:b+1]) * (b - a + 1), res)
#     else:
#         b -= 1
#         res = max(min(heights[a:b+1]) * (b - a + 1), res)
# print(res)

'''回溯法长一些的例子会超时'''
# def search(a, b, compare):
#     if a == b:
#         return compare
#     res = max(min(heights[a:b]) * (b - a), compare)
#     return max(search(a+1, b, res), search(a, b-1, res))
#
# a, b = 0, len(heights)
# initi = min(heights) * (b)
# s1 = search(a, b, initi)
# print(s1)

'''单调栈'''
#暴力法：得到每个柱的左右边界
#单调栈的特殊性刚好满足了这里的低-高-低的结构，能够利用递增性方便地找到每个柱的左右边界
# res = 0
# stack = []
# h = [0] + heights + [0]
# for idx_right in range(len(h)):
#     while stack and h[stack[-1]] > h[idx_right]:
#         h_mid = h[stack.pop()]
#         idx_left = stack[-1]
#         res = max(h_mid*(idx_right - idx_left - 1), res)
#     stack.append(idx_right)
# print(res)
# https://leetcode-cn.com/problems/largest-rectangle-in-histogram/solution/4ye-pptjie-shi-wei-he-ji-ru-he-jiang-dan-diao-zhan/
# https://leetcode-cn.com/problems/largest-rectangle-in-histogram/solution/dong-hua-yan-shi-dan-diao-zhan-84zhu-zhu-03w3/

"""85. 最大矩形"""
#matrix = [["1","0","1","0","0"],["1","0","1","1","1"],["1","1","1","1","1"],["1","0","0","1","0"]]
#如何判断一个矩阵，横向向右扫描1，碰到0或者边界停下来，得到最大边，然后纵向扫描边得到宽
#可以直接转换为上一个问题，对于每行，构建该行往上的heights，求取最大矩形面积，
#注意构建heights时，上方有全1才可以累加，上方有0则heights对应元素置零
# def largestRectangleArea(heights):
#     res = 0
#     stack = []
#     h = [0] + heights + [0]
#     for idx_right in range(len(h)):
#         while stack and h[stack[-1]] > h[idx_right]:
#             h_mid = h[stack.pop()]
#             idx_left = stack[-1]
#             res = max(h_mid * (idx_right - idx_left - 1), res)
#         stack.append(idx_right)
#     return res
#
# m = len(matrix)
# n = len(matrix[0])
# result = 0
# heights = [0] * n
# for i in range(m):
#     for j in range(n):
#         if matrix[i][j] == '0':
#             heights[j] = 0
#         else:
#             heights[j] += 1
#     result = max(largestRectangleArea(heights), result)
# print(result)

"""136. 只出现一次的数字"""
#用字典统计出现次数
# nums = [4,1,2,1,2]
# dict={}
# set_nums = set(nums)
# for uni_value in set_nums:
#     dict[uni_value] = 0
# for i in range(len(nums)):
#     dict[nums[i]] += 1
# for key in dict.keys():
#     if dict[key] == 1:
#         print(key)

"""128. 最长连续序列"""
nums = [100,4,200,1,3,2]

# longest_streak = 0
# num_set = set(nums)
# for num in num_set:
#     if num - 1 not in num_set:
#         #python的set是hash表
#         current_num = num
#         current_streak = 1
#         while current_num + 1 in num_set:
#             current_num += 1
#             current_streak += 1
#         longest_streak = max(longest_streak, current_streak)
# print(longest_streak)

# hash_dict = dict()
# max_length = 0
# for num in nums:
#     if num not in hash_dict:
#         # 左边的数字刚刚已经告诉我了他的连续子序列长度，现在我需要拿出来和右边拼接了
#         # 没告诉过我就说明他也不知道，那就是0
#         len_left = hash_dict.get(num - 1, 0)
#         len_right = hash_dict.get(num + 1, 0)
#
#         cur_length = 1 + len_left + len_right
#         if cur_length > max_length:
#             max_length = cur_length
#
#         # 既然已经告诉旁边的人我的情况了，自然赋值是什么都不重要了
#         hash_dict[num] = 1
#         # 告诉左端点我的右边连续子序列长度是cur_length（或者说只有他左边的数字需要他）
#         hash_dict[num - len_left] = cur_length
#         # 告诉右端点我的左边连续子序列长度是cur_length
#         hash_dict[num + len_right] = cur_length
# print(max_length)

"""139. 单词拆分"""
'''背包问题'''
# s = "applepenapple"
# wordDict = ["apple", "pen"]
# dp = [False]*(len(s) + 1)
# dp[0] = True
# # 遍历背包
# for j in range(1, len(s) + 1):
#     # 遍历单词
#     for word in wordDict:
#         if j >= len(word):
#             dp[j] = dp[j] or (dp[j - len(word)] and word == s[j - len(word):j])
# print(dp[len(s)])

"""142. 环形链表 II"""
'''大多数对象可以用作字典的键，只要是可哈希的'''
# class ListNode:
#     def __init__(self, val, next=None):
#         self.val = val
#         self.next = next
#
# a= ListNode(2,None)
# b= ListNode(1, a)
# MAP = {}
# MAP[a] = 1
# MAP[b] = 0
# print(1)

"""146. LRU 缓存"""
'''
linkhashmap哈希链表满足此题的要求: hashmap哈希表存储，link双向链表来保证优先级顺序。
get 就是使用key-value，会提高该key-value的优先级
'''
# class LRUCache(collections.OrderedDict):
#
#     def __init__(self, capacity: int):
#         super().__init__()
#         self.capacity = capacity
#
#
#     def get(self, key: int) -> int:
#         if key not in self:
#             return -1
#         self.move_to_end(key)
#         return self[key]
#
#     def put(self, key: int, value: int) -> None:
#         if key in self:
#             self.move_to_end(key)
#         self[key] = value
#         if len(self) > self.capacity:
#             self.popitem(last=False)

# 作者：LeetCode-Solution
# 链接：https://leetcode-cn.com/problems/lru-cache/solution/lruhuan-cun-ji-zhi-by-leetcode-solution/
# 来源：力扣（LeetCode）
# 著作权归作者所有。商业转载请联系作者获得授权，非商业转载请注明出处。

"""152. 乘积最大子数组"""
'''没考虑到前面出现一个极大的负数，如[-2,3,-4]'''
# nums = [2,3,-2,4]
# res = 0
# dp = [0]*len(nums)#dp[i]的意义：包含当前元素的向列表首元素寻找到的最大连续子集乘积
# dp[0] = nums[0]
# for i in range(1,len(nums)):
#         dp[i] = max(nums[i], dp[i - 1]*nums[i])
# dp.sort()
# print(dp[-1])

'''解决方法是同时记录包含当前元素的向前最大值和向前最小值'''
# pre_min = nums[0]
# pre_max = nums[0]
# res = nums[0]
# for i in range(1,len(nums)):
#     cur_max = max(nums[i], nums[i]*pre_max, nums[i]*pre_min)
#     cur_min = min(nums[i], nums[i]*pre_max, nums[i]*pre_min)
#     res = max(res, cur_max)
#     pre_max = cur_max
#     pre_min = cur_min
# print(res)

"""200. 岛屿数量"""
'''类的实例中变量范围、地址问题。。。很迷'''
# class Solution:
#
#     # def dfs(self, block, x, y):
#     #     block[x][y] = 0
#     #     nr, nc = len(block), len(block[0])
#     #     for a, b in [(x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)]:
#     #         if 0 <= a < nr and 0 <= a < nc and block[a][b] == "1":
#     #             self.dfs(block, a, b)
#
#     def dfs(self, block, x, y):
#             if not 0 <= x < len(block) or not 0 <= y <len(block[0]) or block[x][y] == '0':
#                 return
#             block[x][y] = '0'
#             self.dfs(block, x - 1 , y)
#             self.dfs(block, x + 1, y)
#             self.dfs(block, x, y - 1)
#             self.dfs(block, x, y + 1)
#     def numIslands(self, grid):
#         # 图遍历，dfs，bfs
#         count = 0
#         for i in range(len(grid)):
#             for j in range(len(grid[0])):
#                 if grid[i][j] == '1':
#                     # self
#                     self.dfs(grid, i, j)
#                     # 为什么这里可以修改参数grid?
#                     print(grid)
#                     count += 1
#         print(count)
# a = Solution()
input = [["1","1","0","0","0"],["1","1","0","0","0"],["0","0","1","0","0"],["0","0","0","1","1"]]
#a.numIslands([["1","1","0","0","0"],["1","1","0","0","0"],["0","0","1","0","0"],["0","0","0","1","1"]])
#这里input, grid, block的地址都是一样的！用id()函数可以查看。

'''看起来不是类的性质，是列表的性质'''
# def dfs(block, x, y):
#     if not 0 <= x < len(block) or not 0 <= y < len(block[0]) or block[x][y] == '0':
#         return
#     block[x][y] = '0'
#     dfs(block, x - 1, y)
#     dfs(block, x + 1, y)
#     dfs(block, x, y - 1)
#     dfs(block, x, y + 1)
#
#
# def numIslands(grid):
#     # 图遍历，dfs，bfs
#     count = 0
#     for i in range(len(grid)):
#         for j in range(len(grid[0])):
#             if grid[i][j] == '1':
#                 # self
#                 dfs(grid, i, j)
#                 # 为什么这里可以修改参数grid?
#                 print(grid)
#                 count += 1
#     print(count)
#
# numIslands(input)

'''列表这类可变对象做参数传给函数的时候，传递的是真实的地址，任何对传入参数的修改都会影响到原来的变量'''
# def test_list(eglist):
#     eglist.append(1)
# a = [1,2,3,4]
# test_list(a)
# print(a)

# matrix = [["1","0","1","0","0"],["1","0","1","1","1"],["1","1","1","1","1"],["1","0","0","1","0"]]
# class solution:
#     def search_rectangle(self, block, x, y, square):
#         if not 0 <= x < len(block) or not 0 <= y < len(block[0]) or block[x][y] == '0':
#             return
#         square += 1
#         self.search_rectangle(block, x + 1, y, square)
#         self.search_rectangle(block, x, y + 1, square)
#         return square
#
#     def maximalSquare(self, grid):
#         #
#         count = 0
#         res = 0
#         for i in range(len(grid)):
#             for j in range(len(grid[0])):
#                 if grid[i][j] == '1':
#                     new_square = self.search_rectangle(grid, i, j, 0)
#                     res = max(res, new_square)
#         print(res)
# a=solution()
# a.maximalSquare(matrix)

#221. 最大正方形
# class Solution:
#     def maximalSquare(self, matrix):
#         #找最右最下
#         def findmaxside(x, y, oldmaxside):
#             right = y
#             down = x
#             k = down - i
#             while right <= len(matrix[0]) - 1 and right - y <= oldmaxside - k - 1:
#                 if matrix[x][right] == '1':
#                     right += 1
#                 else:
#                     break
#             while down <= len(matrix) - 1 and down - x <= oldmaxside - k - 1:
#                 if matrix[down][y] == '1':
#                     down += 1
#                 else:
#                     break
#             maxside = min(down - x, right - y)
#             return maxside
#
#         res = 0
#         new_res = 0
#         maxside = min(len(matrix), len(matrix[0]))
#         for i in range(len(matrix)):
#             for j in range(len(matrix[0])):
#                 if matrix[i][j] == '1':
#                     pre_maxside = min(len(matrix) - i, len(matrix[0]) - j)
#                     maxside = findmaxside(i, j, pre_maxside)
#                     new_res = 1
#                     for k in range(1, maxside):
#                         if matrix[i + k][j + k] == '1':
#                             new_res = (1 + k) ** 2
#                         new_maxside = findmaxside(i + k, j + k, maxside)
#                         if maxside - new_maxside != k and new_maxside == 1:
#                             break
#                 res = max(res, new_res)
#         print(res)
# #matrix = [["1", "0", "1", "0", "0"], ["1", "0", "1", "1", "1"], ["1", "1", "1", "1", "1"], ["1", "0", "0", "1", "0"]]
# #matrix = [["0","0","0","1"],["1","1","0","1"],["1","1","1","1"],["0","1","1","1"],["0","1","1","1"]]
# # matrix = [["1","1","1","1","0"],["1","1","1","1","0"],["1","1","1","1","1"],["1","1","1","1","1"],["0","0","1","1","1"]]
# matrix =  [["0","0","1","0"],["1","1","1","1"],["1","1","1","1"],["1","1","1","0"],["1","1","0","0"],["1","1","1","1"],["1","1","1","0"]]
# #matrix = [["1","1","1","1","1","1","1","1","1","1","1","1","1","1","0","1","0","0","1","1","1","1","1","1","1","1","0","0","1","1","1","0","1","1","1","1","1","1","1","1"],["1","1","1","1","0","1","1","0","1","1","1","1","1","1","1","1","1","0","1","1","0","1","1","1","1","1","0","1","1","1","1","1","1","1","1","1","1","1","1","1"],["0","1","1","1","1","0","1","0","1","1","1","1","1","1","0","1","1","0","1","1","0","1","1","0","1","1","1","1","1","1","1","1","1","1","1","1","1","1","1","1"],["0","1","0","1","1","0","1","0","1","1","1","1","1","1","1","1","1","1","1","1","0","1","1","1","1","1","1","1","0","1","0","1","1","0","1","1","1","1","1","1"],["1","1","1","1","1","1","1","1","1","1","1","1","0","1","1","1","1","1","0","1","1","0","0","1","0","1","1","1","1","1","1","1","1","1","1","1","1","1","1","1"],["1","1","1","1","1","1","1","1","1","0","1","1","0","1","0","1","1","1","1","1","1","1","1","1","1","1","0","1","0","1","1","1","1","1","1","0","1","1","1","1"],["0","1","1","0","1","1","0","1","0","1","1","1","0","0","1","1","1","1","1","1","1","1","1","1","1","1","1","1","1","1","0","1","1","1","1","1","0","1","0","1"],["0","1","1","1","1","1","1","1","1","1","1","1","1","1","0","0","1","1","1","1","1","1","1","0","0","1","1","0","0","1","1","0","1","1","0","1","0","1","0","1"],["1","1","1","1","0","1","1","1","1","0","1","1","1","1","1","1","1","1","1","0","1","1","0","1","1","0","1","1","1","1","0","1","0","1","1","0","1","0","1","1"],["1","1","1","1","1","1","1","1","1","0","1","1","1","1","1","1","1","1","1","1","0","1","1","0","1","1","0","1","1","1","0","1","1","1","1","0","1","1","1","1"],["1","1","1","0","1","1","0","0","1","1","1","1","1","1","1","1","1","1","1","1","0","0","1","1","1","1","1","1","1","1","1","1","1","1","1","1","1","1","1","1"],["1","0","1","1","1","1","1","1","1","0","1","1","1","1","0","1","1","1","1","0","0","1","1","1","0","1","1","1","1","1","1","1","1","1","1","1","1","1","1","1"],["0","1","1","0","1","1","1","1","1","1","1","1","1","1","1","1","1","0","1","1","1","1","1","1","0","1","1","1","0","1","1","1","1","1","0","1","1","1","1","1"],["1","1","1","1","1","1","1","1","1","1","1","1","1","1","1","1","0","1","1","1","1","1","1","1","1","1","1","1","1","1","0","1","1","1","1","1","1","0","1","1"],["1","1","1","1","1","0","0","1","1","1","1","1","1","1","1","0","1","0","1","1","0","0","1","1","1","1","1","1","1","1","1","1","1","1","1","0","1","1","1","1"],["1","1","1","1","1","0","1","1","1","1","1","1","1","1","1","0","1","1","1","1","1","1","0","1","1","1","1","1","1","1","1","1","1","1","0","1","1","1","1","1"],["1","1","1","1","1","0","1","1","1","1","1","1","1","1","1","1","1","1","1","1","1","1","0","1","1","0","0","1","1","1","1","1","1","0","0","1","1","1","1","1"],["1","1","1","1","1","1","0","1","1","1","1","1","1","1","1","1","0","1","1","1","1","1","1","1","1","1","1","1","1","1","0","1","1","1","1","1","0","1","1","1"],["1","0","1","1","1","1","1","1","1","1","1","1","1","1","1","1","0","1","0","1","1","1","1","1","0","0","1","0","1","1","1","1","1","0","1","1","1","1","1","1"],["1","1","1","1","1","1","0","0","1","1","1","1","1","1","1","1","1","1","1","1","1","0","1","1","1","1","1","1","1","1","1","1","1","1","1","1","1","0","1","1"],["1","1","1","1","1","1","1","1","1","1","1","1","1","1","1","1","1","1","0","1","1","1","0","1","1","1","1","1","0","1","1","1","1","1","0","1","1","0","1","1"],["1","1","0","0","0","1","1","0","1","1","1","1","1","1","1","1","1","1","1","1","1","1","0","1","1","1","1","1","0","1","1","1","1","1","1","1","1","1","1","1"],["1","1","1","1","1","0","1","0","1","1","1","1","1","1","1","1","1","1","1","1","0","1","1","0","0","1","0","1","1","1","0","0","1","1","1","1","1","1","1","1"],["1","1","1","0","0","1","0","1","1","1","1","1","1","1","1","1","1","1","1","0","1","1","0","1","1","1","1","0","1","1","1","1","0","1","1","1","1","1","0","1"],["1","1","1","1","1","1","1","1","1","1","1","1","1","1","0","1","1","0","1","1","1","1","1","1","1","1","1","1","1","0","1","1","1","1","1","1","1","1","1","1"],["1","1","1","1","1","1","1","0","1","1","1","1","1","1","0","1","1","1","1","0","1","1","1","1","1","1","1","1","1","1","1","1","1","0","1","1","1","1","1","1"],["1","1","1","0","0","1","1","1","1","1","1","1","1","1","1","0","1","1","1","0","1","1","1","1","1","1","1","1","1","1","0","1","1","1","1","1","0","1","1","1"],["1","1","1","1","1","1","1","1","1","1","1","1","1","1","0","0","1","1","1","1","1","1","0","1","0","1","1","1","1","1","1","1","1","1","1","1","0","1","1","1"],["1","1","1","1","1","1","1","1","1","1","1","1","1","1","1","1","1","1","1","1","1","1","0","0","0","1","1","1","1","1","1","1","1","1","0","1","1","1","0","1"],["1","1","1","1","1","1","0","1","1","1","1","1","1","1","1","1","1","1","1","1","1","1","1","1","1","1","0","1","1","0","0","1","1","1","0","1","1","0","1","1"],["1","1","1","1","0","1","1","0","1","1","1","1","1","1","0","1","1","0","1","1","0","1","1","1","1","1","1","0","1","1","1","1","1","1","1","0","1","1","1","1"],["1","1","1","1","1","1","1","1","1","1","0","1","1","1","1","1","1","1","1","1","1","1","1","1","1","1","1","1","1","1","1","1","1","1","1","1","1","1","1","1"],["1","1","1","1","1","1","1","1","1","1","1","1","1","1","1","1","1","1","1","1","1","1","1","1","1","1","1","1","1","1","1","1","1","1","1","1","1","1","1","1"],["1","1","0","0","0","1","1","1","1","1","1","1","1","1","1","0","1","1","1","1","1","1","1","1","1","1","1","1","1","0","1","1","1","1","1","1","1","0","1","1"],["1","1","1","1","1","0","1","1","1","1","1","1","1","1","0","1","1","1","1","0","1","0","1","1","1","1","1","1","1","1","1","1","1","1","1","1","1","1","1","1"],["0","1","1","1","1","1","1","1","1","1","1","1","0","0","1","1","1","1","1","1","1","1","1","1","0","1","0","1","0","1","1","0","1","1","1","1","1","1","1","1"],["1","0","1","1","0","1","1","1","1","1","1","0","1","1","1","1","1","1","1","1","1","1","1","0","1","1","1","1","1","1","1","1","1","1","1","1","0","0","1","1"],["1","0","1","0","1","1","1","1","1","1","1","1","1","1","1","1","1","1","1","1","1","1","1","1","1","1","1","1","1","1","1","1","1","0","0","1","1","1","1","1"],["0","1","1","1","1","0","1","1","1","1","0","1","1","1","1","1","1","1","1","1","1","1","1","1","1","1","1","1","1","1","0","1","1","1","1","1","1","1","1","1"],["0","1","1","1","1","1","1","0","1","1","1","1","1","1","1","1","0","1","1","1","0","1","1","1","1","0","1","1","1","0","1","1","1","1","1","1","1","1","1","1"],["0","1","1","1","1","1","1","1","1","1","1","1","0","1","0","1","1","1","1","0","1","1","1","1","1","1","0","1","0","1","1","0","0","1","1","1","1","0","1","1"],["1","1","1","1","1","1","1","1","1","1","1","1","1","0","1","1","1","1","0","1","1","1","1","1","1","1","0","1","1","1","1","1","1","1","1","0","1","1","1","0"],["1","1","1","1","1","0","1","1","1","1","1","1","1","1","0","0","1","1","1","1","1","1","1","1","1","1","1","0","1","1","1","1","1","1","0","0","1","1","1","1"],["1","1","0","1","1","0","1","1","1","1","1","1","0","1","0","1","1","1","1","1","0","1","1","1","1","1","1","1","1","0","0","1","1","1","0","1","0","1","0","0"],["0","1","1","0","1","1","1","1","1","1","1","0","0","1","1","1","1","1","0","0","1","0","1","1","1","1","1","0","1","1","1","0","1","1","0","1","1","1","0","1"]]
# a = Solution()
# a.maximalSquare(matrix)

# matrix = [["1","0","1","0","0"],["1","0","1","1","1"],["1","1","1","1","1"],["1","0","0","1","0"]]


# class Solution:
#     def maximalSquare(self, matrix):
#         if len(matrix) == 0 or len(matrix[0]) == 0:
#             return 0
#         maxside = 0
#         maxside2=0
#         column = len(matrix[0])
#         row = len(matrix)
#         dp2 = [[0] * column] * row
#         dp = [[0] * column for _ in range(row)]
#         '''
#         这两种定义出来的二维数组初值一样，但操作dp[i][j]==1上不一样，非常奇怪
#         '''
#         # print(dp)
#         # print(dp2)
#         # print(dp==dp2)
#         for i in range(row):
#             for j in range(column):
#                 if matrix[i][j] == '1':
#                     if i == 0 or j == 0:
#                         dp[i][j] = 1
#                         dp2[i][j] = 1
#                     else:
#                         dp[i][j] = min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1]) + 1
#                         dp2[i][j] = min(dp2[i - 1][j], dp2[i][j - 1], dp2[i - 1][j - 1]) + 1
#                     maxside = max(maxside, dp[i][j])
#                     maxside2 = max(maxside2, dp2[i][j])
#         print(maxside**2,'good')

# a=Solution()
# a.maximalSquare(matrix)

#300. 最长递增子序列
# class Solution:
#     def lengthOfLIS(self, nums):
#         dp = [1] * len(nums)
#         for i in range(len(nums)):
#             temp = 0
#             for j in range(0,i):
#                 if nums[i] > nums[j]:
#                     temp = dp[j] + 1
#                     dp[i] = max(temp, dp[i])
#         print(max(dp))
# a=Solution()
# a.lengthOfLIS([2,4,6,7,2,4,5,3,5,7,43,5,67,9,89,5,8])            


#