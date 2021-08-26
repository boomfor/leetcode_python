# leetcode刷题记录

​                                                                                                                                                                                                                            **author:   ​曹志勇:six:**

编者注：点击题目名称即可进入leetcode刷题，建议先自己独立思考完成。独立思考5-10min；若还没有思路可查看题解。题解查看完成之后自己独立编写代码完成，避免出现眼睛觉得会了，手还不会的情况。

难度星级标识，:smile:越多表示题目越简单





## day1:  [反转字符串中的元音字母](https://leetcode-cn.com/problems/reverse-vowels-of-a-string/)                          

​                                                                                                                                                                                                                            **难度：**:smile::smile::smile::smile:



####    **编写一个函数，以字符串作为输入，反转该字符串中的元音字母。**

**示例 1：**

```
输入："hello"
输出："holle"
```

**示例 2：**

```
输入："leetcode"
输出："leotcede"
```



**解题思路:**

**双指针：**

左指针从左往右遍历碰到元音字母停下，右指针从右往左遍历碰到元音字母停下。左右指针指向的元音字母交换位置即视为翻转。

**PS:**  python字符串为不可变类型，注意不能直接交换位置



```python
class Solution:
    def reverseVowels(self, s: str) -> str:
        initial_digit = ['a','o','e','i','u','A','E','I','O','U']       #元音字母列表
        s = list(s)   #先转换为列表，便于后边交换位置
        left = 0              
        right = len(s) - 1
        while left < right:
            if s[left] not in initial_digit:
                left += 1
                continue
            if s[right] not in initial_digit:
                right -= 1
                continue
            s[left], s[right] = s[right], s[left] 
            left += 1              #注意交换完位置之后左右指针的移动
            right -= 1
        return ''.join(s)       #列表转字符串

```





## day2: [盛最多水的容器](https://leetcode-cn.com/problems/container-with-most-water/)

​                                                                                                                                                                                                                **难度：**:smile::smile::smile:

####   

####       **给你 n 个非负整数 a1，a2，...，an，每个数代表坐标中的一个点 (i, ai) 。在坐标内画 n 条垂直线，垂直线 i 的两个端点分别为 (i, ai) 和 (i, 0) 。找出其中的两条线，使得它们与 x 轴共同构成的容器可以容纳最多的水。**

####       **说明：你不能倾斜容器。**

**示例 1：**

![img](https://aliyun-lc-upload.oss-cn-hangzhou.aliyuncs.com/aliyun-lc-upload/uploads/2018/07/25/question_11.jpg)

```
输入：[1,8,6,2,5,4,8,3,7]
输出：49 
解释：图中垂直线代表输入数组 [1,8,6,2,5,4,8,3,7]。在此情况下，容器能够容纳水（表示为蓝色部分）的最大值为 49。
```

**示例 2：**

```
输入：height = [1,1]
输出：1
```

**示例 3：**

```
输入：height = [4,3,2,1,4]
输出：16
```

**示例 4：**

```
输入：height = [1,2,1]
输出：2
```

**解题思路：**

**双指针：**
$$
围成的面积  squre =  （right-left）* min(height[left],height[right])

$$

$$
当right-left的值变小的时候，我们需要min(height[left],height[right])的值变大，才能使得总体围成的squre的面积尽量大
$$

$$
即当height[left] < height[right]的时候，我们需要height[left]的值尽量大。向右移动左指针
$$

$$
当height[right] <= height[left]的时候，我们需要height[right]的值尽量大。向左移动右指针
$$

```python
题解如下：
class Solution:
    def maxArea(self, height: List[int]) -> int:
        left = 0
        right = len(height) - 1
        max_area = 0
        while left < right:
            if height[left] < height[right]:
                squere = (right-left) * height[left]
                left += 1
            else:
                squere = (right-left) * height[right]
                right -= 1
            max_area = max(squere,max_area)
        return max_area

```





## day3: [反转字符串 II](https://leetcode-cn.com/problems/reverse-string-ii/)

​                                                                                                                                                                                                                         **难度：**:smile::smile::smile::smile:

####    

####     给定一个字符串 s 和一个整数 k，从字符串开头算起，每 2k 个字符反转前 k 个字符。

####     如果剩余字符少于 k 个，则将剩余字符全部反转。

####     如果剩余字符小于 2k 但大于或等于 k 个，则反转前 k 个字符，其余字符保持原样。



**示例 1：**

```
输入：s = "abcdefg", k = 2
输出："bacdfeg"
```

**示例 2：**

```
输入：s = "abcd", k = 2
输出："bacd"
```

**解题思路：**

​    本题主要考察python基本语法的灵活应用，可以利用for循环来控制k的间隔。

   **PS:**  python字符串为不可变类型，注意不能直接交换位置

​          合理使用reversed函数或切片。

```python
class Solution:
    def reverseStr(self, s: str, k: int) -> str:
        s = list(s)
        for i in range(0,len(s),2*k):
            s[i:i+k] = reversed(s[i:i+k])
            #s[i:i+k] = s[i:i+k][::-1]
        return ''.join(s)
```





## day4: [公平的糖果棒交换](https://leetcode-cn.com/problems/fair-candy-swap/)

​                                                                                                                                                                                                              **难度：**:smile::smile::smile:

####     

####     爱丽丝和鲍勃有不同大小的糖果棒：A[i] 是爱丽丝拥有的第 i 根糖果棒的大小，B[j] 是鲍勃拥有的第 j 根糖果棒的大小。

####     因为他们是朋友，所以他们想交换一根糖果棒，这样交换后，他们都有相同的糖果总量。（一个人拥有的糖果总量是他们拥有  的糖果棒大小的总和。）

####     返回一个整数数组 ans，其中 ans[0] 是爱丽丝必须交换的糖果棒的大小，ans[1] 是 Bob 必须交换的糖果棒的大小。

####     如果有多个答案，你可以返回其中任何一个。保证答案存在。



**示例 1：**

```
输入：A = [1,1], B = [2,2]
输出：[1,2]
```

**示例 2：**

```
输入：A = [1,2], B = [2,3]
输出：[1,2]
```

**示例 3：**

```
输入：A = [2], B = [1,3]
输出：[2,3]
```

**示例 4：**

```
输入：A = [1,2,5], B = [2,4]
输出：[5,4]
```

**解题思路：**
$$
设alice给bob-- x颗糖，bob给alice-- y 颗糖；则有以下公式：
$$

$$
sum(a)-x+y = sum(b)-y+x
$$

$$
转换形式得: y-x=(sum(b)-sum(a))/2
$$

$$
因此，我们只需遍历alice中的糖果，只要(sum(b)-sum(a))/2+x 在bob已有的糖果大小中即可交换
$$

```python
写成代码如下：
class Solution:
    def fairCandySwap(self, aliceSizes: List[int], bobSizes: List[int]) -> List[int]:
        target = (sum(bobSizes) - sum(aliceSizes))/2
        for x in aliceSizes:
            if target+x in bobSizes:
                return [x,int(target+x)]
```





## day5: [在排序数组中查找元素的第一个和最后一个位置](https://leetcode-cn.com/problems/find-first-and-last-position-of-element-in-sorted-array/)

​                                                                                                                                                                                                                    **难度：**:smile::smile::smile:

####     

####     给定一个按照升序排列的整数数组 nums，和一个目标值 target。找出给定目标值在数组中的开始位置和结束位置。

####     如果数组中不存在目标值 target，返回 [-1, -1]。

####     进阶：

####     你可以设计并实现时间复杂度为 O(log n) 的算法解决此问题吗？



**示例 1：**

```
输入：nums = [5,7,7,8,8,10], target = 8
输出：[3,4]
```

**示例 2：**

```
输入：nums = [5,7,7,8,8,10], target = 6
输出：[-1,-1]
```

**示例 3：**

```
输入：nums = [], target = 0
输出：[-1,-1]
```

**解题思路：**

时间复杂度O(log n)，首先想到即是二分查找。但是与以往的二分查找不同的是，这次不是要查找某个值的index，而是取值范围。因此我们需要对原有的二分查找算法进行修改。当找到第一个和target相等的值的时候，分别向左向右再接着查找，找到该值的左边界和右边界。写成代码如下所示：

```python
class Solution:
    def searchRange(self, nums: List[int], target: int) -> List[int]:
        def search(left,right,boundary):
            ans = -1
            while left < right:
                mid = (right + left) // 2
                if nums[mid] == target:
                    ans = mid
                    if boundary:                 #找到之后查找左边界
                        right = mid
                    else:                        #找到之后查找右边界
                        left = mid + 1
                elif nums[mid] > target:
                    right = mid
                else:
                    left = mid + 1
            return ans


        left = 0
        right = len(nums)
        ans_left = search(left,right,True)
        ans_right = search(left,right, False) 
        return [ans_left,ans_right]


```



## day6:  [获取生成数组中的最大值](https://leetcode-cn.com/problems/get-maximum-in-generated-array/)

​                                                                                                                                                                                                                                      **难度：**:smile::smile::smile::smile:



####     给你一个整数 n 。按下述规则生成一个长度为 n + 1 的数组 nums ：

####     nums[0] = 0

####     nums[1] = 1

####     当 2 <= 2 * i <= n 时，nums[2 * i] = nums[i]

####     当 2 <= 2 * i + 1 <= n 时，nums[2 * i + 1] = nums[i] + nums[i + 1]

####     返回生成数组 nums 中的 最大值。



**示例 1：**

```
输入：n = 7
输出：3
解释：根据规则：
  nums[0] = 0
  nums[1] = 1
  nums[(1 * 2) = 2] = nums[1] = 1
  nums[(1 * 2) + 1 = 3] = nums[1] + nums[2] = 1 + 1 = 2
  nums[(2 * 2) = 4] = nums[2] = 1
  nums[(2 * 2) + 1 = 5] = nums[2] + nums[3] = 1 + 2 = 3
  nums[(3 * 2) = 6] = nums[3] = 2
  nums[(3 * 2) + 1 = 7] = nums[3] + nums[4] = 2 + 1 = 3
因此，nums = [0,1,1,2,1,3,2,3]，最大值 3
```

**示例 2：**

```
输入：n = 2
输出：1
解释：根据规则，nums[0]、nums[1] 和 nums[2] 之中的最大值是 1
```

**示例 3：**

```
输入：n = 3
输出：2
解释：根据规则，nums[0]、nums[1]、nums[2] 和 nums[3] 之中的最大值是 2
```

**解题思路：**
$$

本题可使用递归和递归两种方式，与斐波那契数列相类似，找到递推关系如下：
$$

$$
如果n为偶数:nums[n] = nums[n//2]
$$

$$
如果n为奇数: nums[n] = nums[n//2] + nums[n//2 + 1]
$$

```python
class Solution:
    def getMaximumGenerated(self, n: int) -> int:
        rs = [0, 1]
        if n == 0:
            return 0 
        for i in range(2, n+1):
            if i % 2 == 0:
                rs.append(rs[i//2])
            else:
                rs.append(rs[i//2]+rs[i//2 + 1])
        return max(rs)
```





## day7: [岛屿数量](https://leetcode-cn.com/problems/number-of-islands/)

​                                                                                                                                                                                                                         **难度：**:smile::smile::smile:

####     

####     给你一个由 '1'（陆地）和 '0'（水）组成的的二维网格，请你计算网格中岛屿的数量。

####     岛屿总是被水包围，并且每座岛屿只能由水平方向和/或竖直方向上相邻的陆地连接形成。

####     此外，你可以假设该网格的四条边均被水包围。

 

**示例 1：**

```
输入：grid = [
  ["1","1","1","1","0"],
  ["1","1","0","1","0"],
  ["1","1","0","0","0"],
  ["0","0","0","0","0"]
]
输出：1
```

**示例 2：**

```
输入：grid = [
  ["1","1","0","0","0"],
  ["1","1","0","0","0"],
  ["0","0","1","0","0"],
  ["0","0","0","1","1"]
]
输出：3
```

**解题思路：**

这是一个典型的图搜索问题，可以以水平和竖直方向相连。即在某个点能以上下左右四个方向。为了防止重复计算，计算完该点之后直接把该点的值赋值为0即可（之前图的搜索一般是用一个visited数组来标识该点是否访问，本题连成片的岛屿不用重复计算，直接把数组值赋值0）。详细代码如下：

```python
class Solution:
    def numIslands(self, grid: List[List[str]]) -> int:
        directs = [(-1,0),(1,0),(0,-1),(0,1)]
        def dfs(x,y):
            grid[x][y] = "0"
            for direct in directs:
                x = x + direct[0]
                y = y + direct[1]
                if -1 < x < len(grid) and -1 < y < len(grid[0]) and grid[x][y] == "1":
                    dfs(x,y)
                x = x - direct[0]
                y = y - direct[1]

                
        island_num = 0
        for i in range(len(grid)):
            for j in range(len(grid[0])):
                if grid[i][j] == "1":
                    island_num += 1
                    dfs(i,j)
        return island_num

```



## day8: [丢失的数字](https://leetcode-cn.com/problems/missing-number/)

​                                                                                                                                                                                                                                 **难度：**:smile::smile::smile::smile:

####     

####     给定一个包含 [0, n] 中 n 个数的数组 nums ，找出 [0, n] 这个范围内没有出现在数组中的那个数。

####     进阶：

####     你能否实现线性时间复杂度、仅使用额外常数空间的算法解决此问题?



**示例 1：**

```
输入：nums = [3,0,1]
输出：2
解释：n = 3，因为有 3 个数字，所以所有的数字都在范围 [0,3] 内。2 是丢失的数字，因为它没有出现在 nums 中。
```

**示例 2：**

```
输入：nums = [0,1]
输出：2
解释：n = 2，因为有 2 个数字，所以所有的数字都在范围 [0,2] 内。2 是丢失的数字，因为它没有出现在 nums 中。
```

**示例 3：**

```
输入：nums = [9,6,4,2,3,5,7,0,1]
输出：8
解释：n = 9，因为有 9 个数字，所以所有的数字都在范围 [0,9] 内。8 是丢失的数字，因为它没有出现在 nums 中。
```

**示例 4：**

```
输入：nums = [0]
输出：1
解释：n = 1，因为有 1 个数字，所以所有的数字都在范围 [0,1] 内。1 是丢失的数字，因为它没有出现在 nums 中。
```

**解题思路:**

​    先对数组进行排序，排序之后根据索引遍历，遍历发现的索引和值不相等即为缺失的值，注意示例4的特殊情况出现。

```python
class Solution:
    def missingNumber(self, nums: List[int]) -> int:
        nums.sort()
        for index, i in enumerate(nums):
            if index != i:
                return index
        else:
            return index + 1
```





## day9: [球会落何处](https://leetcode-cn.com/problems/where-will-the-ball-fall/)

​                                                                                                                                                                                                                               **难度：**:smile::smile::smile:



####      用一个大小为 m x n 的二维网格 grid 表示一个箱子。你有 n 颗球。箱子的顶部和底部都是开着的。

####     箱子中的每个单元格都有一个对角线挡板，跨过单元格的两个角，可以将球导向左侧或者右侧。

####     将球导向右侧的挡板跨过左上角和右下角，在网格中用 1 表示。

####     将球导向左侧的挡板跨过右上角和左下角，在网格中用 -1 表示。

####     在箱子每一列的顶端各放一颗球。每颗球都可能卡在箱子里或从底部掉出来。如果球恰好卡在两块挡板之间的 "V" 形图案，或者被一块挡导向到箱子的任意一侧边上，就会卡住。

####     返回一个大小为 n 的数组 answer ，其中 answer[i] 是球放在顶部的第 i 列后从底部掉出来的那一列对应的下标，如果球卡在盒子里，则返回 -1 。

 

**示例 1：**

![img](https://assets.leetcode-cn.com/aliyun-lc-upload/uploads/2020/12/26/ball.jpg)

```
输入：grid = [[1,1,1,-1,-1],[1,1,1,-1,-1],[-1,-1,-1,1,1],[1,1,1,1,-1],[-1,-1,-1,-1,-1]]
输出：[1,-1,-1,-1,-1]
解释：示例如图：
b0 球开始放在第 0 列上，最终从箱子底部第 1 列掉出。
b1 球开始放在第 1 列上，会卡在第 2、3 列和第 1 行之间的 "V" 形里。
b2 球开始放在第 2 列上，会卡在第 2、3 列和第 0 行之间的 "V" 形里。
b3 球开始放在第 3 列上，会卡在第 2、3 列和第 0 行之间的 "V" 形里。
b4 球开始放在第 4 列上，会卡在第 2、3 列和第 1 行之间的 "V" 形里。
```

**示例 2：**

```
输入：grid = [[-1]]
输出：[-1]
解释：球被卡在箱子左侧边上。
```

**示例 3：**

```
输入：grid = [[1,1,1,1,1,1],[-1,-1,-1,-1,-1,-1],[1,1,1,1,1,1],[-1,-1,-1,-1,-1,-1]]
输出：[0,1,2,3,4,-1]
```

**解题思路：**

只有当数组行相邻的两个数相同的时候，球才会往下一层移动。否则则会形成死叉，求会被卡死。当相邻两个数为1时，球向右侧滚动。当相邻两个数为-1时，球向左侧滚动。

需注意左右两侧挡板，若是相邻的数不在行的范围内，即可视为球在该层被卡住了。代码如下：

```python
class Solution:
    def findBall(self, grid: List[List[int]]) -> List[int]:
        rs = []
        column_rses = [i for i in range(len(grid[0]))]
        for column_rs in column_rses:
            for i in range(len(grid)):
                if grid[i][column_rs] == 1:
                    if column_rs + 1 < len(grid[0]):
                        if grid[i][column_rs] == grid[i][column_rs+1]:
                            column_rs += 1
                        else:
                            rs.append(-1)
                            break
                    else:
                        rs.append(-1)
                        break
                else:
                    if -1 < column_rs - 1 < len(grid[0]):
                        if grid[i][column_rs] == grid[i][column_rs-1]:
                            column_rs -= 1
                        else:
                            rs.append(-1)
                            break
                    else:
                        rs.append(-1)
                        break
            else:
                rs.append(column_rs)
        return rs
```





## day10: [ K 站中转内最便宜的航班](https://leetcode-cn.com/problems/cheapest-flights-within-k-stops/)

​                                                                                                                                                                                                                                   **难度：**:smile::smile:



####      有 n 个城市通过一些航班连接。给你一个数组 flights ，其中 flights[i] = [from-i, to-i, price-i] ，表示该航班都从城市 from-i 开始，以价格 to-i 抵达 price-i。

####     现在给定所有的城市和航班，以及出发城市 src 和目的地 dst，你的任务是找到出一条最多经过 k 站中转的路线，使得从 src 到 dst 的 价格最便宜 ，并返回该价格。 如果不存在这样的路线，则输出 -1。

 



![img](https://s3-lc-upload.s3.amazonaws.com/uploads/2018/02/16/995.png)

**示例 1：**

```
输入: 
n = 3, edges = [[0,1,100],[1,2,100],[0,2,500]]
src = 0, dst = 2, k = 1
输出: 200
解释: 
从城市 0 到城市 2 在 1 站中转以内的最便宜价格是 200，如图中红色所示。
```

**示例 2：**

```
输入: 
n = 3, edges = [[0,1,100],[1,2,100],[0,2,500]]
src = 0, dst = 2, k = 0
输出: 500
解释: 
从城市 0 到城市 2 在 0 站中转以内的最便宜价格是 500，如图中蓝色所示。
```

**解题思路：**

   这是一道典型的图搜索问题。关于该题比较常见的思路是深度优先搜索或广度优先搜索。需要先把图转化为邻接表的形式。但是本题直接用深度优先搜索会超时。因此需要转换一下思路。

利用动态规划来解决，我们用一张二维表格来表示。行表示需要经过几个城市，列表示中转城市的名称，填的值即为花费的钱。拿示例1为例，我们可以构建如下表格：

第一行  经过0个城市---起点为0，从0出发可直接到0。不用花钱。其余的都需要经过1个城市--花费的钱为无穷大。

第二行  经过1个城市--起点为0，从上面的图可以看出，从起点经过一个城市可以到1和2花费分别为100和500。

第三行  经过2个城市--起点很显然只能从第二行找，飞到的下一个城市才能经过两个城市。从1可以到2共花费100+100共200元。其余的都无法到达花费为无穷大

剩余多的行数以此类推......

| k\dst | 0    | 1    | 2       |
| ----- | ---- | ---- | ------- |
| 0     | 0    | inf  | inf     |
| 1     | inf  | 100  | 500     |
| 2     | inf  | inf  | 100+100 |

上述思路即为动态规划过程

```python
动态规划:
class Solution:
    def findCheapestPrice(self, n: int, flights: List[List[int]], src: int, dst: int, k: int) -> int:
        dp = [[80808080808080 for i in range(n)] for j in range(k+2)]
        dp[0][src] = 0
        for row in range(1, k+2):
            for from_, to, price in flights:
                dp[row][to] = min(dp[row][to], dp[row-1][from_]+price)
        return -1 if min(dp[row][dst] for row in range(k+2)) == 80808080808080 else min(dp[row][dst] for row in range(k+2))



深度优先:
    class Solution:
    def findCheapestPrice(self, n: int, flights: List[List[int]], src: int, dst: int, k: int) -> int:
        flight = {}
        for edge in flights:
            if edge[0] not in flight:
                flight[edge[0]] = {edge[1]: edge[2]}
            else:
                if edge[1] not in flight[edge[0]]:
                    flight[edge[0]][edge[1]] = edge[2]
        visited = [False for i in range(n)]
        visited[src] = True

        min_price = 90000000


        def dfs(current_src, current_path, current_price, visited):
            nonlocal min_price
            if current_price > min_price:
                return
            if k + 1 > current_path - 1:
                if current_src == dst:
                    if current_price < min_price:
                        min_price = current_price
                        return
                else:
                    if current_src in flight.keys():
                        for dst_now, dst_price in flight[current_src].items():
                            if not visited[dst_now]:
                                visited[dst_now] = True
                                dfs(dst_now, current_path + 1, current_price + dst_price, visited)
                                visited[dst_now] = False
                    else:
                        return
            else:
                return

        dfs(src,0,0,visited)
        if min_price == 90000000:
            return -1
        return min_price
```





## day11:   [所有可能的路径](https://leetcode-cn.com/problems/all-paths-from-source-to-target/)

​                                                                                                                                                                                                                       **难度：**:smile::smile::smile:



####     给你一个有 n 个节点的 有向无环图（DAG），请你找出所有从节点 0 到节点 n-1 的路径并输出（不要求按特定顺序）

####     二维数组的第 i 个数组中的单元都表示有向图中 i 号节点所能到达的下一些节点，空就是没有下一个结点了。

####     译者注：有向图是有方向的，即规定了 a→b 你就不能从 b→a 。

 

**示例 1：**

![img](https://assets.leetcode.com/uploads/2020/09/28/all_1.jpg)

```
输入：graph = [[1,2],[3],[3],[]]
输出：[[0,1,3],[0,2,3]]
解释：有两条路径 0 -> 1 -> 3 和 0 -> 2 -> 3
```

**示例 2：**

![img](https://assets.leetcode.com/uploads/2020/09/28/all_2.jpg)

```
输入：graph = [[4,3,1],[3,2,4],[3],[4],[]]
输出：[[0,4],[0,3,4],[0,1,3,4],[0,1,2,3,4],[0,1,4]]
```

**示例 3：**

```
输入：graph = [[1],[]]
输出：[[0,1]]
```

**示例 4：**

```
输入：graph = [[1,2,3],[2],[3],[]]
输出：[[0,1,2,3],[0,2,3],[0,3]]
```

**示例 5：**

```
输入：graph = [[1,3],[2],[3],[]]
输出：[[0,1,2,3],[0,3]]
```

**解题思路:**

这是一道典型的图的搜索问题，同[ K 站中转内最便宜的航班](https://leetcode-cn.com/problems/cheapest-flights-within-k-stops/)类似，我们采用dfs的方式进行图的遍历。代码如下

```python
class Solution:
    def allPathsSourceTarget(self, graph: List[List[int]]) -> List[List[int]]:
        visited = [False for i in range(len(graph))]
        visited[0] = True
        def dfs(start,current_rs,visited):
            if start == len(graph) - 1:
                if current_rs[0] != 0:
                    current_rs.insert(0,0)
                rs.append(current_rs[:])
                return 
            for next_ in graph[start]:
                if not visited[next_]:
                    visited[next_] = True
                    current_rs.append(next_)
                    dfs(next_, current_rs,visited)
                    current_rs.pop()
                    visited[next_] = False

        rs = []
        current_rs = [] 
        dfs(0,current_rs,visited)
        return rs
```





## day12:  [课程表](https://leetcode-cn.com/problems/course-schedule/)

​                                                                                                                                                                                                                            **难度：**:smile::smile::smile:



####     你这个学期必须选修 numCourses 门课程，记为 0 到 numCourses - 1 。

####     在选修某些课程之前需要一些先修课程。 先修课程按数组 prerequisites 给出，其中 prerequisites[i] = [ai, bi] ，表示如果要学习课程 ai 则 必须 先学习课程  bi 。

####     例如，先修课程对 [0, 1] 表示：想要学习课程 0 ，你需要先完成课程 1 。

####     请你判断是否可能完成所有课程的学习？如果可以，返回 true ；否则，返回 false 。

 

**示例 1：**

```
输入：numCourses = 2, prerequisites = [[1,0]]
输出：true
解释：总共有 2 门课程。学习课程 1 之前，你需要完成课程 0 。这是可能的。
```

**示例 2：**

```
输入：numCourses = 2, prerequisites = [[1,0],[0,1]]
输出：false
解释：总共有 2 门课程。学习课程 1 之前，你需要先完成课程 0 ；并且学习课程 0 之前，你还应先完成课程 1 。这是不可能的。
```

**解题思路**：

这是一道比较经典的拓扑排序题目。只有当前置课程都学习完成之后才能学习后置的课程。关于课程刻画成类似的图形结构。

![微信截图_20200517052852.png](https://pic.leetcode-cn.com/de601db5bd50985014c7a6b89bca8aa231614b4ba423620dd2e31993c75a9137-%E5%BE%AE%E4%BF%A1%E6%88%AA%E5%9B%BE_20200517052852.png)

本题需要找的是图中是否有环存在，是的课程设置相互制约不合理。就是不存在拓扑排序的结果。即按规则去掉入度为0边之后，是否是有的入度都为0。

如果不是则证明有环存在。代码题解如下：

```python
class Solution:
    def canFinish(self, numCourses: int, prerequisites: List[List[int]]) -> bool:
        edge = {}
        adjant = [0 for i in range(numCourses)]
        for course,precourse in prerequisites:
            if course == precourse:
                return False
            if precourse not in edge.keys():
                edge[precourse] = [course]
            else:
                edge[precourse].append(course)
            adjant[course] += 1

        adjant_0 = [i for i, num in enumerate(adjant) if num== 0]
        while adjant_0:
            precourse = adjant_0.pop(0)
            if precourse in edge.keys():
                for course in edge[precourse]:
                    adjant[course] -= 1
                    if adjant[course] == 0:
                        adjant_0.append(course)
        return True if max(adjant) == 0 else False
```





## day13: [救生艇](https://leetcode-cn.com/problems/boats-to-save-people/)

​                                                                                                                                                                                                                        **难度：**:smile::smile::smile:



####     第 i 个人的体重为 people[i]，每艘船可以承载的最大重量为 limit。

####     每艘船最多可同时载两人，但条件是这些人的重量之和最多为 limit。

####     返回载到每一个人所需的最小船数。(保证每个人都能被船载)。

 

**示例 1：**

```
输入：people = [1,2], limit = 3
输出：1
解释：1 艘船载 (1, 2)
```

**示例 2：**

```
输入：people = [3,2,2,1], limit = 3
输出：3
解释：3 艘船分别载 (1, 2), (2) 和 (3)
```

**示例 3：**

```
输入：people = [3,5,3,4], limit = 5
输出：4
解释：4 艘船分别载 (3), (3), (4), (5)
```

**提示：**

```
1 <= people.length <= 50000
1 <= people[i] <= limit <= 30000
```

**解题思路:**

​    因为每艘船只能最多载两个人，因此我们可以先对people进行排序。利用双指针进行计算。如果指针指向的两个人体重之和 > limit，则先让右边的人单独上船，右指针-1。否则的话两个人一起上船。代码如下：

```python
class Solution:
    def numRescueBoats(self, people: List[int], limit: int) -> int:
        people.sort()
        count = 0
        left = 0
        right = len(people) - 1
        while left <= right:
            if people[left] + people[right] <= limit:
                count += 1
                left += 1
                right -= 1
            else:
                count += 1
                right -= 1
        return count

```





## day14:[优势洗牌](https://leetcode-cn.com/problems/advantage-shuffle/)

​                                                                                                                                                                                                                      **难度：**:smile::smile::smile:



####     给定两个大小相等的数组 A 和 B，A 相对于 B 的优势可以用满足 A[i] > B[i] 的索引 i 的数目来描述。

####     返回 A 的任意排列，使其相对于 B 的优势最大化。

 

**示例 1：**

```
输入：A = [2,7,11,15], B = [1,10,4,11]
输出：[2,11,7,15]
```

**示例 2：**

```
输入：A = [12,24,8,32], B = [13,25,32,11]
输出：[24,32,8,12]
```

**解题思路：**
			这是一道类似于田忌赛马的题目。如果A数组的最小值小于B数组的最小值。那么该值就可以舍弃，拿去和B数组较大的值去做比较。

​    按照这个思路我们可以对A,B两个数组进行排序。如果A的最小值<=B的最小值。我们可以将该值丢弃。否则的话则保留A的当前值和B的值。

​    上述思路的代码实现如下

```python
class Solution:
    def advantageCount(self, nums1: List[int], nums2: List[int]) -> List[int]:
        a = sorted(nums1)
        b = sorted(nums2)
        compared = {i:[] for i in b}
        not_compared = []
        j = 0
        for i in a:
            if i > b[j]:
                compared[b[j]].append(i)
                j += 1
            else:
                not_compared.append(i)
        rs = []
        for i in nums2:
            if compared[i]:
                rs.append(compared[i].pop())
            else:
                rs.append(not_compared.pop())
        return rs
```





## day15:[生命游戏](https://leetcode-cn.com/problems/game-of-life/)

​                                                                                                                                                                                                                            **难度：**:smile::smile::smile:



####     给定一个包含 m × n 个格子的面板，每一个格子都可以看成是一个细胞。每个细胞都具有一个初始状态：1 即为活细胞（live），或 0 即为死细胞（dead）。每个细胞与其八个相邻位置（水平，垂直，对角线）的细胞都遵循以下四条生存定律：

####     如果活细胞周围八个位置的活细胞数少于两个，则该位置活细胞死亡；

####     如果活细胞周围八个位置有两个或三个活细胞，则该位置活细胞仍然存活；

####     如果活细胞周围八个位置有超过三个活细胞，则该位置活细胞死亡；

####     如果死细胞周围正好有三个活细胞，则该位置死细胞复活；

####     下一个状态是通过将上述规则同时应用于当前状态下的每个细胞所形成的，其中细胞的出生和死亡是同时发生的。给你 m x n 网格面板 board 的当前状态，返回下一个状态。

 

**示例 1：**

![img](https://assets.leetcode.com/uploads/2020/12/26/grid1.jpg)

```
输入：board = [[0,1,0],[0,0,1],[1,1,1],[0,0,0]]
输出：[[0,0,0],[1,0,1],[0,1,1],[0,1,0]]
```

**示例 2：**

![img](https://assets.leetcode.com/uploads/2020/12/26/grid2.jpg)

```
输入：board = [[1,1],[1,0]]
输出：[[1,1],[1,1]]
```

**解题思路:**

依次遍历判断当前值周边的活的细胞数。需注意本题要在原数组上修改，且所有值是实时更新的。因此需要对原数组进行深拷贝。遍历比较拷贝数组，修改原数组。代码如下

```python
class Solution:
    def gameOfLife(self, board: List[List[int]]) -> None:
        """
        Do not return anything, modify board in-place instead.
        """
        directed = [(0,-1),(0,1),(-1,0),(1,0),(-1,-1),(-1,1),(1,-1),(1,1)]
        from copy import deepcopy
        new_board = deepcopy(board)
        for i in range(len(new_board)):
            for j in range(len(new_board[0])):
                alive_num = 0
                for x,y in directed:
                    if -1< i+x < len(new_board) and -1 < j+y < len(new_board[0]):
                        if new_board[i+x][j+y] == 1:
                            alive_num += 1
                if new_board[i][j] == 0:
                    if alive_num == 3:
                        board[i][j] = 1
                else:
                    if alive_num < 2 or alive_num > 3:
                        board[i][j] = 0
```


