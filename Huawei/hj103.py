# 输入数据
n = int(input())  # 输入数组长度
heights = list(map(int, input().split()))  # 输入梅花桩的高度

# 初始化dp数组，每个位置的最小步数为1
dp = [1] * n

# 遍历每一个起始点
for i in range(n):
    # 遍历从当前点之后的每一个点
    for j in range(i + 1, n):
        # 如果可以从 i 跳到 j (严格递增)
        if heights[j] > heights[i]:
            # 更新 dp[j] 为跳到 j 位置后的最大步数
            dp[j] = max(dp[j], dp[i] + 1)
            print(heights[i])
            print(dp)

# 输出 dp 数组的最大值，即为最大的步数
print(max(dp))



