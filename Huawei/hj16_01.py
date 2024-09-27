total, k = list(map(int, input().split(' ')))

weight = {}
value = {}
main_key = []

for i in range(1, k+1):
    weight[i] = [0,0,0]
    value[i] = [0,0,0]

for i in range(1, k+1):
    v, p, q = list(map(int, input().split(' ')))
    if q == 0:
        weight[i][0] = v 
        value[i][0] = v*p
        main_key.append(i)
    else:
        if weight[q][1] == 0:
            weight[q][1] = v
            value[q][1] = v*p
        else:
            weight[q][2] = v
            value[q][2] = v*p
        
w_list = []
v_list = []

for key in weight:
    if key in main_key:
        w_list.append(weight[key])
        v_list.append(value[key])


print(weight)
print(w_list)

m = len(w_list)
dp = [[0]*(total+1) for i in range(m+1)]

for i in range(1, m+1):
    w1 = w_list[i-1][0]
    w2 = w_list[i-1][1] 
    w3 = w_list[i-1][2]
    v1 = v_list[i-1][0]
    v2 = v_list[i-1][1] 
    v3 = v_list[i-1][2]

    for j in range(1, total+1):
        dp[i][j] = dp[i-1][j]
        if j-w1>=0:
            dp[i][j] = max(dp[i][j], dp[i-1][j-w1]+v1)
        if j-w1-w2>=0:
            dp[i][j] = max(dp[i][j], dp[i-1][j-w1-w2]+v1+v2)
        if j-w1-w3>=0:
            dp[i][j] = max(dp[i][j], dp[i-1][j-w1-w3]+v1+v3)
        if j-w1-w2-w3>=0:
            dp[i][j] = max(dp[i][j], dp[i-1][j-w1-w2-w3]+v1+v2+v3)


print(dp[m][total])
print(dp)