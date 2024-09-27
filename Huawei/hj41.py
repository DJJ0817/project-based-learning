input_num = int(input())
input_weights = list(map(int, input().split()))
input_counts = list(map(int, input().split()))

possible = [0]
situ = []

for i in range(len(input_counts)):
    for j in range(input_counts[i]):
        situ.append(input_weights[i])

#print(situ) # 1,1,2

for i in range(len(situ)):
    temp = possible
    for j in range(len(temp)):
        possible.append(temp[j] + situ[i])
        possible.append(situ[i])
        possible = list(set(possible))



#print(possible)
print(len(set(possible)))