input_num = int(input())

input_list = [[] for i in range(input_num)]

d = 1
for i in range(input_num):
    for j in range(i+1):
        input_list[i].append(d)
        d += 1 

for i in range(input_num):
    for j in input_list:
        if j:
            print(j.pop(-1), end=' ') 
    print()


