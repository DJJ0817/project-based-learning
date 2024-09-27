input_num = int(input())
id_all = []
for i in range(input_num):
    id_all.append(input().split(' '))

print(len(id_all))
        
test = []
for i in range(len(id_all)):
    for j in range(i+1, len(id_all)-1):
        if id_all[i][0] == id_all[j][0] and len(set(id_all[i]+id_all[j][1::])) < len(id_all[i]+id_all[j][1::]):
            test.append(list(set(id_all[i]+id_all[j][1::])))
            id_all[i] = id_all[i][0]+str(set(id_all[i][1::]+id_all[j][1::]))
            del(id_all[j])

for i in range(len(id_all)):
    print(id_all[i])


'''
1 3 5 
4 
3 1 2 6 
3 2 3 7
3 5 6 8 
2 5 7 
'''


'''
4 
zhangsan 1000 1001
lisi 1020
zhangsan 1000 1002
lisi 1010
'''