input_num = int(input())

distant = input_num
incre = input_num

for i in range(4):
    distant += incre
    incre = incre/2 


print(distant)
print(incre/2)