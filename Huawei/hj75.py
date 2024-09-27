input_1 = input()
input_2 = input()

if len(input_1) < len(input_2):
    input_1 = input_1
else:
    temp = input_1 
    input_1 = input_2 
    input_2 = temp 

maxnum = 0 
for i in range(len(input_1)):
    for j in range(i+1, len(input_1)+1):
        if input_1[i:j:1] in input_2:
            maxnum = max(maxnum, len(input_1[i:j:1]))


print(maxnum)