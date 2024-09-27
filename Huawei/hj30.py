input_str = input().split(' ')
input_com = ''
for i in range(len(input_str)):
    input_com = input_com + input_str[i]

e_part = []
o_part = []
whole = []
e_part = sorted(input_com[1::2])
o_part = sorted(input_com[::2])

for i in range(len(e_part)):
    whole.append(o_part[i])
    whole.append(e_part[i])

if len(o_part) > len(e_part):
    whole.append(o_part[-1])

result = []

for i in range(len(whole)):
    if whole[i].isalpha(): 
        result.append(int(bin(int(whole[i], 16))[2::].zfill(4)[::-1], 2))
    if whole[i].isdigit():
        if int(bin(int(whole[i]))[2::].zfill(4)[::-1],2) >= 10:
            result.append(hex(int(bin(int(whole[i]))[2::].zfill(4)[::-1],2))[2::].upper())
        else:
            result.append(int(bin(int(whole[i]))[2::].zfill(4)[::-1],2))

for i in range(len(result)):
    if result[i] >= 10:
        result[i] = hex(result[i])[2::].upper() 
        
for i in range(len(result)):
    print(result[i], end='')


#print(result)
#print(hex(result[1])[2::].upper())