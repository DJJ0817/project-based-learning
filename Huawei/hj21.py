alpha = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
alpha_l = 'bcdefghijklmnopqrstuvwxyza'

trans_1 = '1'
trans_2 = 'abc'
trans_3 = 'def'
trans_4 = 'ghi'
trans_5 = 'jkl'
trans_6 = 'mno'
trans_7 = 'pqrs'
trans_8 = 'tuv'
trans_9 = 'wxyz'
trans_0 = '0'


input_str = input()
output_str = [0]*len(input_str)

for i in range(len(input_str)): 
    if input_str[i].isupper():
        for j in range(len(alpha)):
            if alpha[j] == input_str[i]:
                output_str[i] = alpha_l[j]
    
    if input_str[i] in trans_1:
        output_str[i] = 1
    if input_str[i] in trans_2:
        output_str[i] = 2
    if input_str[i] in trans_3:
        output_str[i] = 3
    if input_str[i] in trans_4:
        output_str[i] = 4
    if input_str[i] in trans_5:
        output_str[i] = 5
    if input_str[i] in trans_6:
        output_str[i] = 6
    if input_str[i] in trans_7:
        output_str[i] = 7
    if input_str[i] in trans_8:
        output_str[i] = 8
    if input_str[i] in trans_9:
        output_str[i] = 9
    if input_str[i] in trans_0:
        output_str[i] = 0
    
    if input_str[i] not in trans_1 and input_str[i] not in trans_0 and input_str[i].isdigit():
        output_str[i] = input_str[i]
    

for i in output_str:
    print(i, end='')