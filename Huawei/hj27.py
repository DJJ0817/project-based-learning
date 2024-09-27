input_ser = input('').split(' ')

input_test = input_ser[1:int(input_ser[0])+1]
input_ori = input_ser[-2]
input_find = int(input_ser[-1])

find_part = []




for i in input_test:
    if i != input_ori and len(i) == len(input_ori): 
            if sorted(i) == sorted(input_ori):
                find_part.append(i)

sorted_find = sorted(find_part)

print(len(find_part))
if len(sorted_find) >= input_find:
    print(sorted_find[input_find-1])

