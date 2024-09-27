input_num = int(input())
input_dict = {}

for i in range(input_num): 
    a = input()
    k, v = a.split(' ')
    k = int(k)
    v = int(v)
    if k not in input_dict:
        input_dict[k] = v
    else:
        input_dict[k] += v

sorted(input_dict.items(), key = lambda x:x[0])

for key, value in input_dict.items():
    print(key, value)