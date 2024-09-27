input_str = input()

length = int(0)

if len(input_str) == 1:
    length = 1 

if len(input_str) >= 3: 
    for i in range(0, len(input_str)-2):
        if length >= len(input_str[i::]):
            break
        for j in range(2, len(input_str)-i):
            if input_str[i:i+j+1:1] == input_str[i:i+j+1][::-1]:
                length = max(length, len(input_str[i:i+j+1]))
    



print(length)


