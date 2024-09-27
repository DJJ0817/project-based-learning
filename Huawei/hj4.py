"""
input_str = input("Enter a string: ")

if len(input_str) < 8: 
    less_num = 8 - len(input_str)
    output_str = input_str + "0" * less_num
    print(output_str)

if len(input_str) >= 8:
    times = len(input_str) // 8
    for i in range(times):
        print(input_str[i*8:i*8+8])

    less_num = len(input_str) - times * 8
    if less_num > 0:
        print(input_str[times*8::] + "0" * (8-less_num))
"""


input_str = input("")

if len(input_str) % 8 != 0:
    input_str += "0" * (8 - len(input_str) % 8)

for i in range(0, len(input_str), 8): 
    print(input_str[i:i+8])

    
