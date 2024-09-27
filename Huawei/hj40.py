input_str = input()

num_alpha = 0 
num_space = 0 
num_digit = 0 
num_other = 0 

for i in range(len(input_str)):
    if input_str[i].isalpha(): 
        num_alpha += 1 
    if input_str[i].isdigit():
        num_digit += 1 

for i in input_str:
    if i == ' ':
        num_space += 1 

num_other = len(input_str) - num_alpha - num_digit - num_space 

print(num_alpha)
print(num_space)
print(num_digit)
print(num_other)