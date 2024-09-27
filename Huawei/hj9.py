input_num = input()

input_rev = input_num[::-1]

output = ''
for i in input_rev:
    if i not in output:
        output = output + i

print(output)


print(type(input_num))
print(type(output))