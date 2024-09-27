input_str = input()

result = ''

for i in input_str:
    if i.isalpha():
        result += i
    else:
        result += ' '


result_2 = result.split(' ')

for i in result_2[::-1]:
    if i != ' ':
        print(i, end=' ')