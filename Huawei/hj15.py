input_num = int(input())

bin_num = []

while input_num != 0:
    bin_num.append(input_num % 2)
    input_num = input_num //2

times = bin_num.count(1)
print(times)
