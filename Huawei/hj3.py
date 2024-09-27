input_n = input('Enter a number: ')
series = []
for i in range(int(input_n)):
    num = int(input(''))
    series.append(num)

series.sort()
series_output = set(series)

for i in series_output:
    print(i)