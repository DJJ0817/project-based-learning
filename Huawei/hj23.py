import collections

input_str = input()
c = collections.Counter(input_str)
m = min(c.values())
#print(c)

for i in input_str:
    if c[i] != m:
        print(i, end='')
