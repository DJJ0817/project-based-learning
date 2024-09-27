import collections


input_num = int(input())

for i in range(input_num):
    s = input()
    c = collections.Counter(s)
    l = sorted(c.items(), key=lambda x:x[1], reverse=True)
    ans = 0 
    k = 26

    for i in l:
        ans += i[1]*k
        k -= 1
    print(ans)
        