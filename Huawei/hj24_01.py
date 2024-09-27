import bisect 


def maxnum(l):
    res = [1]*len(l)
    arr = [l[0]]
    for i in range(1, len(l)):
        if arr[-1] < l[i]:
            arr.append(l[i])
            res[i] = len(arr)
        else:
            pos = bisect.bisect_left(arr, l[i])
            arr[pos] = l[i]
            res[i] = pos + 1 

    return res
            

input_num = int(input())
input_series = list(map(int, input().split()))

l1 = maxnum(input_series)
l2 = maxnum(input_series[::1])[::-1]

l3 = max(l1[i] + l2[i] for i in range(len(input_series)))

print(len(input_series) - (l3-1))