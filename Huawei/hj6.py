def prime(n):

    factor = []
    while(n % 2 == 0):
        n = n / 2 
        factor.append(2)

    for i in range(3, int(n**0.5) +1, 2): 
        while(n % i == 0):
            n = n / i
            factor.append(i)
    
    if n > 2: 
        factor.append(int(n))
    
    return factor


input_n = int(input())
print(' '.join(map(str, prime(input_n))))
#print(*prime(input_n), sep=' ')