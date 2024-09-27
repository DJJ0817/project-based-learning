def max_drink(n):
    drink = 0 

    while n // 3 != 0:
        drink += n//3
        add_drink = n//3
        n = n%3 
        n += add_drink 

        if n == 2:
            drink += 1
            n = 0
    
    return drink


while True: 
    n = int(input())
    if n != 0:
        print(max_drink(n))
    elif n == 0:
        break


