def check(input_password):
    if len(input_password) < 8:
        return False

    u = 0
    l = 0 
    d = 0 
    s = 0 

    for i in input_password:
        if i.isupper():
            u = 1
        elif i.islower():
            l = 1
        elif i.isdigit():
            d = 1 
        else:
            s = 1
    
    if u+l+d+s < 3: 
        return False 
    
    for i in range(len(input_password)-2):
        part = input_password[i:i+3]
        apart = input_password.split(part)
        if len(apart) >= 3: 
            return False 
        
    return True 




while True: 
    try:
        code = input()
        if check(code):
            print('OK')
        else:
            print('NG')

    except:
        break