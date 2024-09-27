def sort_string(l):
    letters = sorted([char for char in l if char.isalpha()], key=lambda c: c.lower())
    result = []
    index = 0

    for i in l:
        if i.isalpha():
            result.append(letters[index])
            index += 1 
        else: 
            result.append(i)


    return ''.join(result)


l = input()
print(sort_string(l))
