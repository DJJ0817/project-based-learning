input_word = input('input: ').lower()
find_word  = input('find: ').lower()

count_num = 0
for i in input_word:
    if i == find_word:
        count_num += 1

print(count_num)

count_num_2 = input_word.count(find_word) 
print(count_num_2)