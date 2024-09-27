input_num = int(input())

words = []
for i in range(input_num):
    word = input()
    words.append(word)

words.sort()
for i in words:
    print(i)