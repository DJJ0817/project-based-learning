input_sent = input().split(' ')

for i in input_sent[::-1]:
    print(i, end=' ')


"""
def reverse_sentence(sentence):
    # 将句子分割为单词列表
    words = sentence.split()
    # 将单词列表逆序排列并重新组合成字符串
    reversed_sentence = ' '.join(words[::-1])
    return reversed_sentence

# 测试示例
sentence = "I am a boy"
print(reverse_sentence(sentence))  # 输出: "boy a am I"

"""