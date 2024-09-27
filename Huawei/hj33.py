input_ip = input().split('.')
input_10ip = input()

input_ip_bin = []
ip_bin_whole = ''
for i in range(len(input_ip)):
    input_ip_bin.append(bin(int(input_ip[i]))[2::])
    input_ip_bin[i] = input_ip_bin[i].zfill(8)
    ip_bin_whole = ip_bin_whole + input_ip_bin[i]


input_10ip_bin = bin(int(input_10ip))[2::].zfill(32)
input_10ip_bin_sep = []
for i in range(0, len(input_10ip_bin), 8):
    input_10ip_bin_sep.append(int(input_10ip_bin[i:i+8:1], 2))



print(int(ip_bin_whole, 2))
output_10ip = ''
for i in input_10ip_bin_sep:
    output_10ip = output_10ip + str(i)
    output_10ip = output_10ip + '.'

print(output_10ip[::-1][1::][::-1])



