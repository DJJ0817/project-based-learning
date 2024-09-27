from curses.ascii import isdigit


input_dir = input().split(';')
dir_keys  = ['A', 'W', 'S', 'D']

x_loc = int(0) 
y_loc = int(0)

avail_dir = []
for i in range(len(input_dir)):
    if len(input_dir[i]) >= 2:
        if input_dir[i][0] in dir_keys:
            if len(input_dir[i]) == 2:
                if input_dir[i][1].isdigit():
                    avail_dir.append(input_dir[i])
            if len(input_dir[i]) == 3:
                if input_dir[i][1].isdigit() and input_dir[i][2].isdigit():
                    avail_dir.append(input_dir[i])        


for i in range(len(avail_dir)):
    if len(avail_dir[i]) == 2:
        if avail_dir[i][0] == dir_keys[0]:
            x_loc -= avail_dir[i][1]
        if avail_dir[i][0] == dir_keys[1]:
            y_loc += avail_dir[i][1]
        if avail_dir[i][0] == dir_keys[2]:
            y_loc -= avail_dir[i][1]
        if avail_dir[i][0] == dir_keys[3]:
            x_loc += avail_dir[i][1]
    
    if len(avail_dir[i]) == 3:
        if avail_dir[i][0] == dir_keys[0]:
            x_loc -= (int(avail_dir[i][1])*10 + int(avail_dir[i][2]))
        if avail_dir[i][0] == dir_keys[1]:
            y_loc += (int(avail_dir[i][1])*10 + int(avail_dir[i][2]))
        if avail_dir[i][0] == dir_keys[2]:
            y_loc -= (int(avail_dir[i][1])*10 + int(avail_dir[i][2]))
        if avail_dir[i][0] == dir_keys[3]:
            x_loc += (int(avail_dir[i][1])*10 + int(avail_dir[i][2]))
    
print(x_loc,y_loc, sep=',')

#print(input_dir[0][0])
#print(len(input_dir))
#print(input_dir)
#print(avail_dir)