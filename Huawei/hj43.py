'''
def dfs(i,j):
    dx = [0,0,-1,1]
    dy = [-1,1,0,0]
    if i == m-1 and j == n-1:
        for pos in route:
            print('('+str(pos[0])+','+str(pos[1])+')')
        return 
    
    for k in range(4):
        x = i+dx[k]
        y = j+dy[k]
        if x>=0 and x<m and y>=0 and y<n and map1[x][y]==0:
            map1[x][y] = 1
            route.append((x,y))
            print((x,y))
            dfs(x,y)
            map1[x][y] = 0
            route.pop()
        #else:
        #    return 

   

m, n = list(map(int, input().split()))
map1 = []
for i in range(m):
    map1.append(list(map(int, input().split(' '))))
        
route = [(0,0)]
map1[0][0] = 1
dfs(0,0)
'''


def dsf(i,j,m,n):
    dx = [-1,1,0,0]
    dy = [0,0,1,-1]
    if i == m-1 and j == n-1:
        for pos in route:
            print(pos)
        return 
    
    for k in range(4):
        x = i + dx[k]
        y = j + dy[k]
        if x>=0 and x<m and y>=0 and y<n and map1[x][y] == 0:
            map1[x][y] = 1
            route.append((x,y))
            dsf(x,y,m,n)
            map1[x][y] = 0
            route.pop()

m, n = map(int, input().split(' '))
map1 = []
for i in range(m):
    map1.append(list(map(int, input().split(' '))))

map1[0][0] = 1
route = [(0,0)]
dsf(0,0,m,n)
