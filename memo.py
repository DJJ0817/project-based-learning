

"""
#### pipe ####
df = pd.DataFrame({'A': 'a b a b'.split(), 'B': [1, 2, 3, 4]})
#To get the difference between each groups maximum and minimum value in one pass, you can do
print(df)
print(df.groupby('A').)
print(df.groupby('A').pipe(lambda x: x.max() - x.min()))
##############


"""
