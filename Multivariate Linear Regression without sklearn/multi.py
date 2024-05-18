import numpy as np 
import pandas as pd 
import matplotlib as plt 

my_data = pd.read_csv('./home.txt', names=["size","bedroom","price"])

print(my_data)