
import numpy as np 

current =np.array([0.1, 0.2])
print(current.shape)
print(current)

current_reshape = current.reshape(current.shape[0], -1) # -1 means let python decide the secone dimansion
print(current_reshape.shape)
print(current_reshape.T.shape)
print(current_reshape)
print(current_reshape.T)
