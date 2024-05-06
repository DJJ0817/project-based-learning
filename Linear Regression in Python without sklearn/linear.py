import numpy as np
import matplotlib.pyplot as plt 

#read data (100,2)

class reg: 
    def __init__(self, file, alpha, iters, ini_f, ini_b):
        self.datafile = file 
        self.alpha = alpha 
        self.iters = iters
        self.f     = ini_f
        self.b     = ini_b
    
    def read_build_data(self):
        my_data = np.genfromtxt(self.datafile, delimiter=',')

        size_x = my_data.shape[0]
        X_1 = my_data[:, 0].reshape(-1,1)
        X_2 = np.ones([size_x, 1])
        X = np.concatenate([X_1, X_2], axis=1)
        y = my_data[:, 1].reshape(-1, 1)

        theta = np.array([[self.f, self.b]])

        return X, y, theta

    def costfunction(self, theta_update):
        
        X, y, theta = self.read_build_data()
        inner = np.power((X @ theta_update.T) - y, 2)
        rmse = np.sum(inner) / (2 * X.shape[0])
        return rmse 
    
    def gradient_descent(self, verbose = False):
        X, y, theta = self.read_build_data() 

        for i in range(self.iters):
            theta = theta - (self.alpha / X.shape[0]) * np.sum((X @ theta.T - y) * X, axis=0)
            cost = self.costfunction(theta)

            if verbose:
                print(cost)
       
        return theta, cost
    
    def result_plot(self):
        theta, _ = self.gradient_descent()
        f = theta[0,0]
        b = theta[0,1]

        my_data = np.genfromtxt(self.datafile, delimiter=',')
        plt.scatter(my_data[:,0], my_data[:,1], color='blue')
        axes = plt.gca() # obtain the axis information
        x_vals = np.array(axes.get_xlim()) # obtain the min and the max value of x 
        y_vals = f * x_vals + b
        plt.plot(x_vals, y_vals, '--')
        #plt.show()

        return plt.show()


linear_reg = reg(file="./data.csv", alpha=0.0001, iters=100, ini_f=1.0, ini_b=1.0)
#theta, cost = linear_reg.result_plot()
linear_reg.result_plot()









"""
def decode(message_file):

  with open(message_file, 'r') as file:
    lines = [line.strip().split() for line in file]

  numbers_and_words = [(int(num), word) for num, word in lines]
  
  numbers_and_words.sort()

  words = [word for _, word in numbers_and_words]

  
  index = 0
  index_incre = 1
  decoded_message = []
  while True: 
    word_index = index + index_incre
    
    if word_index > len(words):
      break
    
    decoded_message.append(words[word_index-1])


    index_incre += 1 
    index = word_index 

    output_decoded_message = " ".join(decoded_message)
  return print(output_decoded_message)

decode("coding_qual_input.txt")

"""
