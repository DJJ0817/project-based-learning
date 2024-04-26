"""
https://medium.com/@okanyenigun/building-a-neural-network-from-scratch-in-python-a-step-by-step-guide-8f8cab064c8a
https://www.youtube.com/watch?v=0oWnheK-gGk
"""
import numpy as np 
from random import random

# save the activations and derivatives
# implement backpropagation 
# gradient descent 
# train
# train dataset and prediction  

class MLP: 
    def __init__(self, num_inputs=3, num_hidden=[3,3], num_outputs=2): 

        self.num_inputs = num_inputs
        self.num_hidden = num_hidden
        self.num_outputs = num_outputs 

        layers = [self.num_inputs] + self.num_hidden + [self.num_outputs]

        weights = []
        for i in range(len(layers)-1): 
            w = np.random.rand(layers[i], layers[i+1])
            weights.append(w)
        self.weights = weights

        activations = []
        for i in range(len(layers)):
            a = np.zeros(layers[i])
            activations.append(a)
        self.activations = activations
        
        derivatives = []
        for i in range(len(layers)-1):
            d = np.zeros((layers[i], layers[i+1]))
            derivatives.append(d)
        self.derivatives = derivatives

    
    def forward_propagate(self, inputs): 
        activatins = inputs 
        self.activations[0] = inputs 

        for i, w in enumerate(self.weights): 
            net_inputs = np.dot(activatins, w)
            activatins = self._sigmoid(net_inputs)
            self.activations[i+1] = activatins
            #a_3 = sigma(h_3)
            #h_3 = weight_2 * a_2 
            #the activation connect with the second weight matrix
            #is the third layer's activation
        
        return activatins 
    
    def back_propagate(self, error, verbose=False): 
        #dE/Dw_i = (y - a_[i+1]) s'(h_[i+1]) a_i
        #s'(h_[i+1]) s(h_[i+1])(1-s(h_[i+1]))
        #s(h_[i+1]) = a_[i+1]

        for i in reversed(range(len(self.derivatives))):
            activations = self.activations[i+1]
            #delta = (y - a_[i+1]) s'(h_[i+1])
            delta = error * self._sigmoid_derivative(activations) 
            delta_reshaped = delta.reshape(delta.shape[0], -1).T
            current_activations = self.activations[i]
            current_activations_reshaped = current_activations.reshape(current_activations.shape[0], -1)
            self.derivatives[i] = np.dot(current_activations_reshaped, delta_reshaped)

            #dE/dW_[i-1] = ((y - a[i+1]) s'(h_[i+1])) W_i s'(h_i) a_[i-1]
            #error = ((y - a[i+1]) s'(h_[i+1])) W_i 
            error = np.dot(delta, self.weights[i].T)  #???

            if verbose:
                print("Derivatives for W{}: {}".format(i, self.derivatives[i]))

            
        return error  

    def _sigmoid_derivative(self, x):
        #s'(h_[i+1]) s(h_[i+1])(1-s(h_[i+1]))
        return x * (1.0 - x)
    
    def gradient_descent(self, learning_rate):
        for i in range(len(self.weights)):
            weights = self.weights[i]
            #print("original weights w{} {}".format(i, weights))
            derivatives = self.derivatives[i]
            weights += derivatives * learning_rate
            #print("updated weights w{} {}".format(i, weights))
    

    def train(self, inputs, targets, epochs, learning_rate):
        for i in range(epochs):
            sum_error = 0
            #for j, (input, target) in enumerate(zip(inputs, targets)):
            for input, target in zip(inputs, targets):

                output = self.forward_propagate(input)

                error = target - output 

                self.back_propagate(error)

                #gradient descent 
                self.gradient_descent(learning_rate)

                sum_error += self._mse(target, output)
            print("Error: {} at epoch {}".format(sum_error / len(inputs), i))

    def _mse(self, target, output):
        return np.average((target - output)**2)





    def _sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
    

if __name__ == "__main__": 

    #create an mlp 
    #create data
    #forward propagation 
    #backward propagation 

    inputs = np.array([[random() / 2 for _ in range(2)] for _ in range(1000)])   #array([[0.1,0.2], [0.3,0.4]])
    targets = np.array([[i[0] +i[1]] for i in inputs])   #array([[0.3], [0.7]])

    mlp = MLP(2, [5], 1)

    mlp.train(inputs, targets, 50, 0.1)

    #create dummy data
    input = np.array([0.3, 0.1])
    target = np.array([0.4])
    
    output = mlp.forward_propagate(input)
    print("out network says {}+{} is equal to {}".format(input[0], input[1], output[0]))
    

















"""



import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from statistics import mean
from typing import Dict, List, Tuple

np.random.seed(42)

class Neural:
    
    def __init__(self, layers: List[int], epochs: int, 
                 learning_rate: float = 0.001, batch_size: int=32,
                 validation_split: float = 0.2, verbose: int=1):
        self._layer_structure: List[int] = layers
        self._batch_size: int = batch_size
        self._epochs: int = epochs
        self._learning_rate: float = learning_rate
        self._validation_split: float = validation_split
        self._verbose: int = verbose
        self._losses: Dict[str, float] = {"train": [], "validation": []}
        self._is_fit: bool = False
        self.__layers = None
        
    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        # validation split
        X, X_val, y, y_val = train_test_split(X, y, test_size=self._validation_split, random_state=42)
        # initialization of layers
        self.__layers = self.__init_layers()
        for epoch in range(self._epochs):
            epoch_losses = []
            for i in range(1, len(self.__layers)):
                # forward pass
                x_batch = X[i:(i+self._batch_size)]
                y_batch = y[i:(i+self._batch_size)]
                pred, hidden = self.__forward(x_batch)
                # calculate loss
                loss = self.__calculate_loss(y_batch, pred)
                epoch_losses.append(np.mean(loss ** 2))
                #backward
                self.__backward(hidden, loss)
            valid_preds, _ = self.__forward(X_val)
            train_loss = mean(epoch_losses)
            valid_loss = np.mean(self.__calculate_mse(valid_preds,y_val))
            self._losses["train"].append(train_loss)
            self._losses["validation"].append(valid_loss)
            if self._verbose:
                print(f"Epoch: {epoch} Train MSE: {train_loss} Valid MSE: {valid_loss}")
        self._is_fit = True
        return
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        if self._is_fit == False:
            raise Exception("Model has not been trained yet.")
        pred, hidden = self.__forward(X)
        return pred
    
    def plot_learning(self) -> None:
        plt.plot(self._losses["train"],label="loss")
        plt.plot(self._losses["validation"],label="validation")
        plt.legend()
    
    def __init_layers(self) -> List[np.ndarray]:
        layers = []
        for i in range(1, len(self._layer_structure)):
            layers.append([
                np.random.rand(self._layer_structure[i-1], self._layer_structure[i]) / 5 - .1,
                np.ones((1,self._layer_structure[i]))
            ])
        return layers
    
    def __forward(self, batch: np.ndarray) -> Tuple[np.ndarray, List[np.ndarray]]:
        hidden = [batch.copy()]
        for i in range(len(self.__layers)):
            batch = np.matmul(batch, self.__layers[i][0]) + self.__layers[i][1]
            if i < len(self.__layers) - 1:
                batch = np.maximum(batch, 0)
            # Store the forward pass hidden values for use in backprop
            hidden.append(batch.copy())
        return batch, hidden
    
    def __calculate_loss(self,actual: np.ndarray, predicted: np.ndarray) -> np.ndarray:
        "mse"
        return predicted - actual
    
    
    def __calculate_mse(self, actual: np.ndarray, predicted: np.ndarray) -> np.ndarray:
        return (actual - predicted) ** 2
    
    def __backward(self, hidden: List[np.ndarray], grad: np.ndarray) -> None:
        for i in range(len(self.__layers)-1, -1, -1):
            if i != len(self.__layers) - 1:
                grad = np.multiply(grad, np.heaviside(hidden[i+1], 0))
    
            w_grad = hidden[i].T @ grad
            b_grad = np.mean(grad, axis=0)
    
            self.__layers[i][0] -= w_grad * self._learning_rate
            self.__layers[i][1] -= b_grad * self._learning_rate
            
            grad = grad @ self.__layers[i][0].T
        return
    
"""
