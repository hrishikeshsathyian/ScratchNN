from keras.datasets import mnist
import numpy as np

# Loading the data
(x_train, y_train), (x_test, y_test) = mnist.load_data()
# x_train shape (60000, 28, 28) 
# y_train shape (60000,) : flat array of labels
# x_test shape (10000, 28, 28) 
# y_test shape (10000,) : flat array of labels

## Data preprocessing

x_train_flat = x_train.reshape((x_train.shape[0], -1))  / 255.0
y_train = y_train.reshape(-1, 1) 
data = np.hstack((y_train, x_train_flat)) 
np.random.shuffle(data)

x_test_flat = x_test.reshape(x_test.shape[0], -1) / 255.0
y_test = y_test.reshape(-1, 1)
test_data = np.hstack((y_test, x_test_flat))
np.random.shuffle(test_data)

# Get the number of rows and columns
m, n = data.shape
# after transposing, each training example is a column
# after transposing, each training example is a column
data_validation = data[0: 1000].T
Y_validation = data_validation[0]
X_validation = data_validation[1: n]

data_train = data[1000: m].T # make each column a training example
Y_train = data_train[0]
X_train = data_train[1: n] # (784, 59000) because we set aside 1000 for validation

def init_params(): 
    W1 = np.random.randn(10, 784) - 0.5
    b1 = np.random.randn(10, 1) - 0.5
    W2 = np.random.randn(10, 10) - 0.5
    b2 = np.random.randn(10, 1) - 0.5
    return W1, b1, W2, b2

def ReLU(Z):
    return np.maximum(0, Z)

def softmax(Z):
    Z_stable = Z - np.max(Z, axis=0, keepdims=True)
    exp_Z = np.exp(Z_stable)
    return exp_Z / np.sum(exp_Z, axis=0, keepdims=True)

def forward_propagation(X, W1, b1, W2, b2): 
    Z1 = W1 @ X + b1 
    A1 = ReLU(Z1)
    Z2 = W2 @ A1 + b2
    A2 = softmax(Z2)

