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
Y_validation = data_validation[0] # flat array of labels
X_validation = data_validation[1: n]

data_train = data[1000: m].T # make each column a training example
Y_train = data_train[0]
X_train = data_train[1: n] # (784, 59000) because we set aside 1000 for validation

def init_params(): 
    W1 = np.random.rand(10, 784) - 0.5
    b1 = np.random.rand(10, 1) 
    W2 = np.random.rand(10, 10) - 0.5
    b2 = np.random.rand(10, 1) 
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
    return Z1, A1, Z2, A2

def one_hot(Y):
    Y = Y.astype(int).flatten()  # Ensures shape (m,) and integer type
    # Create a 2D array of zeros with:
    # - number of rows equal to the number of elements in Y (i.e., one row per label)
    # - number of columns equal to the maximum value in Y plus 1
    #   (assuming classes are labeled from 0 to Y.max())
    one_hot_Y = np.zeros((Y.size, Y.max() + 1))
    
    # np.arange(Y.size) creates an array of indices [0, 1, 2, ..., Y.size-1]
    # The expression one_hot_Y[np.arange(Y.size), Y] selects the position in each row
    # corresponding to the label in Y, and sets that position to 1.
    one_hot_Y[np.arange(Y.size), Y] = 1

    one_hot_Y = one_hot_Y.T
    return one_hot_Y


def deriv_Relu(Z):
    return Z > 0 # works due to boolean to int conversion

def backwards_propagation(X, Y, Z1, A1, Z2, A2, W1, W2):
    m = Y.size 
    one_hot_Y = one_hot(Y)
    dZ2 = A2 - one_hot_Y 
    dW2 = (dZ2 @ A1.T) / m
    db2 = np.sum(dZ2, axis=1, keepdims=True) / m # take average across all training examples 
    
    dA1 = W2.T @ dZ2
    dZ1 = dA1 * (Z1 > 0)  # Derivative of ReLU
    dW1 = (dZ1 @ X.T) / m
    db1 = np.sum(dZ1, axis=1, keepdims=True) / m
    
    return dW1, db1, dW2, db2

def update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, learning_rate=0.01):
    W1 -= learning_rate * dW1
    b1 -= learning_rate * db1
    W2 -= learning_rate * dW2
    b2 -= learning_rate * db2
    return W1, b1, W2, b2


def get_predictions(A2):
    return np.argmax(A2, 0)

def get_accuracy(predictions, Y):
    Y = Y.flatten()
    return np.mean(predictions == Y)

def gradient_descent(X, Y, learning_rate, iterations):
    W1, b1, W2, b2 = init_params()
    for i in range(iterations):
        Z1, A1, Z2, A2 = forward_propagation(X, W1, b1, W2, b2)
        dW1, db1, dW2, db2 = backwards_propagation(X, Y, Z1, A1, Z2, A2, W1, W2)
        W1, b1, W2, b2 = update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, learning_rate)
        if (i % 10 == 0):
            print("Iteration: ", i)
            predictions = get_predictions(A2)
            print(get_accuracy(predictions, Y))
    return W1, b1, W2, b2

if __name__ == "__main__":
    learning_rate = 0.5
    iterations = 1000
    W1, b1, W2, b2 = gradient_descent(X_train, Y_train, learning_rate, iterations)
    print("Training complete.")
    print(f"Final Values of W1: {W1}, b1: {b1}, W2: {W2}, b2: {b2}")
    
