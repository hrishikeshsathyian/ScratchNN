from keras.datasets import mnist
import numpy as np

# Loading the data
(x, y), (x_test, y_test) = mnist.load_data()

# Data preprocessing

x_flat = x.reshape((x.shape[0], -1))  / 255.0
y = y.reshape(-1, 1) 
data = np.hstack((y, x_flat)) 
np.random.shuffle(data)

x_test_flat = x_test.reshape(x_test.shape[0], -1) / 255.0
y_test = y_test.reshape(-1, 1)
test_data = np.hstack((y_test, x_test_flat))
np.random.shuffle(test_data)

m, n = data.shape
data_validation = data[0: 1000].T
Y_validation = data_validation[0]
X_validation = data_validation[1: n]

data_train = data[1000: m].T
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
    Z_stable = Z - np.max(Z, axis=0, keepdims=True) # stabilize by reducing values, too high would return NaN
    exp_Z = np.exp(Z_stable)
    return exp_Z / np.sum(exp_Z, axis=0, keepdims=True)

def forward_propagation(X, W1, b1, W2, b2): 
    Z1 = W1 @ X + b1 
    A1 = ReLU(Z1)
    Z2 = W2 @ A1 + b2
    A2 = softmax(Z2)
    return Z1, A1, Z2, A2

def one_hot(Y):
    Y = Y.astype(int) 
    one_hot_Y = np.zeros((Y.size, Y.max() + 1))
    one_hot_Y[np.arange(Y.size), Y] = 1 
    one_hot_Y = one_hot_Y.T # transpose such that each column is an output vector for a training example
    return one_hot_Y


def back_propagation(X, Y, Z1, A1, Z2, A2, W1, W2):
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
    return np.argmax(A2, 0) # returns 1D array of predictions

def get_accuracy(predictions, Y):
    Y = Y.flatten()
    return np.mean(predictions == Y)

def gradient_descent(X, Y, learning_rate, iterations):
    W1, b1, W2, b2 = init_params()
    for i in range(iterations):
        Z1, A1, Z2, A2 = forward_propagation(X, W1, b1, W2, b2)
        dW1, db1, dW2, db2 = back_propagation(X, Y, Z1, A1, Z2, A2, W1, W2)
        W1, b1, W2, b2 = update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, learning_rate)
        if (i % 10 == 0):
            print("Iteration: ", i)
            predictions = get_predictions(A2)
            print(get_accuracy(predictions, Y))
    return W1, b1, W2, b2

if __name__ == "__main__":
    learning_rates = [0.1, 0.5, 1.0]
    params = []
    max_lr = 0
    max_accuracy = 0
    # Train and validate the model with different learning rates and pick the best one
    for learning_rate in learning_rates:
        print(f"Training with learning rate: {learning_rate}")
        W1, b1, W2, b2 = gradient_descent(X_train, Y_train, learning_rate, 500)
        Z1, A1, Z2, A2 = forward_propagation(X_validation, W1, b1, W2, b2)
        predictions = get_predictions(A2)
        accuracy = get_accuracy(predictions, Y_validation)
        if accuracy > max_accuracy:
            max_accuracy = accuracy
            max_lr = learning_rate
            params = (W1, b1, W2, b2)
        print(f"Validation accuracy with learning rate {learning_rate}: {accuracy:.4f}")
    # Test the best parameters on the test set
    W1, b1, W2, b2 = params
    Z1, A1, Z2, A2 = forward_propagation(x_test_flat.T, W1, b1, W2, b2)
    predictions = get_predictions(A2)
    test_accuracy = get_accuracy(predictions, y_test)
    print(f"Test accuracy: {test_accuracy:.4f} with learning rate {max_lr}")
