# ScratchNN
Building a two-layer deep neural network from scratch using only Python and NumPy — no external libraries — to classify handwritten digits from the MNIST dataset.

## 💡 Motivation
I built this project as a stepping stone. Rather than relying on high-level frameworks like TensorFlow or PyTorch, I wanted to implement core concepts manually to solidify my foundation and understanding of deep learning before progressing into more complex projects.

## 📊 Loading the Dataset
The MNIST dataset consists of 60,000 training examples and 10,000 test examples of handwritten digits.
Each example is a 28×28 grayscale image, where each pixel value ranges from 0 to 255, representing pixel intensity.
The dataset also includes corresponding digit labels (0–9) for each image.
```python 
(x, y), (x_test, y_test) = mnist.load_data()
```
x shape: (60000, 28, 28)
y shape: (60000,) 
x_test shape: (10000, 28, 28)
y_test shape: (10000,) 

## 📜 Data Preprocessing 
For data preprocessing, I wanted to shuffle the data and then split it up into training and validation. 
I reshaped the input matrix, x into a 2D array where each row would represent a training example. Then I reshaped the label array from a flat array of labels into a 2D column vector. I then appended the label values to each row (training examples) with h stack. Then, I shuffled the data to ensure randomness. 
```python 
x_flat = x.reshape((x.shape[0], -1))  / 255.0
y = y.reshape(-1, 1) 
data = np.hstack((y, x_flat)) 
np.random.shuffle(data)

x_test_flat = x_test.reshape(x_test.shape[0], -1) / 255.0
y_test = y_test.reshape(-1, 1)
test_data = np.hstack((y_test, x_test_flat))
np.random.shuffle(test_data)
```
Then I took the first 1000 rows (1000 training examples) to be my validation set, and transposed it such that each column is a training example. Since the first row would now be the labels, I spliced the matrix and stored the first row in Y_validation as a 1D array, and kept the remaining matrix in X_validation.
I repeated the same process for the remaining rows for my training set.
```python 
m, n = data.shape
data_validation = data[0: 1000].T
Y_validation = data_validation[0]
X_validation = data_validation[1: n]

data_train = data[1000: m].T
Y_train = data_train[0]
X_train = data_train[1: n] # (784, 59000) because we set aside 1000 for validation

```

## ⏩ Overview of Forward Propagation 
The network consists of:

Input layer X: Each input image is flattened into a 784-dimensional column vector (28×28 pixels).
Hidden layer A1: 10 neurons with ReLU activation.
Output layer A2: 10 neurons with softmax activation, each corresponding to a digit (0–9).

I chose ReLU as an activation function for the first hidden layer instead of other common activations such as tanh and sigmoid, so as to avoid the issue of vanishing gradients as well as to be able to reap the benefits of sparse activation. That is, ReLU "turns off" the neurons that have negative inputs and hence prevents overfitting.

For the output layer, I chose softmax as an activation function because turning raw inputs into probabilities was well suited for multi-class classification.

Overall, a quick overview of the math of forward propagation would be as follows:

Z<sup>1</sup>  = W<sup>1</sup>X + b1 <br>
A<sup>1</sup> = ReLU(Z<sup>1</sup>) <br>
Z<sup>2</sup>  = W<sup>2</sup>A<sup>1</sup> + b2 <br>
A<sup>2</sup> = softmax(Z<sup>2</sup>) <br>


## ⏩ Implementation (Forward Propagation)
```python 
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
```

## ⬅️ Overview (Back Propagation)
Before beginning with backpropagation, we have to one hot encode our label array Y. Currently, Y is a flat array with all the label values from 0 to 9. 
I want to one-hot-encode it such that each output label now becomes an array of zeros with length 10 with only the corresponding index being changed to 1.
```python 
def one_hot(Y):
    Y = Y.astype(int) 
    one_hot_Y = np.zeros((Y.size, Y.max() + 1))
    one_hot_Y[np.arange(Y.size), Y] = 1 
    one_hot_Y = one_hot_Y.T # transpose such that each column is an output vector for a training example
    return one_hot_Y
```

For the main chunk of backwards propagation, we trace back the mathematical overview outlined in Forward Propagation and also the chain rule to get the values for our derivatives. 

For instance to get the derivative of W<sup>2</sup> we take that 
dL/dW2 = dL/dZ2 * dZ2/dW2 = dL/dZ1 * A1

Most noteably, using softmax and cross-entropy loss allows us to being with 
```python 
dZ2 = A2 - one_hot_Y 
```
Note, that in our derivatives, sometimes we take the transpose to allow shapes to match for matrix multiplication.


## ⬅️ Implementation (Back Propagation)
```python 
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
```

## 🏔 Gradient Descent
Firstly, we initialise random parameters. And for each iteration in the training, we forward propagate to get the predictions, then call back_propagation to get the delta values and update the values accordingly. Every 10 iterations, we log the accuracy of the predictions.
```python 
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
```

## 🏋️‍♂️ Training 
For the sake of simplicity, I only tested one hyperparameter, the learning rate (alpha). For each learning rate, I ran gradient descent and kept the final weights and bias from the learning rate which had the highest accuracy. Then I ran it on the test set.
```python 
if __name__ == "__main__":
    learning_rates = [0.1, 0.5, 1.0]
    params = []
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
            params = (W1, b1, W2, b2)
        print(f"Validation accuracy with learning rate {learning_rate}: {accuracy:.4f}")
    # Test the best parameters on the test set
    W1, b1, W2, b2 = params
    Z1, A1, Z2, A2 = forward_propagation(x_test_flat.T, W1, b1, W2, b2)
    predictions = get_predictions(A2)
    test_accuracy = get_accuracy(predictions, y_test)
    print(f"Test accuracy: {test_accuracy:.4f}")
```
## 📊 Results 
For the sake of simplicity, I only tested one hyperparameter, the learning rate (alpha). For each learning rate, I ran gradient descent and kept the final weights and bias from the learning rate which had the highest accuracy. Then I ran it on the test set.

```
Validation accuracy with learning rate 0.1: 0.8330
Validation accuracy with learning rate 0.5: 0.8840
Validation accuracy with learning rate 1.0: 0.9100
Test accuracy: 0.9207 with learning rate 1.0
```