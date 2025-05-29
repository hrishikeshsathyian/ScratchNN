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

## Overview of Forward Propagation 
The network consists of:

Input layer X: Each input image is flattened into a 784-dimensional column vector (28×28 pixels).
Hidden layer A1: 10 neurons with ReLU activation.
Output layer A2: 10 neurons with softmax activation, each corresponding to a digit (0–9).

I chose ReLU as an activation function for the first hidden layer instead of other common activations such as tanh and sigmoid, so as to avoid the issue of vanishing gradients as well as to be able to reap the benefits of sparse activation. That is, ReLU "turns off" the neurons that have negative inputs and hence prevents overfitting.

For the output layer, I chose softmax as an activation function because turning raw inputs into probabilities was well suited for multi-class classification.

Overall, a quick overview of the math of forward propagation would be as follows:

Z<sup>1</sup>  = W<sup>1</sup>X + b1
A<sup>1</sup> = ReLU(Z<sup>1</sup>)
Z<sup>2</sup>  = W<sup>2</sup>A<sup>1</sup> + b2
A<sup>2</sup> = softmax(Z<sup>2</sup>)


## Implementation (Forward Propagation)
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
    Z_stable = Z - np.max(Z, axis=0, keepdims=True)
    exp_Z = np.exp(Z_stable)
    return exp_Z / np.sum(exp_Z, axis=0, keepdims=True)

def forward_propagation(X, W1, b1, W2, b2): 
    Z1 = W1 @ X + b1 
    A1 = ReLU(Z1)
    Z2 = W2 @ A1 + b2
    A2 = softmax(Z2)
    return Z1, A1, Z2, A2
```


## Implementation (Back Propagation)