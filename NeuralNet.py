import numpy as np
import math
import GameParser
from matplotlib import pyplot as plt

# Initializes a neural network with 2 layers of sizes input_dim and output_dim
def init_network(neural_net):
    np.random.seed(70)
    num_layers = len(neural_net)

    params_w = []
    params_b = []

    for layer in range(0, num_layers):
        # Uses He-et-al Initialization
        temp_w = np.random.randn(neural_net[layer]["k"], neural_net[layer]["j"]) * np.sqrt(2 / neural_net[layer]["j"]) 
        params_w.append(temp_w)
        temp_b = np.random.randn(neural_net[layer]["k"]) * np.sqrt(2 / neural_net[layer]["j"]) 
        params_b.append(temp_b)

    return params_w, params_b
    
def sigmoid(Z):
    return 1/(1 + np.exp(-Z))

def sigmoid_derivative(Z):
    return sigmoid(Z) * (1 - sigmoid(Z))
 
def forward_prop_layer(prev_A, curr_weights, curr_bias):
    new_vals = np.dot(curr_weights, prev_A) + curr_bias
    return sigmoid(new_vals), new_vals

def error_function(params_w, params_b):
    activations = []
    Z = []
    for layer in range(0, len(neural_net)):
        # Forward prop
        if (layer == 0):
            activations.append(X[i])
        output_A, output_Z = forward_prop_layer(activations[layer], params_w[layer], params_b[layer])
        activations.append(output_A)
        Z.append(output_Z)

    error = -Y[i] * np.log(activations[-1]) - (1 - Y[i]) * np.log(1 - activations[-1])
    return error

def gradient_descent(grads_w, grads_b, alpha):
    return params_w - alpha * grads_w, params_b - alpha * grads_b
    
X = []
Y = []

for game_number in range(1, 10):
    file_name = "games\\" + str(game_number) + ".json"
    print(file_name)
    tempX, tempY = GameParser.populate_inputs(file_name)
    # Only add the game data if the game is valid
    if (tempX is not None):
        X.append(tempX)
        Y.append(tempY)

# Convert X and Y to numpy arrays
X = np.array(X)
Y = np.array(Y)

# Generates all combinations of 1s and 0s for input data
for i in range(0, np.shape(X)[0]):
    num = i
    for j in range(0, np.shape(X)[1]):
        rem = num // (2**(np.shape(X)[1]-1-j))
        X[i][j] = rem
        num -= rem * (2**(np.shape(X)[1]-1-j))

# Generates the correct outputs for the inputs X
for i in range(0, np.shape(Y)[0]):
    for j in range(0, 2):
    # Y[i] = X[i][0] and X[i][1]
        Y[i][j] = (X[i][0] and X[i][1]) or not(X[i][0] or X[i][1])

neural_net = [{"j": 285, "k": 26},
              {"j": 26, "k": 7}]

# Initialize weights and biases
params_w, params_b = init_network(neural_net)

errors = []
num_epochs = 10000
learning_rate = .1
num_examples = np.shape(X)[0]

for epoch in range(0, num_epochs):
    
    # Initialize gradients at 0
    grads_w = []
    grads_b = []
    for i in range(len(neural_net) - 1, -1, -1):
        grads_w.insert(0, np.zeros((neural_net[i]["k"], neural_net[i]["j"])))
    for i in range(len(neural_net) - 1, -1, -1):
        grads_b.insert(0, np.zeros((neural_net[i]["k"])))


    RESULT = [None] * num_examples

    # For each example
    for i in range(0, np.shape(X)[0]):
        activations = None
        activations = []
        output_Z = None
        output_A = None
        Z = []
        for layer in range(0, len(neural_net)):
            # Forward prop
            if (layer == 0):
                activations.append(X[i])
            #                                           30,                 2,30            (1, 2)
            output_A, output_Z = forward_prop_layer(activations[layer], params_w[layer], params_b[layer])
            activations.append(output_A)
            Z.append(output_Z)

        # Back prop 
        # BP 1: Compute error_L = dC/dA * sigmoid_derivative(z_L)

        error_last = None
        error_last = ((-Y[i] / output_A) + (1 - Y[i]) / (1 - output_A)) * sigmoid_derivative(output_Z)
        
        if (error_last.ndim == 1):
            error_last = np.expand_dims(error_last, axis = 1)
        temp_activations = activations[-2]
        if (activations[-2].ndim == 1):
            temp_activations = np.expand_dims(temp_activations, axis = 1)
        weight_gradient = np.dot(error_last, temp_activations.T)
        grads_w[len(neural_net) - 1] = grads_w[len(neural_net) - 1] + weight_gradient
        error_next = error_last

        for k in range(0, len(grads_b[len(neural_net) - 1])):
            grads_b[len(neural_net) - 1][k] = grads_b[len(neural_net) - 1][k] + error_next[k][0]
        
        # BP 2/3: Backprop the error: For each l from L, L-1, ..., 2, compute error_l = np.dot(weights_l+1.T, error_l+1) * sigmoid_derivative(z_curr)
        for j in range(len(neural_net) - 1, 0, -1):
            # Ensure that error_next is 2 dimensions when passed to error_curr
            if (error_next.ndim == 1):
                temp0 = np.copy(error_next)
                temp0 = np.expand_dims(temp0, axis = 0)
                error_curr = np.dot(temp0, params_w[j]) * sigmoid_derivative(Z[j - 1])
            else:
                #                                  30,2            2,1                   30,
                error_curr = np.multiply(np.dot(params_w[j].T, error_next), np.expand_dims(sigmoid_derivative(Z[j - 1]), axis = 1))
            temp1 = np.copy(activations[j-1])
            if (temp1.ndim < 2):
                temp1 = np.expand_dims(temp1, axis = 0)

            weight_gradient = np.dot(error_curr, temp1)
            grads_w[j-1] = grads_w[j-1] + weight_gradient

            for k in range(0, len(grads_b[j-1])):
                grads_b[j-1][k] = grads_b[j-1][k] + error_curr[k][0]

            error_next = error_curr

        RESULT[i] = output_A

    # Divide gradients by number of examples
    grads_w = np.asarray(grads_w)
    grads_w /= np.shape(X)[0]
    grads_b = np.asarray(grads_b)
    grads_b /= np.shape(X)[0]

    # Perform gradient descent
    params_w, params_b = gradient_descent(grads_w, grads_b, learning_rate)

    # Update error values for graphing
    errors.append(error_function(params_w, params_b))

# Results of final epoch
print("======== FINAL EPOCH RESULTS ========")
# for i in range(len(RESULT)):
#     print("Test #" + str(i) + " | Pred: " + " %.4f " % RESULT[i] + " | Real: " + str(Y[i]))

# Graph of Error vs Iterations
plt.plot(errors)
plt.ylabel('Error')
plt.xlabel('Epoch')
plt.show()