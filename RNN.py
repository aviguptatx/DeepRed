import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os
import numpy as np
from torch.autograd import Variable
import matplotlib.pyplot as plt
import gc
import RNNParser
import sys

# Prints entire numpy arrays
np.set_printoptions(threshold=sys.maxsize)

# Deterministic randomness
torch.manual_seed(0)

# Garbage collect
gc.collect()

# Enables device agnostic tensor creation
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Empty the CUDA cache
with torch.no_grad():
    torch.cuda.empty_cache()
torch.cuda.empty_cache()

# Recurrent Neural Network
class RNN(nn.Module):
    def __init__(self, n_inputs, n_neurons, X_in, Y_in):
        super(RNN, self).__init__()
        self.rnn = nn.RNN(n_inputs, n_neurons)
        self.n_neurons = n_neurons
        self.X = X_in
        self.Y = Y_in
        self.FC = nn.Linear(self.n_neurons, 7)
    def forward(self, games):
        # Permute the games tensor dimensions in the correct order
        games_permuted = games.permute(1, 0, 2).to(device)
        # Initialize the hidden state with all zeroes
        self.init_hidden()
        # Calculate values of hidden states
        _, self.hx = self.rnn(games_permuted, self.hx)
        # Run the states through the sigmoid function
        sigmoid = nn.Sigmoid()
        # Fully connected output layer
        out = sigmoid(self.FC(self.hx)).to(device)
        return out
    def init_hidden(self):
        self.hx = torch.zeros(1, len(self.X), self.n_neurons).to(device)

# Number of nodes in input layer
N_INPUT = 91

# Number of nodes in hidden layer
N_NEURONS = 80

# Training set
X, Y, _ = RNNParser.populate_inputs(3, 200, 1)

# Testing set (validation)
test_X, test_Y, test_game_numbers = RNNParser.populate_inputs(200, 400, 1)

# Convert to tensors
X = torch.as_tensor(X, dtype=torch.float32).to(device)
Y = torch.as_tensor(Y).to(device)
test_X = torch.as_tensor(test_X, dtype=torch.float32).to(device)
test_Y = torch.as_tensor(test_Y).to(device)

# Hyperparameters
N_EPOCHS = 1000
LEARNING_RATE = .15
LAMBDA_ = .0012

# Training model
train_model = RNN(N_INPUT, N_NEURONS, X, Y).to(device)
train_model.to(device)

# Testing model
test_model = RNN(N_INPUT, N_NEURONS, test_X, test_Y).to(device)
test_model.to(device)

# Set the loss function to binary cross entropy and use optimizer
criterion = nn.BCELoss().to(device)
optimizer = optim.SGD(train_model.parameters(), lr=LEARNING_RATE, weight_decay=LAMBDA_)

# Contains the errors
train_seat_errors = []
test_seat_errors = []
rounded_test_seat_errors = []
sorted_test_seat_errors = []

# Training loop
for epoch in range(N_EPOCHS):

    # Print epoch number every 100 epochs
    if epoch % 100 == 0:
        print("Epoch " + str(epoch))
    
    train_model.train()
    test_model.train()

    # zero the parameter gradients
    optimizer.zero_grad()

    # reset hidden states
    train_model.init_hidden()
    test_model.init_hidden()

    # Run the model, collect states (store them)
    last_states = train_model(X)

    # Calculate loss and backpropogate
    loss = criterion(last_states.float(), Y.float())

    loss.backward()
    optimizer.step()

    # Evaluate the training set errors and store them
    train_seat_error = torch.sum(torch.abs(Y - last_states)) / len(Y)
    train_seat_errors.append(train_seat_error.detach().item())
    
    train_model.eval()

    # Transfer weights of training model to the testing model for testing
    torch.save(train_model.state_dict(), 'parameters')
    test_model.load_state_dict(torch.load('parameters'))

    # Run the testing model on the test data set and store predictions
    prediction = test_model(test_X)

    # Store the errors for the testing set
    test_seat_error = torch.sum(torch.abs(test_Y - prediction)) / len(test_Y)
    test_seat_errors.append(test_seat_error.detach().item())

    # Store the errors for the testing set, but round to either 0 or 1
    rounded_test_seat_error = torch.sum(torch.abs(test_Y - torch.round(prediction))) / len(test_Y)
    rounded_test_seat_errors.append(rounded_test_seat_error.detach().item())

    # Store these metrics for the middle and last epoch
    if (epoch + 1) % (N_EPOCHS / 2) == 0:
        # Set the three highest predictions to 1s and the rest to 0s
        _, sorted_test_seat_indices = torch.sort(prediction)
        sorted_prediction = torch.zeros(prediction.size()).to(device)
        for i in range(5, 7):
            for game in range(len(test_X)):
                sorted_prediction[0][game][sorted_test_seat_indices[0][game][i]] = 1
        sorted_test_seat_error = torch.sum(torch.abs(test_Y - sorted_prediction)) / len(test_Y)
        sorted_test_seat_errors.append(sorted_test_seat_error.detach().item())

    test_model.eval() 
        
# Print training set results
for game_index in range(len(test_X)):
    print("Game #" + str(test_game_numbers[game_index]))
    print("Pred: " + str(prediction[0][game_index]) + "\nReal: " + str(test_Y[game_index]))

# Print metrics for middle epoch
print("Middle")
print("Train: " + str(train_seat_errors[int(-N_EPOCHS / 2)]))
print("Test: " + str(test_seat_errors[int(N_EPOCHS / 2)]))
print("Test rounded: " + str(rounded_test_seat_errors[int(N_EPOCHS / 2)]))
print("Test sorted: " + str(sorted_test_seat_errors[-2]))

# Print metrics for last epoch
print("\nEnd")
print("Train: " + str(train_seat_errors[-1]))
print("Test: " + str(test_seat_errors[-1]))
print("Test rounded: " + str(rounded_test_seat_errors[-1]))
print("Test sorted: " + str(sorted_test_seat_errors[-1]))

# Graph Train Error
plt.subplot(3, 1, 1)
plt.title("Train Error")
plt.plot(train_seat_errors)

# Graph testing / validation set error
plt.subplot(3, 1, 2)
plt.title("Validation Error")
plt.plot(test_seat_errors)

# Graph testing / validation set error
plt.subplot(3, 1, 3)
plt.title("Rounded Validation Error")
plt.plot(rounded_test_seat_errors)

plt.show()