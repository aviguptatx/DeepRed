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
    def forward(self, games):
        # Permute the games tensor dimensions in the correct order
        games_permuted = games.permute(1, 0, 2).to(device)
        # Initialize the hidden state with all zeroes
        self.init_hidden()
        # Calculate values of hidden states
        states, hidden_states = self.rnn(games_permuted, self.hx)
        # Run the states through the sigmoid function
        sigmoid = nn.Sigmoid()
        states, hidden_states = sigmoid(states), sigmoid(hidden_states)
        # # Softmax the results
        # return 3 * F.softmax(torch.as_tensor(states), dim=2), 3 * F.softmax(torch.as_tensor(hidden_states.view(-1, self.n_neurons)), dim=1)
        return torch.as_tensor(states), torch.as_tensor(hidden_states)
    def init_hidden(self):
        self.hx = torch.zeros(1, len(self.X), self.n_neurons).to(device)

# Number of nodes in input layer
N_INPUT = 91

# Number of nodes in hidden layer
N_NEURONS = 7

# Training set
X, Y, _ = RNNParser.populate_inputs(2, 500, 1)

# Testing set (validation)
test_X, test_Y, test_game_numbers = RNNParser.populate_inputs(500, 1000, 1)

# Convert to tensors
X = torch.as_tensor(X, dtype=torch.float32).to(device)
Y = torch.as_tensor(Y).to(device)
test_X = torch.as_tensor(test_X, dtype=torch.float32).to(device)
test_Y = torch.as_tensor(test_Y).to(device)

# Hyperparameters
N_EPOCHS = 1000
LEARNING_RATE = .03

# Training model
train_model = RNN(N_INPUT, N_NEURONS, X, Y).to(device)
train_model.to(device)

# Testing model
test_model = RNN(N_INPUT, N_NEURONS, test_X, test_Y).to(device)
test_model.to(device)

# Set the loss function to binary cross entropy and use optimizer
criterion = nn.BCELoss().to(device)
optimizer = optim.Adam(train_model.parameters(), lr=LEARNING_RATE)

# Contains the errors
train_seat_errors = []
test_seat_errors = []

# Training loop
for epoch in range(N_EPOCHS):

    print("Epoch " + str(epoch))
    train_model.train()
    test_model.train()

    # zero the parameter gradients
    optimizer.zero_grad()

    # reset hidden states
    train_model.init_hidden()
    test_model.init_hidden()

    # Run the model, collect states (store them)
    all_states, last_states = train_model(X)

    # Calculate loss and backpropogate
    loss = criterion(last_states.float(), Y.float())
    loss.backward()

    # loss.backward(retain_graph=True)
    optimizer.step()

    # Evaluate the training set errors and store them
    train_seat_error = torch.sum(torch.abs(Y - last_states)) / len(Y)
    train_seat_errors.append(train_seat_error)

    train_model.eval()

    # Transfer weights of training model to the testing model for testing
    torch.save(train_model.state_dict(), 'parameters')
    test_model.load_state_dict(torch.load('parameters'))

    # Run the testing model on the test data set and store predictions
    _, prediction = test_model(test_X)

    # Store the errors for the testing set
    test_seat_error = torch.sum(torch.abs(test_Y - prediction)) / len(test_Y)
    test_seat_errors.append(test_seat_error)

    test_model.eval()

for game_index in range(len(test_X)):
    print("Game #" + str(test_game_numbers[game_index]))
    print("Pred: " + str(prediction[0][game_index]) + "\nReal: " + str(test_Y[game_index]))

# Graph Train Error
plt.subplot(2, 1, 1)
plt.title("Train Error")
plt.plot(train_seat_errors)

# Graph testing / validation set error
plt.subplot(2, 1, 2)
plt.title("CV Error")
plt.plot(test_seat_errors)

plt.show()

# Reset in case this is causing a Google Colab memory issue
X = None
Y = None
test_X = None
test_Y = None