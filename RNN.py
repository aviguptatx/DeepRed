import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os
import numpy as np
from torch.autograd import Variable
import matplotlib.pyplot as plt
import gc

# Garbage collect
gc.collect()

# Set CUDA device
cuda = torch.device('cuda')

# Empty the CUDA cache
with torch.no_grad():
    torch.cuda.empty_cache()
torch.cuda.empty_cache()

# Recurrent Neural Network


class RNN(nn.Module):
    def __init__(self, n_inputs, n_neurons, X_in, Y_in):
        super(RNN, self).__init__()
        self.rnn = nn.RNNCell(n_inputs, n_neurons)
        self.n_neurons = n_neurons
        # initialize last hidden state
        self.hx = torch.randn(len(X_in), n_neurons,
                              device=cuda) * np.sqrt(2 / n_inputs)
        self.Xa = X_in
        self.Ya = Y_in

    # Forward propogation
    def forward(self, games):
        output = []
        # for each time step
        for gov_number in range(len(games[0])):
            self.hx = 3 * \
                F.softmax(
                    self.rnn(games[:, gov_number], self.hx).cuda(), dim=1)
            output.append(self.hx)
        return output, self.hx

    def init_hidden(self):
        self.hx = torch.zeros(len(self.Xa), self.n_neurons).cuda()


# Number of nodes in input layer
N_INPUT = 91

# Number of nodes in hidden layer
N_NEURONS = 7

# Training set
X, Y = populate_inputs(1000, 5000, 1)

# Testing set (validation)
test_X, test_Y = populate_inputs(500, 1000, 1)

# Convert to tensors
X = torch.as_tensor(X).cuda()
Y = torch.as_tensor(Y).cuda()
test_X = torch.as_tensor(test_X).cuda()
test_Y = torch.as_tensor(test_Y).cuda()

# Number of epochs
N_EPOCHS = 1000

# Training model
train_model = RNN(N_INPUT, N_NEURONS, X, Y).cuda()
train_model.cuda()

# Testing model
test_model = RNN(N_INPUT, N_NEURONS, test_X, test_Y).cuda()
test_model.cuda()

# Set the loss function to cross entropy and use optimizer
criterion = nn.CrossEntropyLoss().cuda()
optimizer = optim.Adam(train_model.parameters(), lr=.03)

# Contains the errors
train_seat_errors = []
test_seat_errors = []

# Training loop
for epoch in range(N_EPOCHS):

    print("Epoch " + str(epoch))
    train_running_loss = 0.0
    train_model.train().cuda()
    test_model.train().cuda()

    # zero the parameter gradients
    optimizer.zero_grad()

    # reset hidden states
    train_model.init_hidden()
    test_model.init_hidden()

    # Run the model, collect states (store them)
    all_states, last_states = train_model(
        torch.as_tensor(X, dtype=torch.float32).cuda())
    last_states = last_states.cuda()

    # Separates probabilities of liberal and fascist
    output = torch.zeros(len(X), 2, 7).cuda()
    output[:, 0, :] = 1 - last_states
    output[:, 1, :] = last_states

    # Calculate loss and backpropogate
    loss = criterion(output, torch.as_tensor(Y).cuda())
    loss.backward(retain_graph=True)
    optimizer.step()

    # Evaluate the training set errors and store them
    train_seat_error = torch.sum(
        torch.abs(torch.as_tensor(Y).cuda() - last_states)).cuda() / len(Y)
    train_seat_errors.append(train_seat_error)

    train_model.eval().cuda()

    # Transfer weights of training model to the testing model for testing
    test_model.rnn.weight_hh = train_model.rnn.weight_hh
    test_model.rnn.weight_ih = train_model.rnn.weight_ih

    # Run the testing model on the test data set and store predictions
    _, prediction = test_model.forward(
        torch.as_tensor(test_X, dtype=torch.float32).cuda())
    prediction = prediction.cuda()

    # Store the errors for the testing set
    test_seat_error = torch.sum(torch.abs(torch.as_tensor(
        test_Y).cuda() - prediction)).cuda() / len(test_Y)
    test_seat_errors.append(test_seat_error)

    test_model.eval().cuda()

    # Reset in case this is causing a Google Colab memory issue
    all_states = None
    last_states = None
    output = None
    prediction = None

    # train_running_loss += loss.detach().item()

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
