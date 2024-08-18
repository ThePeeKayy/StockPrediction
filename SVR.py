import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Generate sample sine wave data
X = np.linspace(0, 50, 500)
y = np.sin(X)

# Prepare data for LSTM
def create_inout_sequences(input_data, seq_length):
    inout_seq = []
    L = len(input_data)
    for i in range(L-seq_length):
        train_seq = input_data[i:i+seq_length]
        train_label = input_data[i+seq_length:i+seq_length+1]
        inout_seq.append((train_seq, train_label))
    return inout_seq

seq_length = 10
train_data = create_inout_sequences(y, seq_length)

# Convert data to PyTorch tensors
train_tensors = [(torch.FloatTensor(seq).unsqueeze(1).to(device), torch.FloatTensor(label).to(device)) for seq, label in train_data]

# Define the LSTM network
class LSTM(nn.Module):
    def __init__(self, input_size=1, hidden_layer_size=50, output_size=1):
        super(LSTM, self).__init__()
        self.hidden_layer_size = hidden_layer_size
        self.lstm = nn.LSTM(input_size, hidden_layer_size)
        self.linear = nn.Linear(hidden_layer_size, output_size)
        self.hidden_cell = (torch.zeros(1, 1, self.hidden_layer_size).to(device),
                            torch.zeros(1, 1, self.hidden_layer_size).to(device))

    def forward(self, input_seq):
        lstm_out, self.hidden_cell = self.lstm(input_seq.view(len(input_seq), 1, -1), self.hidden_cell)
        predictions = self.linear(lstm_out.view(len(input_seq), -1))
        return predictions[-1]

# Instantiate the model, define the loss function and the optimizer
model = LSTM().to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Train the model
num_epochs = 100
for epoch in range(num_epochs):
    for seq, labels in train_tensors:
        optimizer.zero_grad()
        model.hidden_cell = (torch.zeros(1, 1, model.hidden_layer_size).to(device),
                             torch.zeros(1, 1, model.hidden_layer_size).to(device))
        y_pred = model(seq)
        single_loss = criterion(y_pred, labels)
        single_loss.backward()
        optimizer.step()

    if epoch % 10 == 0:
        print(f'Epoch {epoch} loss: {single_loss.item()}')

# Predict future values
model.eval()
future_predictions = []
current_seq = torch.FloatTensor(y[-seq_length:]).unsqueeze(1).to(device)

for _ in range(100):
    with torch.no_grad():
        model.hidden_cell = (torch.zeros(1, 1, model.hidden_layer_size).to(device),
                             torch.zeros(1, 1, model.hidden_layer_size).to(device))
        future_pred = model(current_seq)
        future_predictions.append(future_pred.item())
        current_seq = torch.cat((current_seq[1:], future_pred.unsqueeze(0)))

# Plot the results
plt.figure(figsize=(12, 6))
plt.plot(X, y, label='Actual Values', color='blue')
plt.plot(np.linspace(50, 60, 100), future_predictions, label='LSTM Predicted Values', color='red')
plt.legend()
plt.show()
