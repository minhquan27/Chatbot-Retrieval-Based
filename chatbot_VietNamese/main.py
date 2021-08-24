import numpy as np
import random
import torch
import torch.nn as nn
import pickle
from torch.utils.data import Dataset, DataLoader
from model import NeuralNet
from data import ChatDataset

# load data preprocessed from utils data
with open("data_train_dict" + '.pkl', 'rb') as f:
    data_train_dict = pickle.load(f)
X_train = data_train_dict["X_train"]
y_train = data_train_dict["y_train"]
vocab_vietnamese = data_train_dict["vocab_vietnamese"]
set_tags = data_train_dict["set_tags"]

# hyper-parameters
num_epochs = 200
batch_size = 8
learning_rate = 0.001
input_size = len(X_train[0])
hidden_size = 100
output_size = len(set_tags)

# load dataset
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
dataset = ChatDataset(X_train, y_train)
train_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True, num_workers=0)
model = NeuralNet(input_size, hidden_size, output_size).to(device)
# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)


def accuracy(outputs, labels):
    acc = 0
    for i in range(len(outputs)):
        if outputs[i] == labels[i]:
            acc = acc + 1
    return acc / len(outputs) * 100


# Train the model
for epoch in range(num_epochs):
    model.train()
    for (words, labels) in train_loader:
        words = words.to(device)
        labels = labels.to(dtype=torch.long).to(device)

        # Forward pass
        outputs = model(words)
        # if y would be one-hot, we must apply
        # labels = torch.max(labels, 1)[1]
        loss = criterion(outputs, labels)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        model.eval()
    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')
        with torch.no_grad():
            his_acc = []
            for words, labels in train_loader:
                words = words.to(device)
                labels = labels.to(dtype=torch.long).to(device)

                # Forward pass
                outputs = model(words)
                # print(labels)
                # print(outputs)
                _, predicted = torch.max(outputs.data, 1)
                # print(predicted)
                acc = accuracy(labels, predicted)
                his_acc.append(acc)
        print("accuracy: {} %".format(np.mean(his_acc)))

data = {
    "model_state": model.state_dict(),
    "input_size": input_size,
    "hidden_size": hidden_size,
    "output_size": output_size,
    "set_tags": set_tags,
    "vocab_vietnamese": vocab_vietnamese
}
FILE = "data.pth"
torch.save(data, FILE)

print(f'training complete. file saved to {FILE}')