import numpy as np
import random
import torch
import torch.nn as nn
import pickle
from torch.utils.data import Dataset, DataLoader
from model import NeuralNet

# load data processed
with open("data_train_bow" + '.pkl', 'rb') as f:
    data_train_bow = pickle.load(f)
X_train = data_train_bow["X_train"]
y_train = data_train_bow["y_train"]
len_target = data_train_bow["target"]
# print(X_train)
# print(y_train)
# print(len_target)

# hyper-parameters
num_epochs = 1000
batch_size = 8
learning_rate = 0.001
input_size = len(X_train[0])
hidden_size = 8
output_size = len_target


# Class Dataset
class ChatDataset(Dataset):

    def __init__(self):
        self.n_samples = len(X_train)
        self.x_data = X_train
        self.y_data = y_train

    # support indexing such that dataset[i] can be used to get i-th sample
    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    # we can call len(dataset) to return the size
    def __len__(self):
        return self.n_samples


def accuracy(outputs, labels):
    acc = 0
    for i in range(len(outputs)):
        if outputs[i] == labels[i]:
            acc = acc + 1
    return acc / len(outputs) * 100


dataset = ChatDataset()
train_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True, num_workers=0)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = NeuralNet(input_size, hidden_size, output_size).to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

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
    if (epoch + 1) % 100 == 0:
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
    "set_tags": data_train_bow["set_tags"],
    "vocab_english": data_train_bow["vocab_english"]
}
FILE = "data.pth"
torch.save(data, FILE)

print(f'training complete. file saved to {FILE}')
