import random
import json
import torch
from model import NeuralNet
from utils_data import preprocess_string
import numpy as np

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

with open('intents.json', 'r') as json_data:
    intents = json.load(json_data)
FILE = "data.pth"
data = torch.load(FILE)

input_size = data["input_size"]
hidden_size = data["hidden_size"]
output_size = data["output_size"]
model_state = data["model_state"]
set_tags = data["set_tags"]
vocab_english = data["vocab_english"]

model = NeuralNet(input_size, hidden_size, output_size).to(device)
model.load_state_dict(model_state)


def encode(sentences, vocab_english):
    punc = '''!()-[]{};:'"\, <>./?@#$%^&*_~'''
    X = []
    for word in sentences.split():
        if word not in punc and preprocess_string(word) in vocab_english:
            X.append(preprocess_string(word))
    bag = np.zeros(len(vocab_english), dtype= np.float32)
    for idx, w in enumerate(vocab_english):
        if w in X:
            bag[idx] = 1
    return bag


bot_name = "Sam"
print("Let's chat! (type 'quit' to exit)")
while True:
    # sentence = "do you use credit cards?"
    sentence = input("You: ")
    if sentence == "quit":
        break

    X = encode(sentence, vocab_english)
    X = X.reshape(1, X.shape[0])
    X = torch.from_numpy(X).to(device)

    output = model(X)
    _, predicted = torch.max(output, dim=1)

    tag = set_tags[predicted.item()]

    probs = torch.softmax(output, dim=1)
    prob = probs[0][predicted.item()]
    if prob.item() > 0.75:
        for intent in intents['intents']:
            if tag == intent["tag"]:
                print(f"{bot_name}: {random.choice(intent['responses'])}")
    else:
        print(f"{bot_name}: I do not understand...")
