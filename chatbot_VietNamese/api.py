import torch
from model import NeuralNet
import numpy as np
from utils_data import preprocess_string
from numpy import dot
from numpy.linalg import norm
import pickle

# device config
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def load_pretrain(FILE):
    # load data
    FILE = "data.pth"
    data = torch.load(FILE)
    # load model_state, input_size, hidden_size, output_size, set_tags, vocab_vietnamese
    model_state = data["model_state"]
    input_size = data["input_size"]
    hidden_size = data["hidden_size"]
    output_size = data["output_size"]
    set_tags = data["set_tags"]
    vocab_vietnamese = data["vocab_vietnamese"]
    return model_state, input_size, hidden_size, output_size, set_tags, vocab_vietnamese


def encode(sentences, vocab_vietnamese):
    # encode sentences from user chat
    punc = '''!()-[]{};:'"\, <>./?@#$%^&*_~'''
    X = []
    for word in sentences.split():
        if word not in punc and preprocess_string(word) in vocab_vietnamese:
            X.append(preprocess_string(word))
    bag = np.zeros(len(vocab_vietnamese), dtype=np.float32)
    for idx, w in enumerate(vocab_vietnamese):
        if w in X:
            bag[idx] = 1
    return bag


def cosin_similarity(a, b):
    return dot(a, b) / (norm(a) * norm(b))


def load_responses_sentences():
    with open("data_train_dict" + '.pkl', 'rb') as f:
        data_train_dict = pickle.load(f)
    X_train = data_train_dict["X_train"]
    y_train = data_train_dict["y_train"]
    vocab_vietnamese = data_train_dict["vocab_vietnamese"]
    set_tags = data_train_dict["set_tags"]
    responses_sentences = data_train_dict["responses_sentences"]
    return X_train, y_train, responses_sentences


def choose_best_response(sentences_decode, predicted_tag):
    X_train, y_train, responses_sentences = load_responses_sentences()
    max_score = 0
    list_idex = []
    for i in range(len(y_train)):
        if y_train[i] == predicted_tag:
            a = cosin_similarity(X_train[i], sentences_decode)
            if a > max_score:
                max_score = a
                list_idex.append(i)
    a = list_idex[-1]
    return responses_sentences[a]


def chat(FILE):
    model_state, input_size, hidden_size, output_size, set_tags, vocab_vietnamese = load_pretrain(FILE)
    model = NeuralNet(input_size, hidden_size, output_size).to(device)
    model.load_state_dict(model_state)
    bot_name = "Quân Nguyễn (Bot)"
    print("Bắt đầu trò chuyện! (Gõ kết thúc để dừng)")
    while True:
        # sentence = "do you use credit cards?"
        sentence = input("You: ")
        if sentence == "Kết thúc " or sentence == "kết thúc":
            break
        X = encode(sentence, vocab_vietnamese)
        X = X.reshape(1, X.shape[0])
        X = torch.from_numpy(X).to(device)
        output = model(X)
        _, predicted = torch.max(output, dim=1)
        tag = set_tags[predicted.item()]
        probs = torch.softmax(output, dim=1)
        prob = probs[0][predicted.item()]
        if prob.item() > 0.5:
            X = encode(sentence, vocab_vietnamese)
            print(tag)
            best_response = choose_best_response(X, predicted.item())
            print("{}".format(bot_name), ":", best_response)
        else:
            print("{}: Tôi không hiểu câu hỏi".format(bot_name))


if __name__ == '__main__':
    FILE = "data.pth"
    chat(FILE)
    # X_train, y_train, res = load_responses_sentences()

    '''
    X = np.ones(2013)
    # print(X)
    X_train, y_train, responses_sentences = load_responses_sentences()
    max_score = 0
    list_idex  = []
    for i in range(len(y_train)):
        if y_train[i] == 2:
            a = cosin_similarity(X_train[i], X)
            if a > max_score:
                max_score = a
                list_idex.append(i)
    print(list_idex)
    '''
