import numpy as np
import re
from collections import Counter
import json
import pickle


# load json file
def load_data(path):
    with open(path) as file:
        data = json.load(file)
    return data


# load intents file json
def load_intents(path):
    data = load_data(path)
    queries_sentences = []
    tags = []
    responses_sentences = []
    for intent in data['intents']:
        for pattern in intent['patterns']:
            queries_sentences.append(pattern)
            tags.append(intent['tag'])
        responses_sentences.append(intent['responses'])
    return queries_sentences, tags, responses_sentences


def preprocess_string(s):
    # preprocess string
    # hàm xử lý chuỗi string s
    # Remove all non-word characters (everything except numbers and letters)
    # xoá các kí tự không phải là từ, ngoại trừ số và chữ cái
    s = re.sub(r"[^\w\s]", '', s)
    # Replace all runs of whitespaces with no space
    # thay thế ccas khoảng trắng bằng không khoảng
    s = re.sub(r"\s+", '', s)
    # replace digits with no space
    # xoá các chữ số thành các khoảng trắng
    s = re.sub(r"\d", '', s)
    # chuyển thành chữ
    s = s.lower()
    return s


def tokenize(queries_sentences):
    vocab_english = []
    for sent in queries_sentences:
        for word in sent.split():
            word = preprocess_string(word)
            if word != "":
                vocab_english.append(word)
    vocab_english = Counter(vocab_english)
    vocab_english = sorted(vocab_english, key=vocab_english.get, reverse=True)
    dictionary_index = {w: i + 4 for i, w in enumerate(vocab_english)}
    dictionary_index['<unk>'], dictionary_index['<pad>'], dictionary_index['<sos>'], dictionary_index[
        '<eos>'] = 0, 1, 2, 3
    return dictionary_index, vocab_english


# encode queries to index
def word_to_index(queries_sentences):
    dictionary_index, _, = tokenize(queries_sentences)
    list_word_to_index = []
    list_index_to_word = []
    punc = '''!()-[]{};:'"\, <>./?@#$%^&*_~'''
    for sent in queries_sentences:
        list_sent_1 = [2]
        list_sent_2 = ['<sos>']
        for word in sent.split():
            if word not in punc and preprocess_string(word) in dictionary_index.keys():
                list_sent_1.append(dictionary_index[preprocess_string(word)])
                list_sent_2.append(preprocess_string(word))
                # Nếu các từ không phải là dấu câu và không thuộc trong tập từ điển
            if word not in punc and preprocess_string(word) not in dictionary_index.keys():
                list_sent_1.append(dictionary_index['<unk>'])
                list_sent_2.append(preprocess_string(word))
        list_sent_1.append(3)
        list_sent_2.append('<eos>')
        list_word_to_index.append(list_sent_1)
        list_index_to_word.append(list_sent_2)
    return list_word_to_index, list_index_to_word


# encode bag of words
def bag_of_words(queries_sentences):
    dictionary_index, vocab_english = tokenize(queries_sentences)
    punc = '''!()-[]{};:'"\, <>./?@#$%^&*_~'''
    list_w_to_index = []
    for sent in queries_sentences:
        list_sent_1 = []
        for word in sent.split():
            if word not in punc and preprocess_string(word) in vocab_english:
                list_sent_1.append(preprocess_string(word))
        list_w_to_index.append(list_sent_1)
    list_word_to_index = []
    for sent in list_w_to_index:
        bag = np.zeros(len(vocab_english), dtype=np.float32)
        for idx, w in enumerate(vocab_english):
            if w in sent:
                bag[idx] = 1
        list_word_to_index.append(bag)
    return list_word_to_index


# encode label
def encode_label(path):
    _, tags, _ = load_intents(path)
    set_tags = sorted(set(tags))
    data_label = []
    for tag in tags:
        label = set_tags.index(tag)
        data_label.append(label)
    return set_tags, data_label


def save_dict(obj, name):
    with open(path + name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def load_dict(name):
    with open(path + name + '.pkl', 'rb') as f:
        return pickle.load(f)


if __name__ == '__main__':
    path = "/Users/nguyenquan/Desktop/mars_project/chatbot_dialoge/simple_chatbot/"
    queries_sentences, tags, responses_sentences = load_intents("intents.json")
    print("queries_sentences:", queries_sentences)
    print(tags)
    print(responses_sentences)
    """
    dictionary_index, vocab_english = tokenize(queries_sentences)
    print(dictionary_index)
    print(vocab_english)
    list_word_to_index, _ = word_to_index(queries_sentences)
    print(list_word_to_index)
    """
    dictionary_index, vocab_english = tokenize(queries_sentences)
    print(vocab_english)
    list_w_to_index = bag_of_words(queries_sentences)
    print(list_w_to_index)
    set_tags, tags = encode_label("intents.json")
    print(set_tags)
    print(tags)
    X_train = np.array(list_w_to_index)
    y_train = np.array(tags)
    data_train_bow = dict()
    data_train_bow["X_train"] = X_train
    data_train_bow["y_train"] = y_train
    data_train_bow["target"] = len(set_tags)
    data_train_bow["set_tags"] = set_tags
    data_train_bow["vocab_english"] = vocab_english
    save_dict(data_train_bow, "data_train_bow")


