from preprocessing_data import return_data_text
import numpy as np
import re
from collections import Counter
import json
import pickle


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
    # chuyển thành chữ thường
    s = s.lower()
    return s


def tokenize(queries_sentences):
    vocab_vietnamese = []
    for sent in queries_sentences:
        for word in sent.split():
            word = preprocess_string(word)
            if word != "":
                vocab_vietnamese.append(word)
    vocab_vietnamese = Counter(vocab_vietnamese)
    vocab_vietnamese = sorted(vocab_vietnamese, key=vocab_vietnamese.get, reverse=True)
    dictionary_index = {w: i + 4 for i, w in enumerate(vocab_vietnamese)}
    dictionary_index['<unk>'], dictionary_index['<pad>'], dictionary_index['<sos>'], dictionary_index[
        '<eos>'] = 0, 1, 2, 3
    return dictionary_index, vocab_vietnamese


# encode bag of words
def bag_of_words(queries_sentences):
    dictionary_index, vocab_vietnamese = tokenize(queries_sentences)
    punc = '''!()-[]{};:'"\, <>./?@#$%^&*_~'''
    list_w_to_index = []
    for sent in queries_sentences:
        list_sent_1 = []
        for word in sent.split():
            if word not in punc and preprocess_string(word) in vocab_vietnamese:
                list_sent_1.append(preprocess_string(word))
        list_w_to_index.append(list_sent_1)
    list_word_to_index = []
    for sent in list_w_to_index:
        bag = np.zeros(len(vocab_vietnamese), dtype=np.float32)
        for idx, w in enumerate(vocab_vietnamese):
            if w in sent:
                bag[idx] = 1
        list_word_to_index.append(bag)
    return list_word_to_index


def label_idx(tags):
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
    base_dir = "/Users/nguyenquan/Desktop/mars_project/chatbot_dialoge/intent/"
    path = "/Users/nguyenquan/Desktop/mars_project/chatbot_dialoge/chatbot_VietNamese/"
    queries_sentences, tags, responses_sentences = return_data_text(base_dir)
    set_tags, data_label = label_idx(tags)
    _, vocab_vietnamese = tokenize(queries_sentences)
    X_train = np.array(bag_of_words(queries_sentences))
    y_train = np.array(data_label)
    data_train_dict = {
        "X_train": X_train,
        "y_train": y_train,
        "vocab_vietnamese": vocab_vietnamese,
        "set_tags": set_tags,
        "responses_sentences": responses_sentences
    }
    save_dict(data_train_dict, "data_train_dict")
