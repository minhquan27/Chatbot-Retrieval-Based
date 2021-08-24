import re
from sklearn.model_selection import train_test_split
from collections import Counter
import numpy as np
import pickle
import os
import glob

base_dir = "/Users/nguyenquan/Desktop/mars_project/chatbot_dialoge/intent/"


def load_text_file(base_dir):
    list_file_txt = os.listdir(base_dir)
    link_file_txt = []
    for file in list_file_txt:
        link_file_txt.append(base_dir + file)
    return list_file_txt, link_file_txt


def read_text_file(file):
    lines = open(file, encoding='utf-8', errors='ignore').read().split('\n')
    basename = os.path.basename(file)
    name_intent = basename.split(".")[0]
    conversation = []
    for line in lines:
        _line = line.split("__eou__")
        _line.pop(-1)
        _line.append(name_intent)
        conversation.append(_line)
    return conversation


def read_text_data(base_dir):
    _, link_file_txt = load_text_file(base_dir)
    data_text = []
    for file in link_file_txt:
        conversation = read_text_file(file)
        data_text = data_text + conversation
    return data_text


def return_data_text(base_dir):
    data_text = read_text_data(base_dir)
    data_text.pop(-1)
    queries_sentences = []
    tags = []
    responses_sentences = []
    for i in range(len(data_text)):
        if len(data_text[i]) == 3:
            queries_sentences.append(data_text[i][0])
            tags.append(data_text[i][2])
            responses_sentences.append(data_text[i][1])
    return queries_sentences, tags, responses_sentences


if __name__ == '__main__':
    base_dir = "/Users/nguyenquan/Desktop/mars_project/chatbot_dialoge/intent/"
    queries_sentences, tags, responses_sentences = return_data_text(base_dir)
    print(queries_sentences[-1])
    print(tags[-1])
    print(responses_sentences[-1])
