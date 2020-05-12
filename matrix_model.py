import numpy as np
from bs4 import BeautifulSoup
from tqdm import tqdm
import os
import json


# read file
def read_file(filename, split):
    with open('dataset/%s/%d.txt' % (split, filename), 'r', encoding='utf-8') as file:
        contents = file.read()
        soup = BeautifulSoup(contents, "html.parser")
        return soup


def freq_calculation():
    name2idx = {}
    idx2name = {}
    with open("classes.txt", 'r') as file:
        for i, line in enumerate(file.readlines()):
            name2idx[line.strip()] = i
            idx2name[i] = line.strip()
    freq_matrix = np.zeros(shape=(len(name2idx), len(name2idx)), dtype=np.int64)

    for i in tqdm(range(191)):
        if read_file(i, 'train').body:
            for tag in read_file(i, 'train').body.find_all():
                try:
                    if tag['class']:
                        for p_cls in tag['class']:
                            for child in tag.children:
                                try:
                                    if child['class']:
                                        for c_cls in child['class']:
                                            freq_matrix[name2idx[p_cls], name2idx[c_cls]] += 1
                                except (KeyError, TypeError):
                                    continue
                except KeyError:
                    continue
    with open('name2idx.json', 'w') as fp:
        json.dump(name2idx, fp)
    with open('idx2name.json', 'w') as fp:
        json.dump(idx2name, fp)
    np.save('freq_matrix.npy', freq_matrix)


def model_test(name2idx, freq_matrix):
    for i in tqdm(range(3)):
        if read_file(i, 'test').body:
            for tag in read_file(i, 'test').body.find_all():
                try:
                    if tag['class']:
                        for p_cls in tag['class']:
                            if p_cls not in name2idx.keys():
                                print("ClassNameError:\t\tno %s class" % p_cls)
                            else:
                                for child in tag.children:
                                    try:
                                        if child['class']:
                                            for c_cls in child['class']:
                                                if c_cls not in name2idx.keys():
                                                    print("ClassNameError:\t\tno %s class" % c_cls)
                                                elif freq_matrix[name2idx[p_cls], name2idx[c_cls]] == 0:
                                                    print(
                                                        "RelationError:\t\t%s and %s has no relation" % (p_cls, c_cls))
                                    except (KeyError, TypeError):
                                        continue
                except KeyError:
                    continue


if __name__ == '__main__':
    freq_calculation()

    with open('name2idx.json', 'r') as f:
        n2i = json.load(f)
    fm = np.load('freq_matrix.npy')
    # model_test(n2i, fm)
    # i = n2i["pt-4"]
    # j = n2i['container-fluid']
    # print(fm[i][j])


