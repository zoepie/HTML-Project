import numpy as np
from bs4 import BeautifulSoup
from tqdm import tqdm
import os
import json


# read file
def read_file(filename):
    with open(filename, 'r', encoding='utf-8')as f:
        contents = f.read()
        soup = BeautifulSoup(contents, "html.parser")
        return soup


class ClassNameError(KeyError):
    pass


class RelationError(Exception):
    pass


# def freq_calculation(train_dir):
#     # get path of txt file into list
#     file_path = []
#     for root, dirs, files in os.walk(train_dir, topdown=True):
#         for name in files:
#             file_path.append(os.path.join(root, name))
#
#     name2idx = {}
#     idx2name = {}
#     freq_matrix = np.zeros(shape=(1174, 1174), dtype=np.int64)
#
#     with open("classes.txt", 'r') as f:
#         file = f.readlines()[0].strip('{}').strip('\'').split('\', \'')
#         counter = 0
#         for i in file:
#             name2idx[i] = counter
#             idx2name[counter] = i
#             counter += 1
#
#     for i in tqdm(file_path):
#         if read_file(i).body:
#             for tag in read_file(i).body.find_all():
#                 try:
#                     if tag['class']:
#                         parent_name = ' '.join(tag['class'])
#                         for child in tag.children:
#                             try:
#                                 if child['class']:
#                                     child_name = ' '.join(child['class'])
#                                     freq_matrix[name2idx[parent_name], name2idx[child_name]] += 1
#                             except (KeyError, TypeError):
#                                 continue
#                 except KeyError:
#                     continue
#
#     with open('name2idx.json', 'w') as fp:
#         json.dump(name2idx, fp)
#     with open('idx2name.json', 'w') as fp:
#         json.dump(idx2name, fp)
#     np.save('freq_matrix.npy', freq_matrix)

def freq_calculation(train_dir):
    # get path of txt file into list
    file_path = []
    for root, dirs, files in os.walk(train_dir, topdown=True):
        for name in files:
            file_path.append(os.path.join(root, name))

    name2idx = {}
    idx2name = {}
    freq_matrix = np.zeros(shape=(792, 792), dtype=np.int64)

    with open("classes.txt", 'r') as f:
        file = f.readlines()[0].strip('{}').strip('\'').split('\', \'')
        counter = 0
        for i in file:
            name2idx[i] = counter
            idx2name[counter] = i
            counter += 1

    for i in tqdm(file_path):
        if read_file(i).body:
            for tag in read_file(i).body.find_all():
                try:
                    if tag['class']:
                        for i in tag['class']:
                            parent_name = i
                            for child in tag.children:
                                try:
                                    if child['class']:
                                        for j in child['class']:
                                            child_name = j
                                            freq_matrix[name2idx[parent_name], name2idx[child_name]] += 1
                                except (KeyError, TypeError):
                                    continue
                except KeyError:
                    continue

    with open('name2idx.json', 'w') as fp:
        json.dump(name2idx, fp)
    with open('idx2name.json', 'w') as fp:
        json.dump(idx2name, fp)
    np.save('freq_matrix.npy', freq_matrix)


# def model_test(test_dir, name2idx, freq_matrix):
#     test_path = []
#     for root, dirs, files in os.walk(test_dir, topdown=True):
#         for name in files:
#             test_path.append(os.path.join(root, name))
#
#     for i in tqdm(test_path):
#         if read_file(i).body:
#             for tag in read_file(i).body.find_all():
#                 try:
#                     if tag['class']:
#                         parent_name = ' '.join(tag['class'])
#                         if parent_name not in name2idx:
#                             raise ClassNameError("no", parent_name, "in classes")
#                         else:
#                             for child in tag.children:
#                                 try:
#                                     if child['class']:
#                                         child_name = ' '.join(child['class'])
#                                         if child_name not in name2idx:
#                                             raise ClassNameError("no", child_name, "in classes")
#                                         elif freq_matrix[name2idx[parent_name], name2idx[child_name]] == 0:
#                                             raise RelationError(parent_name, child_name, "has no relation")
#                                         else:
#                                             print('Found valid class relation.')
#                                 except (KeyError, TypeError):
#                                     continue
#                 except KeyError:
#                     continue

def model_test(test_dir, name2idx, freq_matrix):
    counter = 0
    test_path = []
    for root, dirs, files in os.walk(test_dir, topdown=True):
        for name in files:
            test_path.append(os.path.join(root, name))

    for i in tqdm(test_path):
        if read_file(i).body:
            for tag in read_file(i).body.find_all():
                try:
                    if tag['class']:
                        for i in tag['class']:
                            parent_name = i
                        if parent_name not in name2idx:
                            raise ClassNameError("no", parent_name, "in classes")
                        else:
                            for child in tag.children:
                                try:
                                    if child['class']:
                                        for j in child['class']:
                                            child_name = j
                                        if child_name not in name2idx:
                                            raise ClassNameError("no", child_name, "in classes")
                                        elif freq_matrix[name2idx[parent_name], name2idx[child_name]] == 0:
                                            raise RelationError(parent_name, child_name, "has no relation")
                                        else:
                                            counter +=1
                                            # print('Found valid class relation.')
                                except (KeyError, TypeError):
                                    continue
                except KeyError:
                    continue

if __name__ == '__main__':
    freq_calculation("dataset/train/")
    with open('name2idx.json', 'r') as f:
        n2i = json.load(f)
    fm = np.load('freq_matrix.npy')
    model_test('dataset/test/', n2i, fm)
