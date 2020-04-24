#!/usr/bin/python

import numpy as np
from bs4 import BeautifulSoup
from tqdm import tqdm
import json
import torch
import itertools
import os, errno
import sys


def mkdir(path):
    try:
        os.makedirs(path)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise


class ClassNode(object):
    def __init__(self, data):
        self.data = data
        self.children = []

    def add_child(self, child):
        self.children.append(child)


def read_file(filename):
    with open(filename, 'r', encoding='utf-8') as f:
        contents = f.read()
        soup = BeautifulSoup(contents, "html.parser")
        return soup


def find_child(current, node):
    for idx in current.find_all(recursive=False):

        if 'class' in idx.attrs.keys():
            next_node = ClassNode(idx.attrs['class'])
            node.add_child(next_node)
            find_child(idx, next_node)


def get_all_paths(node):
    if len(node.children) == 0:
        return [[node.data]]
    return [
        [node.data] + path for child in node.children for path in get_all_paths(child)
    ]

def main():
    with open('name2idx.json', 'r') as fp:
        name2idx = json.load(fp)

    for i in tqdm(range(192)):
        print('dataset/train2/%d.txt' % i)
        root = ClassNode('root')
        body = read_file('dataset/train2/%d.txt' % i).body
        if not body:
            continue
        find_child(body, root)
        path_list = get_all_paths(root)
        count = 0
        for j in range(len(path_list)):
            for k in itertools.product(*path_list[j][1:]):
                idx_list = [792] + [name2idx[m] for m in k]
                file_path = 'dataset/tmp/%d/' % (count // 1000)
                mkdir(file_path)
                np.save(file_path + '%d.npy' % count, np.array(idx_list))
                count += 1


if __name__ == '__main__':
    main()