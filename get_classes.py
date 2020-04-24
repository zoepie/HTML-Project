#!/usr/bin/python

import sys
from bs4 import BeautifulSoup
from tqdm import tqdm
import os


# read file
def read_file(filename):
    with open(filename, 'r', encoding='utf-8') as file:
        contents = file.read()
        soup = BeautifulSoup(contents, "html.parser")
        return soup


if __name__ == '__main__':
    # get path of txt file into list
    file_path = []
    for root, dirs, files in os.walk("dataset/train/", topdown=True):
        for name in files:
            file_path.append(os.path.join(root, name))
    # loop file path list to read the line by line, count number of class names
    classes = set()
    for i in tqdm(file_path):
        for tag in read_file(i).find_all():
            try:
                if tag['class']:
                    for string in tag['class']:
                        classes.add(string)
            except KeyError:
                continue

    with open("classes.txt", 'w') as f:
        f.write(str(classes))
