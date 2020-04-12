import numpy as np
from bs4 import BeautifulSoup
from tqdm import tqdm
import json


def read_file(filename):
    with open(filename, 'r', encoding='utf-8') as f:
        contents = f.read()
        soup = BeautifulSoup(contents, "html.parser")
        return soup


if __name__ == '__main__':
    with open('name2idx.json', 'r') as fp:
        name2idx = json.load(fp)

    with open('data/0.txt', 'r', encoding='utf-8')as f:
        contents = f.read()
        soup = BeautifulSoup(contents, "html.parser")

    for tag in soup.find_all():
        try:
            if tag['class']:
                tag['class'] = '###' + str(name2idx[' '.join(tag['class'])])
        except KeyError:
            continue
    print(soup)


