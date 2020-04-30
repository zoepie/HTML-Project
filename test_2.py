import re
from bs4 import BeautifulSoup, NavigableString
import numpy as np
from urllib.request import Request, urlopen
import itertools
import matplotlib.pyplot as plt
#
# def append_string(node):
#     if isinstance(node.next, NavigableString):
#         node.next = 'xxx'
#     else:
#         append_string(node.next)
#     # for i in node.children:
#     #     if isinstance(i, NavigableString):
#     #         if i != '\n' and (len(i) > 3 and ('<' not  in i and '>' not in i and '=' not in i)):
#     #             text.append(str(i))
#     #     else:
#     #         append_string(i)

class ClassNode(object):
    def __init__(self, data):
        self.data = data
        self.children = []

    def add_child(self, child):
        self.children.append(child)

class Tree(object):
    "Generic tree node."
    def __init__(self, name='root', children=None):
        self.name = name
        self.children = []
        if children is not None:
            for child in children:
                self.add_child(child)
    def __repr__(self):
        return self.name
    def add_child(self, node):
        assert isinstance(node, Tree)
        self.children.append(node)

def read_file(filename):
    with open(filename, 'r', encoding='utf-8') as f:
        contents = f.read()
        soup = BeautifulSoup(contents, "html.parser")
        return soup


def find_child(current, node):
    for i in current.find_all(recursive=False):
        if 'class' in i.attrs.keys():
            next_node = ClassNode(i.attrs['class'])
            node.add_child(next_node)
            find_child(i, next_node)

def get_all_paths(node):
    if len(node.children) == 0:
        return [[node]]
    return [
        [node] + path for child in node.children for path in get_all_paths(child)
    ]
#

if __name__ == '__main__':
    # a = [[1,2,3],["a","b"]]
    # c= list(itertools.product(*a))
    # # c = [(x, y) for x in a for y in b]
    # # c = [list(zip(x, b)) for x in itertools.permutations(a,len(b))]
    # print(c)
    # root = ClassNode('root')
    # soup = read_file("dataset/train2/72.txt")
    # print(soup)
    # root = ClassNode('root')
    # get_all_paths(root)
    # find_child(read_file('dataset/train2/test.txt').body, root)

    # html = urlopen(Request('https://getbootstrap.com/', headers={'User-Agent': 'Mozilla/5.0'}))
    # soup = BeautifulSoup(html, "html.parser")
    # with open('test.txt', 'wb') as f:
    #     f.write(soup.prettify().encode('utf8'))
    # # print(soup.prettify())
    # with open('test.txt', 'r') as f:
    #     if f.read()[15] == '\n':
    #         print('f.read()[15]')
    # test = np.load("dataset/tmp/0/10.npy")
    # print(test)
    x_SGD = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    # y_SGD = [43.86033034667247, 47.25073940229256, 47.30711297382331, 47.21271432136284, 47.54070595949921, 47.39585976249251, 47.56451526968951, 47.57063619091125, 47.5990603650478, 47.5805874530834]
    y_SGD = [30.4665704691724, 42.17144006075579, 43.85362938643414, 45.2858903033805, 46.05919518878956, 46.1477203823181, 46.83168219389226, 46.9383078135817, 47.29446600155542, 47.40577345628487]
    plt.xticks(np.arange(len(x_SGD)))
    plt.plot(x_SGD, y_SGD, label="SGD")
    x_adam=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    y_adam = [45.69156187996971, 47.44801388030691, 47.66687825020261, 47.51868311833952, 47.49854722825036, 47.70935662692192, 47.67996630341006, 47.611512869349234, 47.757065478246865, 47.727243067664304]
    plt.xticks(np.arange(len(x_adam)))
    plt.plot(x_adam, y_adam, label="Adam")
    plt.xlabel('epoch')
    # Set the y axis label of the current axis.
    plt.ylabel('percentage of accuracy')
    # show a legend on the plot
    plt.legend()
    # Display a figure.
    # plt.savefig("SGDvsAdam")
    plt.show()

