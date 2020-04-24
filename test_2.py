import re
from bs4 import BeautifulSoup, NavigableString
import numpy as np
from urllib.request import Request, urlopen
import itertools
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
    test = np.load("dataset/tmp/0/10.npy")
    print(test)

