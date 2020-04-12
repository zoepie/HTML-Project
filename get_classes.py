import numpy as np
from bs4 import BeautifulSoup
from tqdm import tqdm
import os

# get path of txt file into list
file_path = []
for root, dirs, files in os.walk("dataset/train/", topdown=True):
   for name in files:
      file_path.append(os.path.join(root, name))
#
#read file
def read_file(filename):
    with open(filename, 'r', encoding='utf-8')as f:
        contents = f.read()
        soup = BeautifulSoup(contents, "html.parser")
        return soup

#loop file path list to read the line by line, count number of class names
# classes = set()
classes = set()
for i in tqdm(file_path):
    # read_file(i)
    for tag in read_file(i).find_all():
        try:
            if tag['class']:
                # txt = ' ' .join(tag['class'])
                # classes.add(txt)
                for string in tag['class']:
                    classes.add(string)
                    # a_class = []
                    # for j in string:
                    #     a_class.append(j)

        except:
            continue
    # for element in read_file(i).find_all("class"):
    #     classes.add(a_class)
    #     for value in element["class"]:
    #         a_class = []
    #         for cla in value:
    #             a_class.append(cla)
    #     print(classes)
with open("classes.txt", 'w') as f:
    f.write(str(classes))


