#!/usr/bin/python

import sys
from bs4 import BeautifulSoup, NavigableString
from urllib.request import Request, urlopen


def save_file(dir_name, data):
    with open(dir_name, 'w', encoding='utf-8') as f:
        f.write(data)

if __name__ == "__main__":
    visited_urls = set()
    unvisited_urls = {'https://mdbootstrap.com/plugins/jquery/gallery/'}
    counter = 0
    while len(unvisited_urls) != 0:
        try:
            url = unvisited_urls.pop()
            print("GET " + url, len(visited_urls), len(unvisited_urls), counter)

            html = urlopen(Request(url, headers={'User-Agent': 'Mozilla/5.0'}))
            soup = BeautifulSoup(html, "html.parser")
            dir_name = 'dataset/test/' + str(counter) + '.txt'
            save_file(dir_name, str(soup))
            counter += 1
            for element in soup.find_all("a", href=True):
                # for link in soup.find_all('img',id = False):
                # try:

                if element["href"] not in visited_urls :
                        unvisited_urls.add(element["href"])

                # except error.HTTPError:
                #     continue

            visited_urls.add(url)
        except ValueError:
            continue
