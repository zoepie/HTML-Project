#!/usr/bin/python

import sys
from bs4 import BeautifulSoup, NavigableString
from urllib.request import Request, urlopen


def save_file(dir_name, data):
    with open(dir_name, 'rb') as f:
        f.write(data)


if __name__ == "__main__":
    visited_urls = set()
    unvisited_urls = {'https://getbootstrap.com/'}
    counter = -1
    while len(unvisited_urls) != 0:
        url = unvisited_urls.pop()
        print("GET " + url, len(visited_urls), len(unvisited_urls), counter)

        html = urlopen(Request(url, headers={'User-Agent': 'Mozilla/5.0'}))
        soup = BeautifulSoup(html, "html.parser")
        counter += 1
        with open('dataset/test2/' + str(counter) + '.txt', 'wb') as f:
            f.write(soup.prettify().encode('utf8'))
        # dir_name = 'dataset/test2/' + str(counter) + '.txt'
        # soup_body = soup.body.prettify().encode('utf8')
        # save_file(dir_name, str(soup_body))
        # counter += 1
        for element in soup.find_all("a", href=True):
            if element["href"].startswith("/docs/4.4/") \
                    and "https://getbootstrap.com/" + element["href"] not in visited_urls:
                unvisited_urls.add("https://getbootstrap.com/" + element["href"])
        visited_urls.add(url)
