#!/usr/bin/python

from bs4 import BeautifulSoup
from urllib.request import Request, urlopen


def crawl_train():
    visited_urls = set()
    unvisited_urls = {'https://getbootstrap.com/'}
    counter = 0
    while len(unvisited_urls) != 0:
        url = unvisited_urls.pop()
        print("GET " + url, len(visited_urls), len(unvisited_urls), counter)
        html = urlopen(Request(url, headers={'User-Agent': 'Mozilla/5.0'}))
        soup = BeautifulSoup(html, "html.parser")
        with open('dataset/train/' + str(counter) + '.txt', 'wb') as f:
            f.write(soup.prettify().encode('utf8'))
        counter += 1
        for element in soup.find_all("a", href=True):
            if element["href"].startswith("/docs/4.4/") \
                    and "https://getbootstrap.com/" + element["href"] not in visited_urls:
                unvisited_urls.add("https://getbootstrap.com/" + element["href"])
        visited_urls.add(url)


def crawl_test():
    visited_urls = set()
    unvisited_urls = {'https://mdbootstrap.com/plugins/jquery/gallery/'}
    counter = 0
    while len(unvisited_urls) != 0:
        url = unvisited_urls.pop()
        print("GET " + url, len(visited_urls), len(unvisited_urls), counter)
        html = urlopen(Request(url, headers={'User-Agent': 'Mozilla/5.0'}))
        soup = BeautifulSoup(html, "html.parser")
        with open('dataset/test/' + str(counter) + '.txt', 'wb') as f:
            f.write(soup.prettify().encode('utf8'))
        counter += 1
        for element in soup.find_all("a", href=True):
            if element["href"].startswith("https://mdbootstrap.com/products/jquery-ui-kit/") \
                    and element["href"] not in visited_urls:
                unvisited_urls.add(element["href"])
        visited_urls.add(url)


if __name__ == "__main__":
    crawl_train()
    crawl_test()
