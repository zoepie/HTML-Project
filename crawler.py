import re
from bs4 import BeautifulSoup, NavigableString
from urllib.request import Request, urlopen


def save_file(dir_name, data):
    with open(dir_name, 'rb') as f:
        f.write(data)


# def replace_tag(soup_body):
#     for tag_a in soup_body.find_all():
#         if tag_a.string:
#             tag_a.string = 'replaced_string'
#     for tag_a in soup_body.find_all('p'):
#         if tag_a.string:
#             tag_a.string = 'replaced_string'
#     return soup_body

# def replace_text(soup_body):
#     soup_text = str(soup_body)
#     for string in soup_body.strings:
#         if str(string) != '\n':
#             soup_text = soup_text.replace(str(string), 'replaced_string')
#     return soup_text

# def replace_string(node):
#     for i in node.children:
#         if isinstance(i, NavigableString):
#             text.append(str(i))
#         else:
#             replace_string(i)


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
