from bs4 import BeautifulSoup
from urllib.request import urlopen
import re
import requests
from time import sleep
import psycopg2


class Search:

    def __init__(self, main_page, start_page=None):
        self.djvu_patterns = [r'.*\.djvu$', r'.*\.\[djv\-fax\]$.zip']
        self.main_page = main_page
        self.start_page = start_page
        self.visited_pages = set()
        self.curr_djvu_list = []

    def insert_link_into_db(self, link):
        connection = psycopg2.connect(user="postgres", password="xxx",
                                      host="127.0.0.1", dbname="djvu")
        cursor = connection.cursor()
        cursor = connection.cursor()
        insert_query = """ INSERT INTO djvu_links  VALUES ('""" + \
                       link + """')"""
        cursor.execute(insert_query)
        connection.commit()

    def dfs(self, curr_page):
        self.visited_pages.add(curr_page)
        print(f'curr_page {curr_page}')
        soup = BeautifulSoup(urlopen(curr_page))
        for link in soup.find_all('a', href=True):
            next_link = link.get('href')
            if next_link is not None and len(
                    next_link) and next_link != '/' and (
                    self.main_page in next_link or next_link[0] == '/'):
                flag = False
                for pattern in self.djvu_patterns:
                    flag |= (re.match(pattern, next_link) is not None)
                try:
                    if next_link[0] == '/':
                        next_link = self.main_page + next_link
                    r = requests.head(next_link)
                    sleep(10)
                    if flag and 'text/html' not in r.headers['content-type']:
                        self.curr_djvu_list.append(next_link)
                        self.insert_link_into_db(next_link)
                        return
                except requests.exceptions.ConnectionError:
                    r.status_code = "Connection refused"

                if r.status_code == 200 and \
                        'text/html' in r.headers['content-type'] and \
                        next_link not in self.visited_pages:
                    self.dfs(next_link)

    def whole_search(self):
        if self.start_page is not None:
            self.dfs(self.start_page)
        else:
            self.dfs(self.main_page)

    def get_djvu_list(self):
        return self.curr_djvu_list


sites = [
    'http://militera.org',
    'http://publ.lib.ru/publib.html',
]

for site in sites:
    curr_search = Search(site)
    curr_search.whole_search()
    print(f'djvu_list is: {curr_search.get_djvu_list()}')