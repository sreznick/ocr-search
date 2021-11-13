from bs4 import BeautifulSoup
from urllib.request import urlopen
import re
import requests
from simhash import Simhash
from time import sleep
import psycopg2
import collections


def calc_simhash_value(curr_page, soup):
    curr_text = soup.get_text()
    lines = (line.strip() for line in curr_text.splitlines())
    chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
    curr_text = '\n'.join(chunk for chunk in chunks if chunk)
    curr_simhash_val = float(Simhash(curr_text).value)
    return curr_simhash_val


class Search:

    def __init__(self, main_page, start_page=None):
        self.main_page = main_page
        self.start_page = start_page
        self.hashes_of_visited_pages = set()
        self.curr_djvu_set = set()
        self.root = start_page if start_page is not None else main_page
        self.queue = collections.deque([self.root])

    def insert_link_into_db(self, link):
        connection = psycopg2.connect(user="xxx", password="xxx",
                                      host="127.0.0.1", dbname="djvu")
        cursor = connection.cursor()
        cursor = connection.cursor()
        insert_query = """INSERT INTO djvu_links  VALUES ('""" + link + """')"""
        cursor.execute(insert_query)
        connection.commit()

    def djvu_links(self, href):
        return href and re.compile("djv").search(href)

    def another_links(self, href):
        return href and not self.djvu_links(href)

    def bfs(self):
        soup = BeautifulSoup(urlopen(self.root))
        root_simhash_val = calc_simhash_value(self.root, soup)
        self.hashes_of_visited_pages.add(root_simhash_val)
        while self.queue:
            curr_page = self.queue.popleft()
            soup = BeautifulSoup(urlopen(curr_page))
            for link in soup.find_all('a', href=self.djvu_links):
                whole_link = curr_page + link.get('href')
                if whole_link not in self.curr_djvu_set:
                    r = requests.head(whole_link)
                    if r.status_code == 200 and 'text/html' not in r.headers['content-type']:
                        self.curr_djvu_set.add(whole_link)
                        self.insert_link_into_db(whole_link.replace("'", "''"))
            for link in soup.find_all('a', href=self.another_links):
                next_link = link.get('href')
                if len(next_link) and next_link[0] != '.' and next_link[0] != '#' and next_link[-1] == '/':
                    try:
                        next_link = curr_page + next_link
                        r = requests.head(next_link)
                        sleep(5)
                        if r.status_code == 200 and 'text/html' in r.headers['content-type']:
                            curr_soup = BeautifulSoup(urlopen(next_link))
                            next_simhash_val = calc_simhash_value(next_link, curr_soup)
                            if next_simhash_val not in self.hashes_of_visited_pages:
                                print(f'new page is {next_link}')
                                self.hashes_of_visited_pages.add(next_simhash_val)
                                self.queue.append(next_link)
                    except requests.exceptions.ConnectionError:
                        r.status_code = "Connection refused"

    def get_djvu_set(self):
        return self.curr_djvu_set


sites = [
    'http://militera.org',
    'http://publ.lib.ru/publib.html',
]

for site in sites:
    curr_search = Search(site)
    curr_search.whole_search()
    print(f'djvu_list is: {curr_search.get_djvu_list()}')
