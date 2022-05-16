"""
A collection of paths used by different scripts.

When executed makes sure all these paths exist.
"""

import os


# current working directory should be either:
#  - repository directory
#  - `word_classifier` subdirectory
SUBDIRECTORY = 'word_classifier'
ROOT = os.getcwd()
if os.path.split(ROOT)[-1] != SUBDIRECTORY:
    assert SUBDIRECTORY in os.listdir(ROOT)
    ROOT = os.path.join(ROOT, SUBDIRECTORY)

# directories for storing data
DATA_DIR = os.path.join(ROOT, 'data')
BOOK_DIR = os.path.join(DATA_DIR, 'books')  # djvu files
PAGE_DIR = os.path.join(DATA_DIR, 'pages')  # scans of individual pages
TEXT_DIR = os.path.join(DATA_DIR, 'text')   # all text extracted from djvu docs

# file with document name -> document id mapping
BOOK_ID_FILE = os.path.join(DATA_DIR, 'book_ids.csv')


if __name__ == '__main__':
    # create directories if they don't exist yet
    for d in (DATA_DIR, BOOK_DIR, PAGE_DIR, TEXT_DIR):
        if not os.exists(d):
            os.mkdir(d)
