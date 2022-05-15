"""
Use annotated djvu documents to generate:
  * png files -- one per page,
  * txt files -- one per document (book).

Currently raises `djvu.decode.NotAvailable` on one of the documents.
"""

import os
from djvu_utils.get_text import dump_text
from djvu_utils.djvu2pngs import book2pngs


DATA_DIR = os.path.join(os.getcwd(), 'data')
BOOK_DIR = os.path.join(DATA_DIR, 'books')
PAGE_DIR = os.path.join(DATA_DIR, 'pages')
TEXT_DIR = os.path.join(DATA_DIR, 'text')


def main():
    # collect paths to all djvu files in BOOK_DIR directory
    djvu_files = []
    for entry in os.scandir(BOOK_DIR):
        # skip non-djvu files
        if entry.is_file and entry.name.endswith('.djvu'):
            djvu_files.append(entry)

    # generate ids for every book
    num_digits = len(str(len(djvu_files) - 1))
    ids = ['{:0{:d}d}'.format(x, num_digits) for x in range(len(djvu_files))]

    # process every file
    with open(os.path.join(DATA_DIR, 'book_ids.csv'), 'w') as id_file:
        for djvu_file, id in zip(djvu_files, ids):
            txt_file = os.path.join(TEXT_DIR, id + '.txt')
            dump_text(djvu_file.path, txt_file)

            png_dir = os.path.join(PAGE_DIR, id)
            if not os.path.exists(png_dir):
                os.mkdir(png_dir)
            book2pngs(djvu_file.path, png_dir + '/')

            id_file.write('"' + djvu_file.name[:-5] + '"')
            id_file.write(',' + id + '\n')


if __name__ == '__main__':
    main()
