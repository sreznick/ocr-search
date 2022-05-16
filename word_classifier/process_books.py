"""
Use annotated djvu documents to generate:
  * png files -- one per page,
  * txt files -- one per document (book).

For whatever reason converting some pages to an image currently raises an
exception. These are skipped, a corresponding message is printed.

If a directory with pages for a given document already exists and is not empty
it is assumed that this document has been processed, i.e., all scans have been
generated. Such documents are skipped.
"""

import os
import shared_paths as paths
from book_ids import get_book_names_and_ids
from djvu_utils.get_text import dump_text
from djvu_utils.djvu2pngs import book2pngs


def main():
    for book, book_id in get_book_names_and_ids():
        input_file = os.path.join(paths.BOOK_DIR, book + '.djvu')

        # 1. extract all text from book
        output_file = os.path.join(paths.TEXT_DIR, book_id + '.txt')
        if not os.path.exists(output_file):
            dump_text(input_file, output_file)

        # 2. save all pages as png images
        output_dir = os.path.join(paths.PAGE_DIR, book_id)
        if not os.path.exists(output_dir):
            os.mkdir(output_dir)
        if not os.listdir(output_dir):
            book2pngs(input_file, output_dir)


if __name__ == '__main__':
    main()
