"""
Generate IDs for all djvu files in the `BOOK_DIR` directory when executed.
"""

from typing import Generator, Tuple
import os
import re
import shutil
import shared_paths as paths


REMOVE_OLD = True  # remove previously generated files (page images, etc)


def main():
    # collect paths to all djvu files in BOOK_DIR directory
    djvu_files = []
    for entry in os.scandir(paths.BOOK_DIR):
        # skip non-djvu files
        if entry.is_file and entry.name.endswith('.djvu'):
            djvu_files.append(entry)

    # generate ids for every book
    num_digits = len(str(len(djvu_files) - 1))
    ids = ['{:0{:d}d}'.format(x, num_digits) for x in range(len(djvu_files))]

    # store file names and ids in a separate file
    # removes `.djvu` extension from file name
    # and wraps both file name and id in double quotes
    with open(paths.BOOK_ID_FILE, 'w') as outfile:
        for djvu_file, id_ in zip(djvu_files, ids):
            outfile.write(f'"{djvu_file.name[:-5]}"')
            outfile.write(',')
            outfile.write(f'"{id_}"')
            outfile.write('\n')

    if REMOVE_OLD:
        for id_ in ids:
            # remove all page scans
            book_pages_dir = os.path.join(paths.PAGE_DIR, id_)
            if os.path.exists(book_pages_dir):
                shutil.rmtree(book_pages_dir)

            # remove all text files with book contents
            book_text_file = os.path.join(paths.TEXT_DIR, id_ + '.txt')
            if os.path.exists(book_text_file):
                os.remove(book_text_file)

        # delete words dataset
        word_images_dir = os.path.join(paths.WORD_DIR, 'images')
        if os.path.exists(word_images_dir):
            shutil.rmtree(word_images_dir)
        for file in os.scandir(paths.WORD_DIR):
            os.remove(file)


def get_book_names_and_ids() -> Generator[Tuple[str, str], None, None]:
    with open(paths.BOOK_ID_FILE, 'r') as f:
        for line in f:
            if line:
                yield re.match('\"(.+)\",\"(.+)\"\n', line).groups()


if __name__ == '__main__':
    main()
