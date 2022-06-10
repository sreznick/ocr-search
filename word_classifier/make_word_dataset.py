import os
import string
import re
import numpy as np
import cv2
from tqdm import tqdm
import shared_paths as paths
from djvu_utils.txt_utils import read_book
from book_ids import get_book_names_and_ids


MAX_HEIGHT = 32
IMG_FORMAT = 'jpg'
RU_LETTERS = 'абвгдеёжзийклмнопрстуфхцчшщъыьэюя'


def main():
    word_num = 0  # counter, defines image name
    IMG_DIR = os.path.join(paths.WORD_DIR, 'images')
    if not os.path.exists(IMG_DIR):
        os.mkdir(IMG_DIR)

    # output text files, labels (words) and word location
    f_labels = open(os.path.join(paths.WORD_DIR, 'labels.txt'), 'w')
    f_info = open(os.path.join(paths.WORD_DIR, 'info.csv'), 'w')
    f_info.write(','.join(('book_id', 'page', 'x', 'y', 'w', 'h')))

    for book_name, book_id in get_book_names_and_ids():
        print(f'Extracting words from `{book_name}.djvu`')

        # read file with book text fields (page, line, word, etc)
        book_txt = os.path.join(paths.TEXT_DIR, book_id + '.txt')
        pages = read_book(book_txt)  # list of Page objects

        img_dir = os.path.join(IMG_DIR, str(book_id))
        assert not os.path.exists(img_dir)
        os.mkdir(img_dir)

        for page_num in tqdm(range(len(pages))):
            page = pages[page_num]
            if len(page.words) == 0:
                continue  # skip pages without words

            # read page image (BGR)
            fname = os.path.join(paths.PAGE_DIR, book_id, f'{page_num}.png')
            if not os.path.exists(fname):
                print('Page {page_num} in document {book_id} not found')
                continue
            page_img = cv2.imread(fname)
            page_height, page_width, _ = page_img.shape

            for word in page.words:
                # skip word if not matched by `filter_word`
                word.value = filter_word(word.value)
                if not word.value:
                    continue

                xywh = word.get_xywh(width=page_width,
                                     height=page_height)
                if not xywh:  # invalid image size
                    continue
                x, y, w, h = xywh
                word_img = page_img[y:y+h+1, x:x+w+1, :]  # crop
                word_img = shrink_image(word_img)         # resize

                # write image
                fname = os.path.join(img_dir, f'{word_num}.{IMG_FORMAT}')
                cv2.imwrite(fname, word_img)

                # write text entries
                f_labels.write(word.value + '\n')
                f_info.write('\n')
                csv_fields = [book_id] + list(map(str, (page_num, x, y, w, h)))
                f_info.write(','.join(csv_fields))

                # update word counter
                word_num += 1

    # close text files
    f_labels.close()
    f_info.close()


def shrink_image(img: np.ndarray, max_height: int = MAX_HEIGHT) -> np.ndarray:
    """
    Resize image so that its height <= max_height.
    """
    h, w, _ = img.shape
    y_ratio = h / max_height
    if y_ratio > 1:
        w_new = int(round(w / y_ratio))
        img = cv2.resize(img, (max_height, w_new),
                         interpolation=cv2.INTER_AREA)
    return img


def filter_word(word: str) -> str:
    """
    Returns substring of word if it satisfies required criteria.
    """
    pattern = re.compile(f'([{RU_LETTERS}]+)[{string.punctuation}]?')
    match = re.match(pattern, word)
    if match is None:
        return ''
    return match.group(1)


if __name__ == '__main__':
    main()
