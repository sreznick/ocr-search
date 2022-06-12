import os
import string
import random
import re
import numpy as np
import cv2
from tqdm import tqdm
import shared_paths as paths
from djvu_utils.txt_utils import read_book
from book_ids import get_book_names_and_ids


MAX_HEIGHT = 32
IMG_FORMAT = 'jpg'
RU_LETTERS_LOWER = 'абвгдеёжзийклмнопрстуфхцчшщъыьэюя'
TEST_SIZE = 0.1  # |test dataset| / |train (+val) dataset|


def main():
    word_num = 0  # counter, defines image name
    TRAIN_DIR = os.path.join(paths.WORD_DIR, 'train')
    TEST_DIR = os.path.join(paths.WORD_DIR, 'test')
    for d in (TRAIN_DIR, TEST_DIR):
        assert not os.path.exists(d), 'Please move or delete existing dataset'
        os.mkdir(d)
        os.mkdir(os.path.join(d, 'images'))

    # output text files, labels (words) and word location
    f_labels_train = open(os.path.join(TRAIN_DIR, 'labels.txt'), 'w')
    f_info_train = open(os.path.join(TEST_DIR, 'info.csv'), 'w')
    f_info_train.write(','.join(('book_id', 'page', 'x', 'y', 'w', 'h')))
    f_labels_test = open(os.path.join(TEST_DIR, 'labels.txt'), 'w')
    f_info_test = open(os.path.join(TEST_DIR, 'info.csv'), 'w')
    f_info_test.write(','.join(('book_id', 'page', 'x', 'y', 'w', 'h')))

    for book_name, book_id in get_book_names_and_ids():
        print(f'Extracting words from `{book_name}.djvu`')

        # read file with book text fields (page, line, word, etc)
        book_txt = os.path.join(paths.TEXT_DIR, book_id + '.txt')
        pages = read_book(book_txt)  # list of Page objects

        img_dir_train = os.path.join(TRAIN_DIR, 'images', str(book_id))
        os.mkdir(img_dir_train)
        img_dir_test = os.path.join(TEST_DIR, 'images')

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

                # randomly choose whether to save image to train or val dataset
                if random.random() < TEST_SIZE:
                    f_labels = f_labels_test
                    f_info = f_info_test
                    img_dir = img_dir_test
                else:
                    f_labels = f_labels_train
                    f_info = f_info_train
                    img_dir = img_dir_train

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
    f_labels_train.close()
    f_info_train.close()
    f_labels_test.close()
    f_info_test.close()


def shrink_image(img: np.ndarray, max_height: int = MAX_HEIGHT) -> np.ndarray:
    """
    Resize image so that its height <= max_height.
    """
    h, w, _ = img.shape
    y_ratio = h / max_height
    if y_ratio > 1:
        w_new = int(round(w / y_ratio))
        # shape is specified as (width, height) in opencv
        img = cv2.resize(img, (w_new, max_height),
                         interpolation=cv2.INTER_AREA)
    return img


def filter_word(word: str) -> str:
    """
    Returns substring of word if it satisfies required criteria.
    """
    letters = RU_LETTERS_LOWER + str.upper(RU_LETTERS_LOWER)
    pattern = re.compile(f'([{letters}]+)[{string.punctuation}]?')
    match = re.match(pattern, word)
    if match is None:
        return ''
    return match.group(1)


if __name__ == '__main__':
    main()
