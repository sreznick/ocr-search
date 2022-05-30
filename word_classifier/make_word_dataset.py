import os
import numpy as np
import cv2
from tqdm import tqdm
import shared_paths as paths
from djvu_utils.txt_utils import read_book
from book_ids import get_book_names_and_ids


MAX_WIDTH = 220
MAX_HEIGHT = 50


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
                xywh = word.get_xywh(width=page_width,
                                     height=page_height)
                if not xywh:  # invalid image size
                    continue
                x, y, w, h = xywh
                word_img = page_img[y:y+h+1, x:x+w+1, :]  # crop
                word_img = shrink_image(word_img)         # resize

                # write image
                fname = os.path.join(IMG_DIR, f'{word_num}.png')
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


def shrink_image(img: np.ndarray) -> np.ndarray:
    h, w, _ = img.shape
    y_ratio = h / MAX_HEIGHT
    x_ratio = w / MAX_WIDTH
    if max(x_ratio, y_ratio) <= 1:  # no need to resize
        return img

    # calculate new height and width
    if x_ratio > y_ratio:
        new_width = MAX_WIDTH
        new_height = int(round(h / x_ratio))
    else:
        new_width = int(round(w / y_ratio))
        new_height = MAX_HEIGHT

    img = cv2.resize(img, (new_width, new_height),
                     interpolation=cv2.INTER_AREA)
    return img


if __name__ == '__main__':
    main()
