import os
import numpy as np
import cv2
import shared_paths as paths
from word_detectors import mser


INPUT_IMAGE = os.path.join(paths.PAGE_DIR, '07', '71.png')
MAX_WIDTH = 800
INTERIM_IMAGE = os.path.join(paths.DATA_DIR, 'tmp.png')
OUTPUT_IMAGE = os.path.join(paths.DATA_DIR, 'out.png')


def main():
    img = cv2.imread(INPUT_IMAGE)
    _, w, _ = img.shape
    k = min(1, MAX_WIDTH / w)
    img = cv2.resize(img, None, fx=k, fy=k)
    img2 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img2 = mser.preprocess(img2, (100, 200),
                           dilate_kernel=np.ones((1, 7), dtype='uint8'))
    cv2.imwrite(INTERIM_IMAGE, img2)
    bboxes = mser.mser_detector(img2, min_area=100, max_area=10000)
    for bbox in bboxes:
        x, y, w, h = bbox
        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 1)
    cv2.imwrite(OUTPUT_IMAGE, img)


if __name__ == '__main__':
    main()
