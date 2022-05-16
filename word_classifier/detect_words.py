import os
import numpy as np
import cv2


DATA_DIR = os.path.join(os.getcwd(), 'data')

# testing -- process single page
INPUT_FILE = os.path.join(DATA_DIR, 'pages', '02', '21.png')
INTERIM_FILE = os.path.join(DATA_DIR, 'tmp.png')
OUTPUT_FILE = os.path.join(DATA_DIR, 'out.png')
MAX_WIDTH = 800


def main():
    img = cv2.imread(INPUT_FILE)
    img2 = img.copy()
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, w = img.shape
    k = min(1, MAX_WIDTH / w)
    img = cv2.resize(img, None, fx=k, fy=k)
    img2 = cv2.resize(img2, None, fx=k, fy=k)
    # img = cv2.blur(img, (1, 5))
    _, img = cv2.threshold(img, 100, 255, cv2.THRESH_BINARY_INV)
    kernel = np.ones((1, 7), dtype='uint8')
    img = cv2.dilate(img, kernel, iterations=1)
    cv2.imwrite(INTERIM_FILE, img)

    mser = cv2.MSER.create(min_area=150, max_area=10000)
    _, bboxes = mser.detectRegions(img)
    print(len(bboxes))
    for bbox in bboxes:
        x, y, w, h = bbox
        cv2.rectangle(img2, (x, y), (x+w, y+h), (0, 255, 0), 1)
    cv2.imwrite(OUTPUT_FILE, img2)


if __name__ == '__main__':
    main()
