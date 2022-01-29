import cv2
import numpy as np
#import matplotlib.pyplot as plt


def cut_lines(bi: np.ndarray):
    bi_erode = cv2.erode(bi, np.ones((1, 7), np.uint8), iterations=1)
    
    # Init 'scanline'
    line_h = 3
    line_w = bi.shape[1]
    
    begin = []
    end = []
    line_he = []
    
    # Mark top and bottom of each line
    for y in range(bi.shape[0] - line_h):
        line = bi_erode[y:y+line_h, :]
        cnt = np.sum(line.flatten() == 0)*100/(line_w*line_h)
        
        
        if len(begin) == len(end) and cnt > 15.0:
            begin.append(y)
        elif len(begin) > len(end) and cnt < 10.0:
            end.append(y + line_h)
            line_he.append(y + line_h - begin[len(begin) - 1])
            
    line_he = np.array(line_he).mean()
    if len(begin) > len(end):
        begin = begin[:-1]
            
    # Mark top and bottom of the page
    top = np.array(np.append([max(0, begin[0] - line_he)], end), dtype=int)
    bottom = np.array(np.append(begin, [min(bi.shape[0]-1, end[len(end)-1] + line_he)]), dtype=int)
    
    # Cut lines with margin
    lines = []
    
    for i in range(len(top)-1):
        lines.append(bi[max(0, top[i]-line_h):min(bi.shape[0]-1, bottom[i+1]+line_h), :])
    
    return lines

def cut_words(bi_line: np.ndarray):
    bi_erode = cv2.erode(bi_line, np.ones((9, 7), np.uint8), iterations=1)
    
    # Init 'scanline'
    scan_h = bi_line.shape[0]
    scan_w = 5
    
    begin = []
    end = []
    space_w = []
    
    # Mark right and left of each line
    space_len = 0
    for x in range(bi_line.shape[1] - scan_w):
        col = bi_erode[:, x:x+scan_w]
        cnt = np.sum(col.flatten() == 0)*100/(scan_h*scan_w)
        space_len += 1
            #print(cnt)
        
        
        if len(begin) == len(end) and cnt >= 10.0:
            begin.append(x)
            if len(begin):
                space_w.append(space_len)
        else:
            #print(cnt)
            if len(begin) > len(end) and cnt <= 5.0:
                end.append(x + scan_w)
                space_len = 0

    if not len(space_w) or not len(begin) or not len(end):
        return []
    space_w = int(np.array(space_w).mean())
    
    if len(begin) > len(end):
        begin = begin[:-1]
            
    # Mark top and bottom of the page
    left = [max(0, begin[0] - space_w)]
    right = [max(0, begin[0] - space_w)]
    k = 0
    for i in range(len(begin)):
        if i == 0 or begin[i] - right[k] >= space_w/3:
            left.append(begin[i])
            right.append(end[i])
            k += 1
        else:
            right[k] = end[i]
            
    left.append(min(bi_line.shape[1]-1, end[-1] + space_w))
    right.append(min(bi_line.shape[1]-1, end[-1] + space_w))
    left = np.array(left, dtype=int)
    right = np.array(right, dtype=int)
    
    
    #left = np.array(np.append([max(0, begin[0] - space_w)], end), dtype=int)
    #right = np.array(np.append(begin, [min(bi_line.shape[1]-1, end[-1] + space_w)]), dtype=int)
    
    # Cut lines with margin
    words = []
    blank = np.full((scan_h, space_w), 255, dtype=np.uint8)
    
    for i in range(len(right)-2):
        words.append(np.hstack((blank, bi_line[:, left[i+1]:right[i+1]], blank)).T)
    
    return words

def intersecting(rect1, rect2): # rect1 левее rect2
    (x1, y1, w1, h1) = rect1
    (x2, y2, w2, h2) = rect2
    if x1 + w1 <= x2 or x1 >= x2 + w2:
        return False
    if y1 + h1 <= y2 or y1 >= y2 + h2:
        return False
    return True

def intersect(rect1, rect2):
    (x1, y1, w1, h1) = rect1
    (x2, y2, w2, h2) = rect2
    z1 = x1 + w1
    t1 = y1 + h1
    z2 = x2 + w2
    t2 = y2 + h2
    
    x = min(x1, x2)
    y = min(y1, y2)
    z = max(z1, z2)
    t = max(t1, t2)
    
    w = z - x
    h = t - y
    
    return (x, y, w, h)

def contour_letters(img_binary):
    
    # Detect letters' contours
    img_erode = cv2.erode(img_binary, np.ones((7, 1), np.uint8), iterations=1)
    contours, hi = cv2.findContours(cv2.bitwise_not(img_erode), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Get rectangles
    rectangles = []
    for i, contour in enumerate(contours):
            (x, y, w, h) = cv2.boundingRect(contour)
            if x == 0 or y == 0 or y+h == img_binary.shape[0] or x+w == img_binary.shape[1]:
                continue
            rectangles.append((x, y, w, h))
    
    rectangles = np.array(rectangles, dtype=[('x', '<i4'), ('y', '<i4'), ('w', '<i4'), ('h', '<i4')])
    ids = np.argsort(rectangles, order=('x', 'w', 'y', 'h'))
    rectangles = rectangles[ids]
    
    if rectangles.shape[0] == 0:
        return rectangles
    
    # Join intersecting rectangles
    letters = []
    prev = rectangles[0]
    for i, rec in enumerate(rectangles[1:], start=1):
        if intersecting(prev, rec):
            prev = intersect(prev, rec)
        else:
            letters.append(prev)
            prev = rec
    letters.append(prev)
            
    # Filter rectangles
    mid = [letters[i][1] + letters[i][3]/2 for i in range(len(letters))]
    mean = np.mean(mid)
    
    result = []
    for letter in np.array(letters, dtype=[('x', '<i4'), ('y', '<i4'), ('w', '<i4'), ('h', '<i4')]):
        (x, y, w, h) = letter
        if y > mean or y + h < mean:
            continue
        result.append(letter)
    
    return np.array(result, dtype=[('x', '<i4'), ('y', '<i4'), ('w', '<i4'), ('h', '<i4')])

def binary(image):
    if type(image) is str:
        img = cv2.imread(image)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    elif type(image) is np.ndarray:
        img = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        gray = image
    else:
        print("WRONG TYPE")
        return None, None
    
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return img, thresh

'''def contour_letters_draw(image):
    img, bi = binary(image)
    rectangles = contour_letters(bi)
    
    contoured = img.copy()
    
    
    rectangles = contour_letters(bi)
    for rec in rectangles:
        (x, y, w, h) = rec
        cv2.rectangle(contoured, (x, y), (x + w, y + h), (170, 0, 0), h//100 + 2)
    
    plt.imshow(contoured)'''
    
    
def letter_28x28(letter_cut):
    (h, w) = letter_cut.shape
    ln = max(w, h)
    letter_sq = 255 * np.ones(shape=(ln, ln), dtype=np.uint8)

    if w < h:
        letter_sq[:, ln//2 - w//2 : ln//2 + (w+1)//2] = letter_cut
    if w > h:
        letter_sq[ln//2 - h//2 : ln//2 + (h+1)//2, :] = letter_cut
    if w == h:
        letter_sq = letter_cut

    return cv2.resize(letter_sq, (28, 28))
    

def contour_letters_cut_28x28(image):
    _, bi = binary(image)
    
    letters = []
    
    # Cut the image into lines
    lines = cut_lines(bi)
    for line in lines:
        
        # Cut the line into words
        words = cut_words(line)
        for word in words:
            word = word.T
            letters_now = []
            positions = []
            # Contour letters
            rectangles = contour_letters(word)

            for rec in rectangles:
                # Resize the letter to recognition-friendly format
                (x, y, w, h) = rec
                letter = letter_28x28(word[y:y+h, x:x+w])

                # Run recognition
                letter = letter #TODO

                positions.append((x, y))
                letters_now.append(np.array(letter, dtype=float))
    
            pos = np.array(positions, dtype=[('x', '<i4'), ('y', '<i4')])
            ids = np.argsort(pos, order=('x', 'y'))
            letters.append(np.array(letters_now, dtype=np.ndarray)[ids])
            
    return np.array(letters, dtype=np.ndarray)