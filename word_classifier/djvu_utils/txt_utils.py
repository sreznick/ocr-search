"""
Tools for working with `get_text.dump_text` output, i.e., txt files.
"""

from typing import Union
import re


class BBox:
    def __init__(self, x1: int, y1: int, x2: int, y2: int):
        self.x1 = x1
        self.x2 = x2
        self.y1 = y1
        self.y2 = y2

    def get_coords(self) -> tuple[int, int, int, int]:
        """
        Original coordinates (x1, y1, x2, y2). Measured in pixels from
        bottom-left corner.
        """
        return (self.x1, self.y1, self.x2, self.y2)

    def get_xywh(self, width: int, height: int) -> tuple[int, int, int, int]:
        """
        Get bounding box coordinates (x, y, width, height) inside an image with
        size (`width`, `height`). Measured in pixels from top-right corner.
        """
        if not (self.x1 < self.x2 <= width and self.y1 < self.y2 <= height):
            return None
        x = self.x1
        w = self.x2 - self.x1
        y = height - self.y2
        h = self.y2 - self.y1
        return (x, y, w, h)


class Page(BBox):
    def __init__(self, x1: int, y1: int, x2: int, y2: int):
        super().__init__(x1, y1, x2, y2)
        self.words = []

    def __repr__(self):
        return f'page[{self.x1, self.y1, self.x2, self.y2}]'

    def get_words(self, page_text: str):
        p = re.compile(r'word\[(\d+), (\d+), (\d+), (\d+)\]\s*?\"(.*?)\"',
                       re.DOTALL)
        for w in re.findall(p, page_text):
            x1, y1, x2, y2 = map(int, w[:4])
            # get rid of leading / trailing whitespace and newlines
            value = w[4].strip().replace('\n', ' ')
            if len(value) > 0 and not str.isspace(value):
                self.words.append(Word(x1, y1, x2, y2, value))


class Word(BBox):
    def __init__(self, x1: int, y1: int, x2: int, y2: int, value: str):
        super().__init__(x1, y1, x2, y2)
        self.value = value

    def __repr__(self):
        return f'word[{self.x1, self.y1, self.x2, self.y2}]'

    def __str__(self):
        return self.value


def match_page(line: str) -> Union[tuple[int], None]:
    """
    Match line `page[x1, y1, x2, y2]` and return `(x1, y1, x2, y2)`.
    """
    m = re.match(r'page\[(\d+), (\d+), (\d+), (\d+)\]', line)
    if not m:
        return None
    return tuple(map(int, m.groups()))


def read_book(fname: str) -> list[Page]:
    """
    Read txt file generated with `get_text.dump_text` and return a list of
    `Page` objects, each containing a list of `Word` objects.
    """
    pages = []
    page = None
    # words can have newlines, so collect all lines and then find words
    page_contents = []

    with open(fname, 'r') as f:
        for line in f:
            coords = match_page(line)
            if coords is not None:    # matched page[x1, y1, x2, y2]
                if page is not None:  # process previous page
                    page.get_words(''.join(page_contents))
                    pages.append(page)
                page = Page(*coords)
                page_contents = []
            else:
                if page is None:  # not inside a page for whatever reason
                    continue
                page_contents.append(line)

    # process last page
    if page is not None:
        page.get_words(''.join(page_contents))
        pages.append(page)
    return pages
