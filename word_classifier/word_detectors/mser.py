from typing import Union
import numpy as np
import cv2


def preprocess(img: np.ndarray,
               threshold: tuple[int, int],
               invert: bool = True,
               blur: Union[tuple[int, int], None] = None,
               dilate_kernel: Union[np.ndarray, None] = None,
               dilate_iterations: int = 1) -> np.ndarray:
    """
    Preprocess image `img` before finding word bounding boxes with MSER.
    Asserts input image is grayscale.
    """
    assert img.ndim == 2 and img.dtype == np.uint8

    # thresholding -> blur [optional] -> dilation [optional]
    thresh_type = cv2.THRESH_BINARY_INV if invert else cv2.THRESH_BINARY
    _, img = cv2.threshold(img, threshold[0], threshold[1], thresh_type)
    if blur:
        img = cv2.blur(img, blur)
    if dilate_kernel is not None and dilate_iterations > 0:
        img = cv2.dilate(img, dilate_kernel, dilate_iterations)
    return img


def mser_detector(img: np.ndarray, *args, **kwargs) -> np.ndarray:
    """
    Detect words in `img` with MSER. All positional and keyword arguments are
    passed to `cv2.MSER.create`.
    Returns array of bounding boxes.
    """
    mser = cv2.MSER.create(*args, **kwargs)
    _, bboxes = mser.detectRegions(img)
    return bboxes
