import cv2
import numpy as np

def single_scale_retinex(img, sigma):
    blur = cv2.GaussianBlur(img, (0, 0), sigma)
    return np.log1p(img) - np.log1p(blur)


def multi_scale_retinex(gray, sigmas=(15, 80, 250)):
    gray = gray.astype(np.float32) + 1.0

    msr = np.zeros_like(gray)

    for sigma in sigmas:
        msr += single_scale_retinex(gray, sigma)

    msr /= len(sigmas)

    msr = cv2.normalize(msr, None, 0, 255, cv2.NORM_MINMAX)

    return msr.astype(np.uint8)