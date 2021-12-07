import cv2
import numpy as np
import json

from database import (
    fetch_last_one_result
)

async def rle2contours(section_id, patch_id):
    response = await fetch_last_one_result()
    section_id = int(section_id)
    patch_id = int(patch_id)
    dict = json.loads(response)
    rle = dict['sections'][(section_id-1)]['patches'][(patch_id-1)]['patch_RLE']
    temp = dict['timestamp'].split(':')
    img_path = "./static/result/" + temp[0] + "/" + temp[1] + "/s_" + str(section_id) + "/" + str(patch_id) + ".png"
    img = cv2.imread(img_path)
    contour_img = mask2contours(img, rle2mask(rle), color = [0, 0, 255])
    return contour_img


def mask2contours(image, mask_layer, color):
    """ converts a mask to contours using OpenCV and draws it on the image
    """

    # https://docs.opencv.org/4.1.0/d4/d73/tutorial_py_contours_begin.html
    # _, image_binary = cv2.threshold(mask_layer,  45, 255, cv2.THRESH_TOZERO)
    contours, _ = cv2.findContours(mask_layer, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    image = cv2.drawContours(image, contours, -1, color, 2)

    return image
    
# RLE to array 
def rle2mask(mask_rle, shape=(256,256)):
    '''
    mask_rle: run-length as string formated (start length)
    shape: (width,height) of array to return 
    Returns numpy array, 1 - mask, 0 - background
    Source: https://www.kaggle.com/paulorzp/rle-functions-run-lenght-encode-decode
    '''
    s = mask_rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(shape[0]*shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    return img.reshape(shape).T



