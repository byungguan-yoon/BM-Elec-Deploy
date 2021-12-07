import numpy as np
import torch
import math
import cv2
import imutils

import pytorch_lightning as pl
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Optional
from timm_unet import CustomUnet
from trans import get_transforms
from glob import glob
from tqdm import tqdm
from datetime import date
import json
import os
import time
from pprint import pprint
import pdb
# 아래 코드 유무에 따라 precision이 달라지는 것으로 보임
# os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def do_center_crop(img, cropx, cropy):
    y,x = img.shape
    startx = x//2 - cropx//2
    starty = y//2 - cropy//2
    return img[starty:starty+cropy, startx:startx+cropx]

def split_image2patch(img, patch_gird=4):
    h, w = img.shape
    p_h, p_w = int(h / patch_gird), int(w / patch_gird)
    h_slice_img = np.split(img, patch_gird, axis=0)
    h_slice_img = np.array(h_slice_img)
    
    slice_img = np.split(h_slice_img, patch_gird, axis=2)
    slice_img = np.array(slice_img)
    slice_img = slice_img.reshape(-1, p_h, p_w)
    return slice_img

def stack_patch2img(imgs):

    hs_img_list = list()
    for i in range(0, 16, 4):
        hs_img = np.hstack(imgs[i : i+ 4])
        hs_img_list.append(hs_img)
    img = np.vstack(hs_img_list)
    return img


def load_model_pl(path):
    model = LitClassifier()
    model = model.load_from_checkpoint(path)
    return model

class LitClassifier(pl.LightningModule):
    """
    >>> LitClassifier(Backbone())  # doctest: +ELLIPSIS +NORMALIZE_WHITESPACE
    LitClassifier(
      (backbone): ...
    )
    """

    def __init__(
        self,
        scale_list = [0.25, 0.5], # 0.125, 
        backbone: Optional[CustomUnet] = None,
        learning_rate: float = 0.0001,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=['backbone'])
        if backbone is None:
            backbone = CustomUnet()
        self.backbone = backbone

        self.split_patch = True
    def forward(self, batch):
        output = self.backbone.model(batch.float())
        return output.sigmoid()

def do_split_img_fatch(obj):
    kc, kh, kw = 1, 256, 256  # kernel size
    dc, dh, dw = 1, 256, 256  # stride
    obj = obj.permute(1,2,0)
    obj = obj.unfold(0, kw, dw).unfold(1, kh, kw)
    # print('split patch : ', obj.contiguous().view(-1, 3, 256, 256).shape)
    return obj.contiguous().view(-1, 1, 256, 256)

def get_h_line(img):
    cany_1 = cv2.Canny(img, 80, 120)
    lines = cv2.HoughLinesP(cany_1, 1, math.pi/2, 1, None, 30, 1)
    
    if lines is None:
        return None
    idx = 100
    sample_lines = lines[:idx]
    th_line = sample_lines[(lines[:idx, 0, 2] - lines[:idx, 0, 0] ) > 180]
    h_index = th_line[:, 0, 1]
    h_index.sort()
    
    margine = 30
    temp = None
    
    result_list = list()
    for h_i in h_index:

        if temp:
            diff = h_i - temp
        else:
            diff = 999

        temp = h_i

        if diff < margine:
            continue
        else:
            result_list.append(h_i)
    return result_list

def get_cent_point(mask):
    result_list = list()
    
    # find contours in the thresholded image
    cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    
    for c in cnts:
        M = cv2.moments(c)
        try:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            result_list.append((cX, cY))
        except:
            continue
    return result_list

def get_repre_point(cent_point_list, h_line_list):
    # section에 defect가 없는 경우
    # if len(cent_point_list) == 0:
    #     return []
    
    cent_point_arr = np.array(cent_point_list)
    from_slice_list = [0] + h_line_list
    to_slice_list = h_line_list + [1901]
    
    repre_list = list()
    for f, t in zip(from_slice_list, to_slice_list):
        try:
            idx = np.nonzero((f <= cent_point_arr[: , 1]) * (cent_point_arr[: , 1] < t))
            # 제품별 defect point 뽑기
            if len(idx[0]) != 0:
                repre_list.append(cent_point_arr[idx[0][0]].tolist())
            else:
                repre_list.append([None])
        except:
        # 제품에 defect가 없는 경우에도 제품 구분을 위한 빈 list 추가
            repre_list.append([None])
    print(repre_list)

    return repre_list

def show_img_label(img, mask):
    fig = plt.figure(figsize=(12, 12))
    ax1 = fig.add_subplot(1, 2, 1)
    ax2 = fig.add_subplot(1, 2, 2)
    ax1.imshow(img, cmap='gray')
    ax2.imshow(mask)
    plt.show()
    plt.close()

def save_img_label(img, mask, file_name):
    fig = plt.figure(figsize=(12, 12))
    ax1 = fig.add_subplot(1, 2, 1)
    ax2 = fig.add_subplot(1, 2, 2)
    ax1.imshow(img, cmap='gray')
    ax2.imshow(mask)
    os.makedirs('result', exist_ok=True)
    fig.savefig(f'result/{file_name}.png')
    plt.close()

def save_img_mark_label(img, mask, marking_path, file_name):
    marking_img = read_image(marking_path)
    fig = plt.figure(figsize=(12, 12))
    
    ax1 = fig.add_subplot(1, 3, 1)
    ax2 = fig.add_subplot(1, 3, 2)
    ax3 = fig.add_subplot(1, 3, 3)
    
    ax1.imshow(img, cmap='gray')
    ax2.imshow(mask)
    ax3.imshow(marking_img, cmap='gray')
    
    os.makedirs('result', exist_ok=True)
    fig.savefig(f'result/{file_name}.png')
    plt.close()

def read_image(path):
    return cv2.imread(path, cv2.IMREAD_GRAYSCALE)

def save_patch(patches, path):
    '''
    patches : torch tensor type (B, C, H, W)

    '''
    patches = patches.permute(0, 2, 3, 1)
    patches = patches.detach().numpy()
    for i, patch in enumerate(patches):
        file_name = f'{(i + 1)}.png'
        # print(os.path.join(path, file_name))
        cv2.imwrite(os.path.join(path, file_name), patch)

def get_rle_masks(masks):
    '''
    masks : torch tensor type (B, 1, H, W)
    '''
    output = list()
    for mask in masks:
        rle = mask2rle(mask)
        output.append(rle)
    return output

def mask2rle(img):
    '''
    Efficient implementation of mask2rle, from @paulorzp
    --
    img: numpy array, 1 - mask, 0 - background
    Returns run length as string formated
    Source: https://www.kaggle.com/xhlulu/efficient-mask2rle
    '''
    pixels = img.T.flatten()
    pixels = np.pad(pixels, ((1, 1), ))
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)

# def read_inference(img_path, target, output_save=False):
def read_inference(img_path, target, id, pre_h_line=None, path=None, output_save=False):

    model_path = 'epoch=28-val_dice_score=0.7206_.ckpt' # epoch=28-val_dice_score=0.7206_.ckpt
    model = load_model_pl(model_path)
    model = model.eval().cuda()

    image = read_image(img_path)
    # init으로 조절  필요
    trans = get_transforms(data='test')
    
    transformed = trans(image=image)
    input_image = transformed["image"]
    
    # split patch
    patches = do_split_img_fatch(input_image)
    if output_save:
        save_patch(patches, path)

    # normalize
    # patches = torch.divide(patches, 255.)
    patches = patches / 255.
    
    with torch.no_grad():
        output = model(patches.cuda())

    th = 0.15
    th_output = output > th 
    th_output = th_output.cpu().detach().numpy()[:, 0] # (P, H=1024, W=1024)
    rle_list = get_rle_masks(th_output)

    th_output = stack_patch2img(th_output)
    th_output = th_output.astype(np.uint8)

    section_flag = ''.join(rle_list)
    section_flag = len(section_flag) == 0
    info = [dict(patch_id=i+1, patch_RLE=rle) for i, rle in enumerate(rle_list)]
    # # -> 1900, 1900 resize
    th_output = cv2.resize(th_output, (1900, 1900)) 
    # # -> 2448 2048  zero pad
    # D_H, D_W =  (2048 - 1900) // 2, (2448 - 1900) // 2
    # th_output = cv2.copyMakeBorder(th_output, D_H, D_H, D_W, D_W, cv2.BORDER_CONSTANT, value=0)

    # # idx = 5
    # # show_img_label(patches[idx, 0], output[idx, 0].cpu().detach())
    # # show_img_label(image, th_output)
    # if output_save:
    #     save_name = '_'.join(img_path.split('/')[-2:]).split('.')[0].replace('/', '_')
    #     # save_img_label(image, th_output, save_name)

    #     raw_name = img_path.split('/')[-1].split('.')[0]
    #     mark_img_path = f'./data/data/{target}_marking/{raw_name}.png' # marking_path, file_name
    #     save_img_mark_label(image, th_output, file_name=save_name, marking_path=mark_img_path)
    # ------ 절취선 ------
    # start_time = time.time()

    if id % 7 == 0:
        h_line = get_h_line(image)
    else:
        h_line = pre_h_line
    # h_spend_time = time.time() - start_time
    # print(f'H Line : {h_line} | time : {h_spend_time}')
    
    if h_line is None:
        return info, None, 1
    else:
    # start_time = time.time()
        cent_pint_list = get_cent_point(th_output)
        # c_spend_time = time.time() - start_time
        # print(f'Center point : {cent_pint_list} | time : {c_spend_time}')

        # start_time = time.time()
        
        repre_coor_list = get_repre_point(cent_pint_list, h_line)

        if id <= 6:
            repre_coor_list = repre_coor_list[1:]
        # r_spend_time = time.time() - start_time
        # print(f'representation point : {repre_coor_list} | time : {r_spend_time}')

        return info, repre_coor_list, (section_flag + 1), h_line
    
    # return info, section_flag
def load_model():
    model_path = 'epoch=28-val_dice_score=0.7206_.ckpt' # epoch=28-val_dice_score=0.7206_.ckpt
    model = LitClassifier()
    model = model.load_from_checkpoint(model_path)
    model = model.eval().cuda()
    return model

def inference(model, image, path, id, pre_h_line=None):
    # init으로 조절  필요
    trans = get_transforms(data='test')
    cv2.imwrite(os.path.join(path, 'all.png'), image)
    transformed = trans(image=image)
    input_image = transformed["image"]
    
    # split patch
    patches = do_split_img_fatch(input_image)
    save_patch(patches, path)
    # normalize
    patches = torch.divide(patches, 255.)
    
    with torch.no_grad():
        output = model(patches.cuda())
    
    th = 0.15
    th_output = output > th 
    th_output = th_output.cpu().detach().numpy()[:, 0] # (P, H=1024, W=1024)
    rle_list = get_rle_masks(th_output)

    th_output = stack_patch2img(th_output)
    th_output = th_output.astype(np.uint8)

    section_flag = ''.join(rle_list)
    section_flag = len(section_flag) == 0 
    info = [dict(patch_id=i+1, patch_RLE=rle) for i, rle in enumerate(rle_list)]
    # -> 1900, 1900 resize
    th_output = cv2.resize(th_output, (1900, 1900), interpolation=cv2.INTER_NEAREST) 

    if id % 7 == 1:
        h_line = get_h_line(image)
    else:
        h_line = pre_h_line
    
    print(h_line)
    # pdb.set_trace()

    if h_line is None:
        return info, None, 1, h_line
    else:
        cent_pint_list = get_cent_point(th_output)
        
        repre_coor_list = get_repre_point(cent_pint_list, h_line)

        if id <= 7:
            repre_coor_list = repre_coor_list[1:]

        return info, repre_coor_list, (section_flag + 1), h_line

# def del_none_product_repre(repre_list, h_line):


#     return repre_list

def h_sum(repre_list):
    # pprint('--'*30)
    h_repre_tf_list = [False for _ in range(len(repre_list[0]))]
    h_repre_list = [False for _ in range(len(repre_list[0]))]
    # print('repre_list')
    # pprint(repre_list)
    for repre in repre_list:
        for p_id, j in enumerate(repre):
            if len(j) == 2:
                # pdb.set_trace()
                # print('h_repre_list')
                # pprint(h_repre_list)
                # print('h_repre_tf_list')
                # pprint(h_repre_tf_list)
                # print('p_id')
                # print(p_id)
                h_repre_tf_list[p_id] = True
                h_repre_list[p_id] = j[1]


    # print(h_repre_list)
    # print(h_repre_tf_list)
    return h_repre_list, h_repre_tf_list

# result ex) [False, False, False, False, True, False, False, False, False, False, False, True, None, None, None, None]
def find_none_idx(result):
    for i, result_e in enumerate(result):
        if result_e == None:
            idx = i
            break
    return idx

def insert_result(result, h_pro_tf_e, idx):
    for i, h_pro_tf_e_e in enumerate(h_pro_tf_e):
        if result[idx + i] == None:
            result[idx + i] = h_pro_tf_e_e
        else:
            result[idx + i] = result[idx + i] or h_pro_tf_e_e
    return result

# h_pro_tf: horizontal production true(NG)/false(OK)
# pro_is: production intersection ex) [1, 1, 1, 1, 1, 1, 1, 1]
def v_sum(h_pro_tf, pro_is):
    result = [None for _ in range(16)]
    for i, (h_pro_tf_e, pro_is_e) in enumerate(zip(h_pro_tf, pro_is)):
        # h_pro_tf_e ex) [False, False, False]
        if i == 0:
            # h_pro_tf_e_e ex) False
            for j, h_pro_tf_e_e in enumerate(h_pro_tf_e):
                result[j] = h_pro_tf_e_e
        else:
            idx = find_none_idx(result) - (pro_is_e + 1)
            result = insert_result(result, h_pro_tf_e, idx)
    return result

def down_num_hline(hline_list):
    pro_num_hline = list()
    for hlines in hline_list:
        count = 0
        for hline in hlines:
            if hline <= (1900 - 1290):
                count = count + 1
        pro_num_hline.append(count)
    return pro_num_hline


if __name__ == '__main__':
    target = '15_51_59'
    # target = 'data_2'
    raw = glob(f'../data/{target}/*png')
    raw = sorted(raw)

    sections = list()
    
    pre_h_line = None
    h_repre_list, h_repre_tf_list = list(), list()
    
    repre_list = list()
    h_line_list = list()
    for i, raw_img_path in enumerate(tqdm(raw[:])):
        info, repre, flag, pre_h_line = read_inference(raw_img_path, id=i, target=target,  pre_h_line=pre_h_line) #, path='./', output_save=True
        repre_list.append(repre)
        
        # 다음줄 넘어가기 전에 줄별 summary
        if i % 7 == 6:
            pprint(pre_h_line)
            h_line_list.append(pre_h_line)
            c_h_repre_list, c_h_repre_tf_list = h_sum(repre_list)
            h_repre_list.append(c_h_repre_list)
            h_repre_tf_list.append(c_h_repre_tf_list)
            repre_list = list()
            # pprint(repre_list)
            # pprint(h_repre_list)
            # pprint(h_repre_tf_list)
            # pprint(pre_h_line)
    else:
        h_repre_list[-1] = h_repre_list[-1][:-1]
        h_repre_tf_list[-1] = h_repre_tf_list[-1][:-1]

    # print(h_repre_list)
    init_pro = down_num_hline(h_line_list)
    # pprint(init_pro)


    result = v_sum(h_repre_tf_list, init_pro)
    # print(result)

    # repre_list = [[[449, 1076], [499, 1172], [None]], [[1552, 924], [385, 1478], [None]], [[100, 935], [None], [None]], [[297, 523], [None], [None]], [[None], [None], [None]], [[1283, 500], [None], [None]], [[146, 373], [810, 1815], [None]]]
    # pprint(repre_list)
    # h_repre_list, h_repre_tf_list = h_sum(repre_list)
    # pprint(h_repre_list)
    # pprint(h_repre_tf_list)

    #     section_info = dict(
    #         section_id=i+1, 
    #         section_flag=flag,
    #         patches=info
    #         )
    #     sections.append(section_info)
    # exp_info = dict(timestamp=date.today().strftime("%y-%m-%d"), sections=sections)
    # with open("exp.json", "w") as json_file:
    #     json.dump(exp_info, json_file)
    # print(exp_info)