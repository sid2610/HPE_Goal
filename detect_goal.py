import numpy as np
import tensorflow as tf
import imageio
import skimage
import cv2
import random
import matplotlib.pyplot as plt
import pickle
import argparse

from skimage import color
from scipy.ndimage.filters import convolve
from skimage.color import rgb2gray
from skimage.transform import resize

from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, Dense, Dropout, BatchNormalization, MaxPooling2D, Flatten

ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dir", required=True,
	help="address of folder")
ap.add_argument("-v", "--video", required=True,
	help="name of input video")
args = vars(ap.parse_args())

def sobel_edges(img):
    img = rgb2gray(img)
    Kx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], np.float32)
    Ky = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], np.float32)
    Ix = convolve(img, Kx)
    Iy = convolve(img, Ky)
    return np.hypot(Ix, Iy)

def penalty_area(frame):
    frame_1 = np.zeros(frame.shape)
    frame_1 = frame.astype(np.float)/255
    for y in range(frame.shape[0]):
        for x in range(frame.shape[1]):
            if (frame[y][x][1]<(np.max([frame[y][x][0],frame[y][x][2]]))):
                frame_1[y][x] *= 0

    frame_2 = color.rgb2gray(frame_1)

    v = 0
    for i in range(frame_2.shape[0]):
        if (np.sum(frame_2[i])/frame_2.shape[1])>0.2:
            v = i
            break
    frame_3 = frame_2[v:]

    e = sobel_edges(frame_3)
    e_1 = (e>0.1) * 0.5

    uy = 0
    while np.sum(e_1[uy:uy+5,900:1000])/100<1.5:
        uy += 1

    ux = 750
    while np.sum(e_1[uy:uy+10,ux:ux+5])!=0:
        ux += 10
    ux -= 10
    while np.sum(e_1[uy:uy+10,ux:ux+5])!=0:
        ux += 1

    ugy = uy + 50
    while np.sum(e_1[ugy:ugy+5,ux-100:ux])/100<1.5:
        ugy += 1

    ugx = ux
    while np.sum(e_1[ugy:ugy+10,ugx:ugx+5])!=0:
        ugx += 10
    ugx -= 10
    while np.sum(e_1[ugy:ugy+10,ugx:ugx+5])!=0:
        ugx += 1

    upy = ugy + 50
    while np.sum(e_1[upy:upy+5,ugx-50:ugx])/50<1.5:
        upy += 1

    upx = ugx
    while np.sum(e_1[upy:upy+10,upx-10:upx])!=0:
        upx -= 10
    upx += 10
    while np.sum(e_1[upy:upy+10,upx-10:upx])!=0:
        upx -= 1

    lpy = upy + 50
    while np.sum(e_1[lpy:lpy+5,upx+100:upx+150])/50<1.5:
        lpy += 1

    lpx = upx+150
    while np.sum(e_1[lpy:lpy+10,lpx:lpx+5])!=0:
        lpx += 10
    lpx -= 10
    while np.sum(e_1[lpy:lpy+10,lpx:lpx+5])!=0:
        lpx += 1

    return upx, v+upy, lpx, v+lpy

def extract_frame(frame, x, y, h, w):
    hh = int(h/2)
    ww = int(w/2)
    if x>=ww and x<(frame.shape[1]-ww):
        lx = ww
    elif x<ww:
        lx = x
    else:
        lx = w-(frame.shape[1]-x)
    if y>=hh and y<(frame.shape[0]-hh):
        ly = hh
    elif y<hh:
        ly = y
    else:
        ly = h-(frame.shape[0]-y)
    return frame[y-ly:y+h-ly, x-lx:x+w-lx]

def sliding_window(img, win, step):
    for y in range(0,img.shape[0]-win[1],step):
        for x in range(0,img.shape[1]-win[0],step):
            yield (x,y,img[y:y+win[1],x:x+win[0]])

model_cnn_adam = load_model('./ball_detector.h5')

vid = imageio.get_reader(args["dir"] + args["video"])

win = (100,100)
step = 10

result = []
for j in range(0, vid.get_length()):
    frame = vid.get_data(j)
    if j%5 == 0:
        (ux,uy,lx,ly) = penalty_area(frame)
    locs = []
    preds = [0]
    test = frame[uy:ly, ux:lx]
    test_bw = rgb2gray(test)
    test_bw = np.expand_dims(test_bw, -1)
    for x,y,roi in sliding_window(test_bw, win, step):
        locs.append((x,y))
        t = np.expand_dims(roi, 0)
        preds.append(model_cnn_adam.predict(t))
    clone = test.copy()
    if np.max(preds)>0.99:
        for k in range(len(preds)):
            if np.max(preds) > 0.99:
                i = np.argmax(preds)-1
                cv2.rectangle(clone, (locs[i][0],locs[i][1]), (locs[i][0]+100,locs[i][1]+100), (0,0,255), 2)
    result.append(clone)

for i in range(len(result)):
    result[i] = resize(result[i], (300,300))

imageio.mimwrite(args["dir"] + 'result_1.mp4', result, fps=30)
