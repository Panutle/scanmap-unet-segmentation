import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from keras.models import Sequential,Model, load_model
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, UpSampling2D, Concatenate ,UpSampling2D,Reshape,Conv2DTranspose, concatenate ,Input,AveragePooling2D,Conv3D,BatchNormalization
from keras.utils import to_categorical
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping,Callback
from tensorflow.keras.preprocessing.image import load_img, img_to_array

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler,MinMaxScaler
import cv2
import os
import requests

def reconstruct_image_rgb(patches, image_shape, patch_size=64, step=64):
    h, w, c = image_shape
    recon_image = np.zeros((h, w, c), dtype=np.float32)
    weight_map = np.zeros((h, w, c), dtype=np.float32)

    patch_idx = 0
    for i in range(0, h - patch_size + 1, step):
        for j in range(0, w - patch_size + 1, step):
            recon_image[i:i+patch_size, j:j+patch_size] += patches[patch_idx]
            weight_map[i:i+patch_size, j:j+patch_size] += 1
            patch_idx += 1

    recon_image = recon_image / np.maximum(weight_map, 1e-8)

    # Clamp ค่าให้อยู่ในช่วง [0,1]
    recon_image = np.clip(recon_image, 0, 1)

    return recon_image

path_img = '/Users/urc/Desktop/code/ScanMap/Datasets/BKS2_X.bmp'

img_X = cv2.imread(path_img)

height_kernel = 64
width_kernel = 64

step_per_img_h = 64
step_per_img_w = 64

list_input = []

n = 1
for i in range(0,img_X.shape[0]-height_kernel,step_per_img_h):
    for j in range(0,img_X.shape[1]-width_kernel,step_per_img_w):
        img_crop_X = img_X[i:i+height_kernel, j:j+width_kernel]

        cv2.imwrite(f'/Users/urc/Desktop/code/ScanMap/Datasets/Test_full_img/X_{n}.bmp', img_crop_X)

        img = load_img(f'/Users/urc/Desktop/code/ScanMap/Datasets/Test_full_img/X_{n}.bmp', color_mode='rgb')
        img_np = np.array(img_to_array(img)/255).reshape(64,64,3)

        list_input.append(img_np)

        os.remove(f'/Users/urc/Desktop/code/ScanMap/Datasets/Test_full_img/X_{n}.bmp')

        n += 1

model = load_model('/Users/urc/Desktop/code/ScanMap/Models/model1.keras')

X = np.array(list_input)

list_pred = []

predictions = model.predict(X)

if not(os.path.isdir('/Users/urc/Desktop/code/ScanMap/Outputs/Predictions')):
    os.makedirs('/Users/urc/Desktop/code/ScanMap/Outputs/Predictions')

nn = 0
for i in range(img_X.shape[0]//height_kernel):
    for j in range(img_X.shape[1]//width_kernel):
        # print(i,j)
        cv2.imwrite(f'/Users/urc/Desktop/code/ScanMap/Outputs/Predictions/img_{nn}_{i}_{j}.bmp', predictions[nn])

        list_pred.append(predictions[nn])
        nn+=1

# img_shape = ((7013//height_kernel)*height_kernel, (5100//width_kernel)*width_kernel, 3)
img_shape = (7013, 5100, 3)

full_mask = reconstruct_image_rgb(list_pred, img_shape, patch_size=64, step=step_per_img_h)

mask_uin8 = (full_mask*255).astype(np.uint8)

cv2.imwrite('Final_con.bmp', cv2.cvtColor(mask_uin8, cv2.COLOR_BGR2RGB))

