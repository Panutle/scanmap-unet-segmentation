import numpy as np

import matplotlib.pyplot as plt

from keras.models import Sequential,Model, load_model
from keras.layers import Conv2D, MaxPooling2D, MaxPooling3D, Flatten, Dense, Dropout, UpSampling2D, Concatenate, UpSampling2D, Reshape, Conv2DTranspose, Conv3DTranspose, concatenate, Input, AveragePooling2D, Conv3D, BatchNormalization, Lambda, UpSampling3D, ZeroPadding3D
from keras.utils import to_categorical
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, Callback, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.regularizers import l2
import gc
import tensorflow.keras.backend as K
import tensorflow as tf
import multiprocessing


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler,MinMaxScaler
import cv2
import os
import requests
from datetime import datetime
import random
import time


class StopTrainingWhenValLossIsLow(Callback):
    def __init__(self, threshold):
        super(StopTrainingWhenValLossIsLow, self).__init__()
        self.threshold = threshold

    def on_epoch_end(self, epoch, logs=None):
        if logs["val_loss"] < self.threshold:
            print(f"\nStopping training because val_loss has reached below {self.threshold}")
            self.model.stop_training = True

class Variable():
    def __init__(self):
        self.time = self.get_time()
        self.path_input_X = self.get_path_X_512()
        self.path_input_y = self.get_path_y_512()
        self.img_shape = 512
        self.n_train_data = 1000
        self.n_train_step = 1000
        self.channel = 6
        self.epochs = 10000000
        self.batch = 32

    def get_path_X_256(self):
        return f'/Users/urc/Desktop/code/ScanMap/Datasets/Train'

    def get_path_y_256(self):
        return f'/Users/urc/Desktop/code/ScanMap/Datasets/Test'
    
    def get_path_X_512(self):
        return f'/Users/urc/Desktop/code/ScanMap/Datasets/X512'
    
    def get_path_y_512(self):
        return f'/Users/urc/Desktop/code/ScanMap/Datasets/Y512'
    
    def get_path_models(self, start_time):
        return f'/Users/urc/Desktop/code/ScanMap/Models/model_{start_time}.keras'
    
    def get_path_models_checkpoint(self, start_time):
        return f'/Users/urc/Desktop/code/ScanMap/Models/model_checkpoint_{start_time}.keras'

    def get_time(self):
        return datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
    
    def get_path_outputs_loss(self):
        return f'/Users/urc/Desktop/code/ScanMap/Outputs/Plot_Loss/loss_{self.time}.png'
    
    def get_path_outputs_test(self, time):
        return f'/Users/urc/Desktop/code/ScanMap/Outputs/Predictions/test_{time}.png'
    
    def get_path_outputs_pred(self, time):
        return f'/Users/urc/Desktop/code/ScanMap/Outputs/Predictions/pred_{time}.png'
    
    def get_path_outputs_pred_2(self, time):
        return f'/Users/urc/Desktop/code/ScanMap/Outputs/Predictions/pred_2_{time}.png'
    
    def get_path_save_X(self, n):
        return f'/Users/urc/Desktop/code/ScanMap/X_{n}.bmp'
    
    def get_path_save_y(self, n):
        return f'/Users/urc/Desktop/code/ScanMap/y_{n}.bmp'

    
var = Variable()
    
def pre_data():

    img_X = cv2.imread(var.path_input_X)
    img_y = cv2.imread(var.path_input_y)

    gray_img_y = cv2.cvtColor(img_y, cv2.COLOR_BGR2GRAY)

    ret, thresh_y = cv2.threshold(gray_img_y, 50, 255, cv2.THRESH_BINARY_INV)

    height_kernel = 64
    width_kernel = 64

    step_per_img_h = 5
    step_per_img_w = 5

    list_X = []
    list_y = []

    if not(os.path.isdir(var.get_path_X_256())):
        os.makedirs(var.get_path_X_256())
    if not(os.path.isdir(var.get_path_y_256())):
        os.makedirs(var.get_path_y_256())

    # print(img_X.shape, thresh_y.shape)
    n = 1
    for i in range(0,img_X.shape[0]-height_kernel,step_per_img_h):
        for j in range(0,img_X.shape[1]-width_kernel,step_per_img_w):
            img_crop_X = img_X[i:i+height_kernel, j:j+width_kernel]

            img_crop_y = thresh_y[i:i+height_kernel, j:j+width_kernel]

            if np.sum(img_crop_y == 0) != height_kernel*width_kernel:
                for f in range(2):
                    img_crop_X = cv2.flip(img_crop_X, 1)

                    img_crop_y = cv2.flip(img_crop_y, 1)

                    for s in range(4):
                        print(n)
                        
                        img_crop_X = cv2.rotate(img_crop_X, cv2.ROTATE_90_CLOCKWISE)
                        img_crop_y = cv2.rotate(img_crop_y, cv2.ROTATE_90_CLOCKWISE)

                        cv2.imwrite(f'{var.get_path_X_256()}/X_{n}.bmp', img_crop_X)
                        cv2.imwrite(f'{var.get_path_y_256()}/y_{n}.bmp', img_crop_y)

                        img_X_load = load_img(f'{var.get_path_X_256()}/X_{n}.bmp', color_mode='rgb')
                        img_X_np = np.array(img_to_array(img_X_load)/255).reshape(64,64,3)

                        list_X.append(img_X_np)

                        img_y_load = load_img(f'{var.get_path_y_256()}/y_{n}.bmp', color_mode='grayscale')
                        img_y_np = np.array(img_to_array(img_y_load)/255).reshape(64,64,1)

                        list_y.append(img_y_np)

                        os.remove(f'{var.get_path_X_256()}/X_{n}.bmp')
                        os.remove(f'{var.get_path_y_256()}/y_{n}.bmp')
                        
                        n += 1

    X = np.array(list_X)
    y = np.array(list_y)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    return X_train, X_test, y_train, y_test


def pre_data_v2(ls_n_img):
    
    list_X = []
    list_y = []

    n=1

    for i, n_img in enumerate(ls_n_img):
        
        if not n_img.endswith(("tif", "bmp")) :
            continue
         
        img_X = cv2.imread(f'{var.path_input_X}/{n_img}')
        img_y = cv2.imread(f'{var.path_input_y}/{n_img}', cv2.IMREAD_GRAYSCALE)

        if type(img_X) == type(None) or type(img_y) == type(None):    
            continue

        img_bgr_X = auto_enhance(img_X, 180)

        img_X_np = prepare_rgb_hsv_combined(img_bgr_X)

        img_y_np = prepare_gray_image(img_y)

        list_X.append(img_X_np)

        list_y.append(img_y_np)

        n += 1

    # print(list_X[0], list_X[0].shape, list_y[0], list_y[0].shape)
    X = np.array(list_X)
    y = np.array(list_y)

    # สำหรับ 3D เท่านั้น
    # y = y[np.newaxis, ...]
    # X = X[np.newaxis, ...]

    print(X.shape)

    return X, y


def auto_enhance(img, target_mean=130):

    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # ทำ Histogram Equalization เฉพาะ V channel
    img_hsv[:, :, 2] = cv2.equalizeHist(img_hsv[:, :, 2])

    # แปลงกลับเป็น RGB
    img_enhanced = cv2.cvtColor(img_hsv, cv2.COLOR_HSV2RGB)


    hsv = cv2.cvtColor(img_enhanced, cv2.COLOR_RGB2HSV).astype(np.float32)
    h, s, v = cv2.split(hsv)

    current_mean = np.mean(s)
    if current_mean == 0:
        scale = 1.0
    else:
        scale = target_mean / current_mean
    
    # จำกัด scale ไว้ ไม่ให้เกินไป
    scale = np.clip(scale, 0.8, 2.0)
    s *= scale
    s = np.clip(s, 0, 255)

    hsv_enhanced = cv2.merge([h, s, v])
    enhanced_img_bgr = cv2.cvtColor(hsv_enhanced.astype(np.uint8), cv2.COLOR_HSV2BGR)
    return enhanced_img_bgr


def prepare_rgb_hsv_combined(img_bgr):

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB).astype("float32") / 255.0  # (H, W, 3)
    img_hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV).astype("float32")
    img_hsv[..., 0] /= 179.0
    img_hsv[..., 1] /= 255.0
    img_hsv[..., 2] /= 255.0  # (H, W, 3)

    img_combined = np.concatenate([img_rgb, img_hsv], axis=-1)  # (H, W, 6)

    # img_combined = np.expand_dims(img_combined, axis=0) # (1, H, W, 6)
    # img_combined = np.transpose(img_combined, (3, 1, 2, 0))
    # print(img_combined.shape)

    if var.channel == 3:
        return img_hsv
    elif var.channel == 6:
        return img_combined
    else:
        img_bgr


def prepare_gray_image(img):
    img = img.astype('float32') / 255.0            # Normalize เป็น [0, 1]
    img = np.expand_dims(img, axis=-1)             # จาก (H, W) → (H, W, 1)
    # img = np.expand_dims(img, axis=0)              # เพิ่ม batch dim → (1, H, W, 1)

    return img


def split_data():
    ls_n_img = os.listdir(var.path_input_X)
    if len(ls_n_img) < var.n_train_data:
        k = len(ls_n_img)
    else:
        k = var.n_train_data

    ls_n_img = random.sample(ls_n_img, k=k)

    random.shuffle(ls_n_img)

    train, test, _, _ = train_test_split(ls_n_img, ls_n_img, test_size=0.2, random_state=random.randint(0, 99))
    
    return train, test


def cov_img(predictions):

    pred_img = predictions.squeeze()
    pred_img_uint8 = (pred_img * 255).astype(np.uint8)
    _, binary = cv2.threshold(pred_img_uint8, 150, 255, cv2.THRESH_BINARY)

    # kernel = np.ones((5, 5), np.uint8) 
    
    # binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=2)

    print(binary.shape)

    # binary = cv2.dilate(binary, kernel, iterations=5)
    
    # binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

    # binary = cv2.erode(binary, kernel, iterations=1)
    
    # skeleton = skeletonize(binary)
    
    # skeleton_img = (skeleton.astype(np.uint8)) * 255

    # dila_2 = cv2.dilate(skeleton_img, kernel, iterations=1)

    return binary


def unet(input_size=(var.img_shape, var.img_shape, var.channel)):
    inputs = Input(input_size)
    kernel_1 = 9
    kernel_2 = 9
    kernel_2a = 9
    kernel_B = 9
    filters_1 = 64
    filters_2 = 128
    filters_2a = 256
    filters_B = 512

    # Contracting Path
    conv1 = Conv2D(filters_1, kernel_1, activation='relu', padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(1e-4))(inputs)
    conv1 = BatchNormalization()(conv1)
    conv1 = Conv2D(filters_1, kernel_1, activation='relu', padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(1e-4))(conv1)
    conv1 = BatchNormalization()(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1) # /2
    
    conv2 = Conv2D(filters_2, kernel_2, activation='relu', padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(1e-4))(pool1)
    conv2 = BatchNormalization()(conv2)
    conv2 = Conv2D(filters_2, kernel_2, activation='relu', padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(1e-4))(conv2)
    conv2 = BatchNormalization()(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2) # /2
    
    conv2a = Conv2D(filters_2a, kernel_2a, activation='relu', padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(1e-4))(pool2)
    conv2a = BatchNormalization()(conv2a)
    conv2a = Conv2D(filters_2a, kernel_2a, activation='relu', padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(1e-4))(conv2a)
    conv2a = BatchNormalization()(conv2a)
    pool2a = MaxPooling2D(pool_size=(2, 2))(conv2a) # /2
    
    
    # Bottom
    conv3 = Conv2D(filters_B, kernel_B, activation='relu', padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(1e-4))(pool2a)
    conv3 = Conv2D(filters_B, kernel_B, activation='relu', padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(1e-4))(conv3) 
    
    # Expansive Path
    up4a = Conv2DTranspose(filters_2a, (2, 2), strides=(2, 2), padding='same')(conv3) # x2
    up4a = concatenate([up4a, conv2a])
    conv4a = Conv2D(filters_2a, kernel_2a, activation='relu', padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(1e-4))(up4a)
    conv4a = BatchNormalization()(conv4a)
    conv4a = Conv2D(filters_2a, kernel_2a, activation='relu', padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(1e-4))(conv4a)
    conv4a = BatchNormalization()(conv4a)

    up4 = Conv2DTranspose(filters_2, (2, 2), strides=(2, 2), padding='same')(conv4a) # x2
    up4 = concatenate([up4, conv2])
    conv4 = Conv2D(filters_2, kernel_2, activation='relu', padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(1e-4))(up4)
    conv4 = BatchNormalization()(conv4)
    conv4 = Conv2D(filters_2, kernel_2, activation='relu', padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(1e-4))(conv4)
    conv4 = BatchNormalization()(conv4)
    
    up5 = Conv2DTranspose(filters_1, (2, 2), strides=(2, 2), padding='same')(conv4) # x2
    up5 = concatenate([up5, conv1])
    conv5 = Conv2D(filters_1, kernel_1, activation='relu', padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(1e-4))(up5)
    conv5 = BatchNormalization()(conv5)
    conv5 = Conv2D(filters_1, kernel_1, activation='relu', padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(1e-4))(conv5)
    conv5 = BatchNormalization()(conv5)

    # Output layer
    outputs = Conv2D(1, 1, activation='sigmoid')(conv5)  #sigmoid, softmax

    model = Model(inputs=inputs, outputs=outputs)
    return model


def unet_3d_to_2d(input_size=(None, var.img_shape, var.img_shape, var.channel)):  # (depth, height, width, channels)
    inputs = Input(input_size) # (6, 512, 512, 1)
    kernel_size = (3, 3, 3)

    filters_1 = 8
    filters_2 = 8
    filters_3 = 8
    filters_4 = 8

    # Down path
    conv1 = Conv3D(8, kernel_size, activation='relu', padding='valid', kernel_initializer='he_normal', kernel_regularizer=l2(1e-4))(inputs)
    # conv1 = BatchNormalization()(conv1)
    conv1 = Conv3D(8, kernel_size, activation='relu', padding='valid', kernel_initializer='he_normal', kernel_regularizer=l2(1e-4))(conv1)
    # conv1 = BatchNormalization()(conv1)
    pool1 = MaxPooling3D(pool_size=(2, 2, 2))(conv1)  # (3, 256, 256, 1)


    conv2 = Conv3D(8, kernel_size, activation='relu', padding='valid', kernel_initializer='he_normal', kernel_regularizer=l2(1e-4))(pool1)
    # conv2 = BatchNormalization()(conv2)
    conv2 = Conv3D(8, kernel_size, activation='relu', padding='valid', kernel_initializer='he_normal', kernel_regularizer=l2(1e-4))(conv2)
    # conv2 = BatchNormalization()(conv2)
    pool2 = MaxPooling3D(pool_size=(2, 2, 2))(conv2)  # (2, 128, 128, 1)


    # conv2a = Conv3D(filters_3, kernel_size, activation='relu', padding='valid', kernel_initializer='he_normal', kernel_regularizer=l2(1e-4))(pool2)
    # # conv2a = BatchNormalization()(conv2a)
    # conv2a = Conv3D(filters_3, kernel_size, activation='relu', padding='valid', kernel_initializer='he_normal', kernel_regularizer=l2(1e-4))(conv2a)
    # # conv2a = BatchNormalization()(conv2a)
    # pool2a = MaxPooling3D(pool_size=(1, 2, 2))(conv2a)  # (2, 64, 64, 1)


    # Bottom (no pooling)
    conv3 = Conv3D(8, kernel_size, activation='relu', padding='valid', kernel_initializer='he_normal', kernel_regularizer=l2(1e-4))(pool2)
    # conv3 = BatchNormalization()(conv3)
    conv3 = Conv3D(8, kernel_size, activation='relu', padding='valid', kernel_initializer='he_normal', kernel_regularizer=l2(1e-4))(conv3)
    # conv3 = BatchNormalization()(conv3)


    # Up path
    # up2a = Conv3DTranspose(filters_2, (2, 2, 2), strides=(1, 2, 2), padding='valid')(conv3)
    # up2a = concatenate([up2a, conv2a], axis=4)
    # conv4a = Conv3D(filters_3, kernel_size, activation='relu', padding='valid', kernel_initializer='he_normal', kernel_regularizer=l2(1e-4))(up2a)
    # # conv4a = BatchNormalization()(conv4a)
    # conv4a = Conv3D(filters_3, kernel_size, activation='relu', padding='valid', kernel_initializer='he_normal', kernel_regularizer=l2(1e-4))(conv4a)
    

    # up2 = Conv3DTranspose(filters_2, (2, 2, 2), strides=(1, 1, 1), padding='valid')(conv3)
    # up2 = concatenate([up2, conv2], axis=4)

    conv4 = Conv3DTranspose(8, kernel_size, padding='valid')(conv3)
    conv4 = Conv3DTranspose(8, kernel_size, padding='valid')(conv4)
    up2 = UpSampling3D(size=(2, 2, 2))(conv4)
    up2 = ZeroPadding3D(padding=((0, 1), (0, 0), (0, 0)))(up2)
    up2 = concatenate([up2, conv2], axis=-1)


    # up1 = Conv3DTranspose(filters_1, (2, 2, 2), strides=(1, 1, 1), padding='valid')(conv4)
    # up1 = concatenate([up1, conv1], axis=4)

    conv5 = Conv3DTranspose(8, kernel_size, padding='valid')(up2)
    conv5 = Conv3DTranspose(8, kernel_size, padding='valid')(conv5)
    up1 = UpSampling3D(size=(2, 2, 2))(conv5)
    up1 = concatenate([up1, conv1], axis=-1)

    up1 = Conv3DTranspose(8, kernel_size, padding='valid')(up1)
    up1 = Conv3DTranspose(8, kernel_size, padding='valid')(up1)

    # Output 
    conv_out = Conv3D(1, (1, 1, 1), activation='sigmoid', padding='valid')(up1)  # Shape: (batch, 6, H, W, 1)

    # # Extract middle slice
    # def extract_middle_slice(x):
    #     d = tf.shape(x)[1] // 2
    #     return x[:, d:d+1, :, :, :]  # Shape: (batch, 1, H, W, 1)

    # middle = Lambda(extract_middle_slice)(conv_out)
    # outputs = Lambda(lambda x: tf.squeeze(x, axis=1))(middle)  # Shape: (batch, H, W, 1)

    model = Model(inputs=inputs, outputs=conv_out)
    return model


def unet_test(input_size=(var.img_shape, var.img_shape, var.channel)):
    input_layer = Input(shape=(512, 512, 6))
    kernel_1 = (3, 3)

    conv1 = Conv2D(16, kernel_1, activation='relu', padding='same')(input_layer)
    conv1 = BatchNormalization()(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Conv2D(32, kernel_1, activation='relu', padding='same')(pool1)
    conv2 = BatchNormalization()(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv2a = Conv2D(64, kernel_1, activation='relu', padding='same')(pool2)
    conv2a = BatchNormalization()(conv2a)
    pool2a = MaxPooling2D(pool_size=(2, 2))(conv2a)

    conv2b = Conv2D(128, kernel_1, activation='relu', padding='same')(pool2a)
    conv2b = BatchNormalization()(conv2b)
    pool2b = MaxPooling2D(pool_size=(2, 2))(conv2b)


    # Bottom
    conv3 = Conv2D(256, kernel_1, activation='relu', padding='same')(pool2b)
    conv3 = Conv2D(256, kernel_1, activation='relu', padding='same')(conv3)

    # Expansive Path
    up4b = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(conv3)
    up4b = concatenate([up4b, conv2b])
    conv4b = Conv2D(128, kernel_1, activation='relu', padding='same')(up4b)
    conv4b = BatchNormalization()(conv4b)

    up4a = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(conv4b)
    up4a = concatenate([up4a, conv2a])
    conv4a = Conv2D(64, kernel_1, activation='relu', padding='same')(up4a)
    conv4a = BatchNormalization()(conv4a)

    up4 = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(conv4a)
    up4 = concatenate([up4, conv2])
    conv4 = Conv2D(32, kernel_1, activation='relu', padding='same')(up4)
    conv4 = BatchNormalization()(conv4)

    up5 = Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same')(conv4)
    up5 = concatenate([up5, conv1])
    conv5 = Conv2D(16, kernel_1, activation='relu', padding='same')(up5)
    conv5 = BatchNormalization()(conv5)


    # Output layer
    outputs = Conv2D(1, 1, activation='sigmoid')(conv5)  #sigmoid, softmax



    model = Model(inputs=input_layer, outputs=outputs)

    return model


def train_model(pr, i, n_train, n_train_step, train, early_stopping, checkpoint, start_time, X_test, y_test):
    
    print(f'\nstep: {i}/{n_train}\ntime: {datetime.now()}\n')

    if i+n_train_step >= n_train:
        min_index = i
        max_index = -1
    else:
        min_index = i
        max_index = i+n_train_step

    X_train, y_train = pre_data_v2(train[min_index:max_index])
    print(len(X_train), len(y_train))

    if not os.path.exists(var.get_path_models_checkpoint(start_time)):
        
        model = unet()
        model.summary()
    
        learning_rate = 0.0001  # Example learning rate
        adam_optimizer = Adam(learning_rate=learning_rate, clipnorm=1.0, beta_1=0.9, beta_2=0.999)
        
        model.compile(optimizer=adam_optimizer, loss=combined_loss, metrics=['accuracy'])  #sparse_categorical_crossentropy , binary_crossentropy , categorical_crossentropy

    else:
        
        model = load_model(var.get_path_models_checkpoint(start_time))

    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5)

    # Train the model
    model.fit(X_train, y_train, epochs=var.epochs, batch_size=var.batch, validation_data=(X_test, y_test), validation_split=0.2, callbacks=[early_stopping, checkpoint])
   
    model.save(var.get_path_models(start_time))

    print(var.get_path_models(start_time))

    # loss, accuracy = model.evaluate(X_train, y_train, batch_size=var.batch, callbacks=[early_stopping, checkpoint, reduce_lr])
    # print(loss, accuracy)

    loss, accuracy = model.evaluate(X_test, y_test, batch_size=var.batch, callbacks=[early_stopping, checkpoint])
    print(loss, accuracy)

    pr.put(accuracy)


def pred_model(start_time, X_test, test):
        
    model = load_model(var.get_path_models(start_time))

    time_ = datetime.now().strftime('%Y_%m_%d_%H_%M_%S')

    for i in range(1):
        plt.imshow(cv2.cvtColor(cv2.imread(f'{var.path_input_X}/{test[i]}'), cv2.COLOR_BGR2RGB))
        plt.axis(False)
        plt.savefig(var.get_path_outputs_test(time_), dpi=300, bbox_inches='tight')
        plt.close()

        predictions = model.predict(X_test[i].reshape(-1, var.img_shape, var.img_shape, var.channel))[0]

        # predictions = model.predict(X_test[:50])[0]
        plt.imshow(predictions, cmap='gray')
        plt.axis(False)
        plt.savefig(var.get_path_outputs_pred(time_), dpi=300, bbox_inches='tight')
        plt.close()
        
        # predictions = (predictions > np.median(predictions)).astype(int)

        closed_img = cov_img(predictions)
        
        plt.imshow(closed_img, cmap='gray')
        plt.axis(False)
        plt.savefig(var.get_path_outputs_pred_2(time_), dpi=300, bbox_inches='tight')
        plt.close()
        print(np.max(predictions))
        print(np.min(predictions))
        print(np.mean(predictions))


def dice_loss(y_true, y_pred):
    smooth = 1.
    y_true_f = tf.reshape(y_true, [-1])
    y_pred_f = tf.reshape(y_pred, [-1])
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    return 1 - ((2. * intersection + smooth) /
                (tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + smooth))
@tf.keras.utils.register_keras_serializable()
def combined_loss(y_true, y_pred):
    bce = tf.keras.losses.BinaryCrossentropy()(y_true, y_pred)
    dsc = dice_loss(y_true, y_pred)
    return bce + dsc


# def fit_models():
if __name__ == '__main__':
    start_time = datetime.now()

    # var = var

    train, test = split_data()

    # train = train[:var.n_train_data]

    print(len(train), len(test))

    gpus = tf.config.experimental.list_physical_devices('GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

    early_stopping = EarlyStopping(monitor='val_loss', # Monitor validation loss
                                patience=100, # How many epochs to wait after min has been hit
                                mode='min', # Stops training when the quantity monitored has stopped decreasing
                                restore_best_weights=True) # Restores model weights from the epoch with the minimum validation loss
    
    checkpoint = ModelCheckpoint(
        var.get_path_models_checkpoint(start_time.strftime('%Y_%m_%d_%H_%M_%S')),
        monitor='val_loss',
        save_best_only=True,
        save_weights_only=False,
        verbose=1
    )

    n_train = len(train)
    n_train_step = var.n_train_step

    print('set test')
    X_test, y_test = pre_data_v2(test)

    for i in range(0, n_train, n_train_step):

        # train_model(i, model, n_train, n_train_step, train, early_stopping, checkpoint, reduce_lr)
        pr = multiprocessing.Queue()
        p = multiprocessing.Process(target=train_model, args=(pr, i, n_train, n_train_step, train, early_stopping, checkpoint, start_time.strftime('%Y_%m_%d_%H_%M_%S'), X_test, y_test))
        p.start()
        p.join()

        K.clear_session()
        gc.collect()

        q = multiprocessing.Process(target=pred_model, args=(start_time.strftime('%Y_%m_%d_%H_%M_%S'), X_test, test))
        q.start()
        q.join()

        K.clear_session()
        gc.collect()


    end_time = datetime.now()
    
    # model.save(var.get_path_models())

    model = load_model(var.get_path_models(start_time.strftime('%Y_%m_%d_%H_%M_%S')))

    accuracy = pr.get()

    requests.post("https://ntfy.sh/Notification_Train_Models", 
                  data=f"Notification from Mac M2 Ultra: Model {os.path.basename(var.path_input_X)}\nModel time: {var.time}\nAccuracy: {round((accuracy*100), 2)}%\nTotal parameters: {model.count_params():,}\nTrain time: {end_time-start_time}")


    # Plot training & validation loss values
    # plt.plot(history.history['accuracy'])
    # plt.plot(history.history['val_accuracy'])
    # plt.plot(history.history['loss'])
    # plt.plot(history.history['val_loss'])
    # plt.title('Model loss')
    # plt.ylabel('Loss')
    # plt.xlabel('Epoch')
    # plt.legend(['accuracy', 'val_accuracy'], loc='upper left')
    # plt.savefig(var.get_path_outputs_loss(), dpi=300, bbox_inches='tight')
    # plt.close()


# if __name__ == '__main__':
#     fit_models()
    # split_data()

