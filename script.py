import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
import os
train = pd.read_csv("../input/airbus-ship-detection/train_ship_segmentations_v2.csv")
#count number to delete images without ships, 80 000 images will left
del_to =  - 80000 + len(train.ImageId.unique())
deleted = train[train.EncodedPixels.isna()][:del_to].ImageId
train = train[~train.ImageId.isin(deleted)]

#return image mask from list of encoded ships 
def mask_from_list(ships):
    if ships.isna().all(): return np.zeros((768,768), dtype=np.uint8)
    mask = np.zeros(768*768, dtype=np.uint8)
    for ship in ships:
        ship = list(map(int, ship.split()))
        for pair in range(len(ship)//2):
            start = ship[pair*2]
            end = start + ship[pair*2+1]
            mask[start:end] = 255
    return np.reshape(mask, (768,768)).transpose()
#return mask for list of image names
def masks_for_all(image_filenames):
    #l = len(image_filenames)
    masks= np.array([mask_from_list(train[train['ImageId']==image_filenames[0]].EncodedPixels)/255.0])
    images = np.array([cv2.imread('../input/airbus-ship-detection/train_v2/'+image_filenames[0])/255.0])
    #p = list(range(0,100,10))
    for i, filename in enumerate(image_filenames[1:]):
        #pr = int(i*100/l)
        #if pr in p:
        #    print(pr,"%")
        #    p.pop(0)
        mask = mask_from_list(train[train['ImageId']==filename].EncodedPixels)
        masks = np.append(masks,[mask/255.0], axis=0)
        images = np.append(images,[cv2.imread('../input/airbus-ship-detection/train_v2/'+filename)/255.0],axis=0)
    return images, masks
#return mask for one image name
def mask_for_one(filename):
    img, mask = masks_for_all([filename])
    return img[0], mask[0]
#function for plotting original image + true mask + predicted mask
def show_img_mask(img, mask, pred = False, pred_mask=None):
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize = (20, 5))
    ax1.imshow(img)
    ax1.set_title('Image')
    ax2.imshow(mask, cmap='gray')
    ax2.set_title('Masks')
    if not pred:
        ax3.imshow(cv2.multiply(img, cv2.merge([mask, mask, mask])))
        ax3.set_title('Masks in colors')
    else:
        ax3.imshow(pred_mask, cmap='gray')
        ax3.set_title('Prediction mask')
    plt.show()
	
#batch generator in order to using less memory
def mygenerator(name_list, batch_size=2, augment=False):
    k = 0
    while True:
        if k+batch_size >= len(name_list):
            k = 0
        imgs = []
        masks = []
        for name in name_list[k:k+batch_size]:
                tmp_img = cv2.imread('../input/airbus-ship-detection/train_v2/' + name)
                imgs.append(tmp_img)
                mask_list = train['EncodedPixels'][train['ImageId'] == name]
                one_mask = np.zeros((768, 768, 1))
                one_mask[:,:,0] += mask_from_list(mask_list)
                masks.append(one_mask)
        imgs = np.stack(imgs, axis=0)
        masks = np.stack(masks, axis=0)
        imgs = imgs / 255.0
        masks = masks / 255.0
        k += batch_size
        yield imgs, masks
		
#split on train and validation sets(first all images without ships)
split_to = int(len(train[train.EncodedPixels.isna()].ImageId.unique())*0.9)
names_train_na, names_valid_na = train[train.EncodedPixels.isna()].ImageId.to_list()[:split_to], train[train.EncodedPixels.isna()].ImageId.to_list()[split_to:]
split_to = int(len(train[~train.EncodedPixels.isna()].ImageId.unique())*0.9)
names_train_nna, names_valid_nna = list(train[~train.EncodedPixels.isna()].ImageId.unique())[:split_to], list(train[~train.EncodedPixels.isna()].ImageId.unique())[split_to:]
names_train = names_train_na + names_train_nna
names_valid = names_valid_na + names_valid_nna

#create train/validation data
BATCH_SIZE = 2
train_data = mygenerator(names_train, batch_size=BATCH_SIZE, augment=False)
valid_data = mygenerator(names_valid, batch_size=BATCH_SIZE, augment=False)

#import keras 
from keras.models import Model, load_model
from keras.layers import Input, BatchNormalization, Activation, Dropout
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.layers.pooling import MaxPooling2D, AveragePooling2D
from keras.layers.merge import concatenate
from keras.optimizers import Adam

#return Unet model
def unet_model(images, n_filters=8, kernel_size=3, dropout=0.3):
    #first 2 concolution layers, number of filters increas twise with each 2 layers
    conv0 = Conv2D(filters=n_filters, kernel_size=(kernel_size,kernel_size), padding='same', kernel_initializer='he_normal')(images)
    #with normalization layers
    conv0 = BatchNormalization()(conv0)
    #and relu activation
    conv0 = Activation('relu')(conv0)
    conv0 = Conv2D(filters=n_filters, kernel_size=(kernel_size,kernel_size), padding='same', kernel_initializer='he_normal')(images)
    conv0 = BatchNormalization()(conv0)
    conv0 = Activation('relu')(conv0)
    #pooling layer
    pool0 = AveragePooling2D((6,6))(conv0)
    #and layer for decrease overfitting
    pool0 = Dropout(dropout)(pool0)
    
    conv1 = Conv2D(filters=n_filters*2, kernel_size=(kernel_size,kernel_size), padding='same', kernel_initializer='he_normal')(pool0)
    conv1 = BatchNormalization()(conv1)
    conv1 = Activation('relu')(conv1)
    conv1 = Conv2D(filters=n_filters*2, kernel_size=(kernel_size,kernel_size), padding='same', kernel_initializer='he_normal')(conv1)
    conv1 = BatchNormalization()(conv1)
    conv1 = Activation('relu')(conv1)
    pool1 = AveragePooling2D((4,4))(conv1)
    pool1 = Dropout(dropout)(pool1)

    conv2 = Conv2D(filters=n_filters*4, kernel_size=(kernel_size,kernel_size), padding='same', kernel_initializer='he_normal')(pool1)
    conv2 = BatchNormalization()(conv2)
    conv2 = Activation('relu')(conv2)
    conv2 = Conv2D(filters=n_filters*4, kernel_size=(kernel_size,kernel_size), padding='same', kernel_initializer='he_normal')(conv2)
    conv2 = BatchNormalization()(conv2)
    conv2 = Activation('relu')(conv2)
    pool2 = MaxPooling2D((4,4))(conv2)
    pool2 = Dropout(dropout)(pool2)

    conv3 = Conv2D(filters=n_filters*8, kernel_size=(kernel_size,kernel_size), padding='same', kernel_initializer='he_normal')(pool2)
    conv3 = BatchNormalization()(conv3)
    conv3 = Activation('relu')(conv3)
    conv3 = Conv2D(filters=n_filters*8, kernel_size=(kernel_size,kernel_size), padding='same', kernel_initializer='he_normal')(conv3)
    conv3 = BatchNormalization()(conv3)
    conv3 = Activation('relu')(conv3)
    pool3 = MaxPooling2D((2,2))(conv3)
    pool3 = Dropout(dropout)(pool3)

    conv4 = Conv2D(filters=n_filters*16, kernel_size=(kernel_size,kernel_size), padding='same', kernel_initializer='he_normal')(pool3)
    conv4 = BatchNormalization()(conv4)
    conv4 = Activation('relu')(conv4)
    conv4 = Conv2D(filters=n_filters*16, kernel_size=(kernel_size,kernel_size), padding='same', kernel_initializer='he_normal')(conv4)
    conv4 = BatchNormalization()(conv4)
    conv4 = Activation('relu')(conv4)
    pool4 = MaxPooling2D((2,2))(conv4)
    pool4 = Dropout(dropout)(pool4)

    conv5 = Conv2D(filters=n_filters*32, kernel_size=(kernel_size,kernel_size), padding='same', kernel_initializer='he_normal')(pool4)
    conv5 = BatchNormalization()(conv5)
    conv5 = Activation('relu')(conv5)
    conv5 = Conv2D(filters=n_filters*32, kernel_size=(kernel_size,kernel_size), padding='same', kernel_initializer='he_normal')(conv5)
    conv5 = BatchNormalization()(conv5)
    conv5 = Activation('relu')(conv5)

    
    #use upsampling layer instead one of concolutional 
    upcv6 = Conv2DTranspose(n_filters*16, (3,3), strides=(2,2), padding='same')(conv5)
    #and concatenate with privious results
    mrge6 = concatenate([conv4, upcv6], axis=3)
    mrge6 = Dropout(dropout)(mrge6)
    conv6 = Conv2D(filters=n_filters*16, kernel_size=(kernel_size,kernel_size), padding='same', kernel_initializer='he_normal')(mrge6)
    conv6 = BatchNormalization()(conv6)
    conv6 = Activation('relu')(conv6)
    conv6 = Conv2D(filters=n_filters*16, kernel_size=(kernel_size,kernel_size), padding='same', kernel_initializer='he_normal')(conv6)
    conv6 = BatchNormalization()(conv6)
    conv6 = Activation('relu')(conv6)

    upcv7 = Conv2DTranspose(n_filters*8, (3,3), strides=(2,2), padding='same')(conv6)
    mrge7 = concatenate([conv3, upcv7], axis=3)
    mrge7 = Dropout(dropout)(mrge7)
    conv7 = Conv2D(filters=n_filters*8, kernel_size=(kernel_size,kernel_size), padding='same', kernel_initializer='he_normal')(mrge7)
    conv7 = BatchNormalization()(conv7)
    conv7 = Activation('relu')(conv7)
    conv7 = Conv2D(filters=n_filters*8, kernel_size=(kernel_size,kernel_size), padding='same', kernel_initializer='he_normal')(conv7)
    conv7 = BatchNormalization()(conv7)
    conv7 = Activation('relu')(conv7)

    upcv8 = Conv2DTranspose(n_filters*4, (3,3), strides=(4,4), padding='same')(conv7)
    upcv8 = BatchNormalization()(upcv8)
    mrge8 = concatenate([conv2, upcv8], axis=3)
    mrge8 = Dropout(dropout)(mrge8)
    conv8 = Conv2D(filters=n_filters*4, kernel_size=(kernel_size,kernel_size), padding='same', kernel_initializer='he_normal')(mrge8)
    conv8 = BatchNormalization()(conv8)
    conv8 = Activation('relu')(conv8)
    conv8 = Conv2D(filters=n_filters*4, kernel_size=(kernel_size,kernel_size), padding='same', kernel_initializer='he_normal')(conv8)
    conv8 = BatchNormalization()(conv8)
    conv8 = Activation('relu')(conv8)

    upcv9 = Conv2DTranspose(n_filters*2, (3,3), strides=(4,4), padding='same')(conv8)
    upcv9 = BatchNormalization()(upcv9)
    mrge9 = concatenate([conv1, upcv9], axis=3)
    mrge9 = Dropout(dropout)(mrge9)
    conv9 = Conv2D(filters=n_filters*2, kernel_size=(kernel_size,kernel_size), padding='same', kernel_initializer='he_normal')(mrge9)
    conv9 = BatchNormalization()(conv9)
    conv9 = Activation('relu')(conv9)
    conv9 = Conv2D(filters=n_filters*2, kernel_size=(kernel_size,kernel_size), padding='same', kernel_initializer='he_normal')(conv9)
    conv9 = BatchNormalization()(conv9)
    conv9 = Activation('relu')(conv9)

    upcv10 = Conv2DTranspose(n_filters, (3,3), strides=(6,6), padding='same')(conv9)
    mrge10 = concatenate([upcv10, conv0], axis=3)
    mrge10 = Dropout(dropout)(mrge10)
    conv10 = Conv2D(filters=n_filters, kernel_size=(kernel_size,kernel_size), padding='same', kernel_initializer='he_normal')(mrge10)
    conv10 = BatchNormalization()(conv10)
    conv10 = Activation('relu')(conv10)
    conv10 = Conv2D(filters=n_filters, kernel_size=(kernel_size,kernel_size), padding='same', kernel_initializer='he_normal')(conv10)
    conv10 = BatchNormalization()(conv10)
    conv10 = Activation('relu')(conv10)
    
    #output layer
    outputs = Conv2D(1, (1,1), activation='sigmoid')(conv10)
    model = Model(inputs=images, outputs=outputs)
    return model
	
#initialize model
inputs = Input(shape=(768,768,3))
model = unet_model(inputs)
from keras.backend import sum as Ksum, abs as Kabs, square as Ksquare
def dice_coef(y_true, y_pred, smooth=1):
    """
    Dice = (2*|X & Y|)/ (|X|+ |Y|)
         =  2*sum(|A*B|)/(sum(A^2)+sum(B^2))
    ref: https://arxiv.org/pdf/1606.04797v1.pdf
    """
    intersection = Ksum(Kabs(y_true * y_pred), axis=-1)
    return (2. * intersection + smooth) / (Ksum(Ksquare(y_true),-1) + Ksum(Ksquare(y_pred),-1) + smooth)

def dice_coef_loss(y_true, y_pred):
    return 1-dice_coef(y_true, y_pred)
	
#set parameters to model and start learning
from keras.callbacks import EarlyStopping,ReduceLROnPlateau, ModelCheckpoint
model.compile(optimizer = Adam(), loss='binary_crossentropy', metrics=[dice_coef])
callbacks = [EarlyStopping(patience=10, verbose=1, mode='max'),
        ReduceLROnPlateau(monitor='val_dice_coef',mode='max', factor=0.1,patience=5,min_lr=0.00001, verbose=1),
        ModelCheckpoint('../output/model-ships.h5',monitor="val_dice_coef", mode='max', verbose=1, save_best_only=True, save_weights_only=True)]
history = model.fit_generator(train_data,steps_per_epoch=500, epochs=25, callbacks=callbacks)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(22, 10))
ax1.plot(history.history["loss"], label="loss")
ax1.plot( np.argmin(history.history["loss"]), np.min(history.history["loss"]), marker="x", color="r", label="best model")
ax1.set_title('binary crossentropy per epoch')
ax1.legend();
ax2.plot(history.history["dice_coef"], label="val_loss")
ax2.set_title('dice coef per epoch')
ax2.legend();
plt.show()

#return score of prediction
def calc_score_one_image(mask_true, mask_pred):
    mask_true = mask_true.reshape(768,768)
    mask_pred = mask_pred.reshape(768,768)
    if mask_true.sum() == 0 and mask_pred.sum() == 0:
        score = 1
    elif mask_true.sum() == 0 and mask_pred.sum() != 0:
        score = 0
    elif mask_true.sum() != 0 and mask_pred.sum() == 0:
        score = 0
    else:
        score = dice_coef(mask_true, mask_pred)
    return score
	

#return score for botch of images
def calc_score_all_image(batch_mask_true, batch_mask_pred, threshold=0.5):
    num = batch_mask_true.shape[0]
    #tmp = batch_mask_pred > threshold
    batch_mask_pred = np.where(batch_mask_pred>threshold, 1, 0)
    scores = list()
    for i in range(num):
        score = calc_score_one_image(batch_mask_true[i], batch_mask_pred[i])
        scores.append(score)
    return np.mean(scores)
	
#import trackbar
from tqdm import tqdm
#go throught different threschold to choose the best one
scores_list = dict()
threshold_list = [i/100 for i in range(10,90,5)]
v_names = names_valid[:500]+names_valid[-500:]#
for threshold in threshold_list:
    scores = []
    print(threshold)
    for i in tqdm(range(len(v_names)//2)):
        temp_list = names_valid[i*2:(i+1)*2]
        val_img, val_mask = masks_for_all(temp_list)
        pred_mask = model.predict(val_img)
        score = calc_score_all_image(val_mask, pred_mask, threshold=threshold)
        scores.append(score)
    val = np.sum(scores)/(len(names_valid)//2 *2)
    scores_list[threshold] = val

#choose best threshold
threshold = max(scores_list, key=scores_list.get)

visualize_image_list = names_valid[-3:]
for i in visualize_image_list:
    img, true_mask = masks_for_all([i])
    pred_mask = model.predict(img)
    pred_mask = np.where(pred_mask>0.5, 1, 0) 
    pred_mask = pred_mask.reshape(768,768)
    show_img_mask(img[0], true_mask[0], True, pred_mask)
	
#save model
#model.load_weights(weight_path)
model.save('model-ships.h5')

#import test image names
test = pd.read_csv("../input/airbus-ship-detection/sample_submission_v2.csv")

#functions to transfor result to apropriate format for submission
from skimage.morphology import label
def multi_rle_encode(img):
    '''
    Encode connected regions as separated masks
    '''
    labels = label(img)
    if img.ndim > 2:
        return [rle_encode(np.sum(labels==k, axis=2)) for k in np.unique(labels[labels>0])]
    else:
        return [rle_encode(labels==k) for k in np.unique(labels[labels>0])]

# ref: https://www.kaggle.com/paulorzp/run-length-encode-and-decode
def rle_encode(img):
    '''
    img: numpy array, 1 - mask, 0 - background
    Returns run length as string formated
    '''
    if np.max(img) < 0.5:
        return '' ## no need to encode if it's all zeros
    pixels = img.T.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)
	
output_names = []
output_encodes = []

visualize_image_list = test.ImageId
for i in visualize_image_list:
    img = cv2.imread('../input/airbus-ship-detection/test_v2/' + i)
    pred_mask = model.predict(np.array([img]))
    pred_mask = np.where(pred_mask>threshold, 1, 0) 
    pred_mask = pred_mask.reshape(768,768).astype(np.uint8)
    encodes = multi_rle_encode(pred_mask)
    output_encodes += encodes
    [output_names.append(i) for j in range(len(encodes))]

output = pd.DataFrame (zip(output_names, output_encodes), columns = ['ImageId','EncodedPixels'])
output.to_csv('submission.csv', index=False)