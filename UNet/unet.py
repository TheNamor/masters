
import tensorflow as tf
import os
import random
import numpy as np

from tqdm import tqdm

from skimage.io import imread, imshow
from skimage.transform import resize
import scipy
import matplotlib.pyplot as plt
from patchify import patchify

def get_model(IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS):
    #Build the model
    inputs = tf.keras.layers.Input((IMG_HEIGHT//2, IMG_WIDTH//2, IMG_CHANNELS))
    s = tf.keras.layers.Lambda(lambda x: x / 255)(inputs)

    #Contraction path
    c1 = tf.keras.layers.Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(s)
    c1 = tf.keras.layers.Dropout(0.1)(c1)
    c1 = tf.keras.layers.Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c1)
    p1 = tf.keras.layers.MaxPooling2D((2, 2))(c1)

    c2 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p1)
    c2 = tf.keras.layers.Dropout(0.1)(c2)
    c2 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c2)
    p2 = tf.keras.layers.MaxPooling2D((2, 2))(c2)

    c3 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p2)
    c3 = tf.keras.layers.Dropout(0.2)(c3)
    c3 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c3)
    p3 = tf.keras.layers.MaxPooling2D((2, 2))(c3)

    c4 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p3)
    c4 = tf.keras.layers.Dropout(0.2)(c4)
    c4 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c4)
    p4 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(c4)

    c5 = tf.keras.layers.Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p4)
    c5 = tf.keras.layers.Dropout(0.3)(c5)
    c5 = tf.keras.layers.Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c5)

    #Expansive path
    u6 = tf.keras.layers.Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(c5)
    u6 = tf.keras.layers.concatenate([u6, c4])
    c6 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u6)
    c6 = tf.keras.layers.Dropout(0.2)(c6)
    c6 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c6)

    u7 = tf.keras.layers.Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(c6)
    u7 = tf.keras.layers.concatenate([u7, c3])
    c7 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u7)
    c7 = tf.keras.layers.Dropout(0.2)(c7)
    c7 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c7)

    u8 = tf.keras.layers.Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(c7)
    u8 = tf.keras.layers.concatenate([u8, c2])
    c8 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u8)
    c8 = tf.keras.layers.Dropout(0.1)(c8)
    c8 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c8)

    u9 = tf.keras.layers.Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same')(c8)
    u9 = tf.keras.layers.concatenate([u9, c1], axis=3)
    c9 = tf.keras.layers.Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u9)
    c9 = tf.keras.layers.Dropout(0.1)(c9)
    c9 = tf.keras.layers.Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c9)

    outputs = tf.keras.layers.Conv2D(1, (1, 1), activation='sigmoid')(c9)

    model = tf.keras.Model(inputs=[inputs], outputs=[outputs])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    #model.summary()

    return model

if __name__ == "__main__":
    seed = 42
    np.random.seed = seed

    IMG_WIDTH = 1024
    IMG_HEIGHT = 1024
    IMG_CHANNELS = 3

    TRAIN_PATH = '../consep_dataset/CoNSeP/Train/'
    TEST_PATH = '../consep_dataset/CoNSeP/Test/'

    train_ids = os.listdir(TRAIN_PATH + "Images/")
    test_ids = os.listdir(TEST_PATH + "Images/")

    X_train = np.zeros((len(train_ids)*4, IMG_HEIGHT//2, IMG_WIDTH//2, IMG_CHANNELS), dtype=np.uint8)
    Y_train = np.zeros((len(train_ids)*4, IMG_HEIGHT//2, IMG_WIDTH//2, 1), dtype=bool)

    # patchify(input, (512, 512, 3), step=512)

    print('Resizing training images and masks')
    for n, id_ in tqdm(enumerate(train_ids), total=len(train_ids)):
        img = imread(TRAIN_PATH + "Images/" + id_)[:,:,:IMG_CHANNELS]
        img = resize(img, (IMG_HEIGHT, IMG_WIDTH), mode='constant', preserve_range=True)
        x_patches = patchify(img, (512, 512, 3), step=512)
        for i in range(2):
            for j in range(2):
                X_train[n*4+i*2+j] = x_patches[i,j,:,:]  #Fill empty X_train with values from img
        mask = np.zeros((IMG_HEIGHT, IMG_WIDTH, 1), dtype=bool)
        mat = scipy.io.loadmat(TRAIN_PATH + 'Labels/' + id_[:-4] + '.mat')
        #print(mat)
        mask_ = np.asarray([[1 if x > 0 else 0 for x in row] for row in mat["inst_map"]])
        mask_ = np.expand_dims(resize(mask_, (IMG_HEIGHT, IMG_WIDTH), mode='constant',
                                        preserve_range=True), axis=-1)
        mask = np.maximum(mask, mask_)

        y_patches = patchify(mask, (512, 512, 1), step=512)
        for i in range(2):
            for j in range(2):
                Y_train[n*4+i*2+j] = y_patches[i,j,:,:]

    # test images
    X_test = np.zeros((len(test_ids)*4, IMG_HEIGHT//2, IMG_WIDTH//2, IMG_CHANNELS), dtype=np.uint8)
    Y_test = np.zeros((len(test_ids)*4, IMG_HEIGHT//2, IMG_WIDTH//2, 1), dtype=bool)
    sizes_test = []
    print('Resizing test images')
    for n, id_ in tqdm(enumerate(test_ids), total=len(test_ids)):
        img = imread(TEST_PATH + "Images/" + id_)[:,:,:IMG_CHANNELS]
        sizes_test.append([img.shape[0], img.shape[1]])
        img = resize(img, (IMG_HEIGHT, IMG_WIDTH), mode='constant', preserve_range=True)
        x_patches = patchify(img, (512, 512, 3), step=512)
        for i in range(2):
            for j in range(2):
                X_test[n*4+i*2+j] = x_patches[i,j,:,:]
        mask = np.zeros((IMG_HEIGHT, IMG_WIDTH, 1), dtype=bool)
        mat = scipy.io.loadmat(TEST_PATH + 'Labels/' + id_[:-4] + '.mat')
        mask_ = np.asarray([[1 if x > 0 else 0 for x in row] for row in mat["inst_map"]])
        mask_ = np.expand_dims(resize(mask_, (IMG_HEIGHT, IMG_WIDTH), mode='constant',
                                        preserve_range=True), axis=-1)
        mask = np.maximum(mask, mask_)
            
        y_patches = patchify(mask, (512, 512, 1), step=512)
        for i in range(2):
            for j in range(2):
                Y_test[n*4+i*2+j] = y_patches[i,j,:,:]

    print('Done!')

    image_x = random.randint(0, len(train_ids)-1)
    imshow(X_train[image_x])
    plt.show()
    imshow(np.squeeze(Y_train[image_x]))
    plt.show()


    model = get_model(IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS)

    ################################
    #Modelcheckpoint
    checkpointer = tf.keras.callbacks.ModelCheckpoint('unet_model.h5', verbose=1, save_best_only=True)

    callbacks = [
            tf.keras.callbacks.EarlyStopping(patience=2, monitor='val_loss'),
            tf.keras.callbacks.TensorBoard(log_dir='logs'),
            checkpointer]

    results = model.fit(X_train, Y_train, validation_split=0.2, batch_size=16, epochs=10, callbacks=callbacks)

    ####################################

    idx = random.randint(0, len(X_train)-1)


    preds_train = model.predict(X_train[:int(X_train.shape[0]*0.8)], verbose=1)
    preds_val = model.predict(X_train[int(X_train.shape[0]*0.8):], verbose=1)
    preds_test = model.predict(X_test, verbose=1)


    preds_train_t = (preds_train > 0.5).astype(np.uint8)
    preds_val_t = (preds_val > 0.5).astype(np.uint8)
    preds_test_t = (preds_test > 0.5).astype(np.uint8)


    # Perform a sanity check on some random training samples
    ix = random.randint(0, len(preds_train_t)-1)
    imshow(X_train[ix])
    plt.show()
    imshow(np.squeeze(Y_train[ix]))
    plt.show()
    imshow(np.squeeze(preds_train_t[ix]))
    plt.show()

    # Perform a sanity check on some random validation samples
    ix = random.randint(0, len(preds_val_t)-1)
    imshow(X_train[int(X_train.shape[0]*0.8):][ix])
    plt.show()
    imshow(np.squeeze(Y_train[int(Y_train.shape[0]*0.8):][ix]))
    plt.show()
    imshow(np.squeeze(preds_val_t[ix]))
    plt.show()


    from keras.metrics import MeanIoU
    n_classes = 2
    IOU_keras = MeanIoU(num_classes = 2)
    IOU_keras.update_state(Y_test[:,:,:,0], preds_test_t)
    print(IOU_keras.result().numpy())