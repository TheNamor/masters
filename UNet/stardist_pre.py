import os
import numpy as np

from tqdm import tqdm

from skimage.io import imread, imshow
from skimage.transform import resize
import scipy
import random
import matplotlib.pyplot as plt
from patchify import patchify

from stardist.models import StarDist2D
from stardist.plot import render_label
from csbdeep.utils import normalize

def predict_all(model, data):
    out = []
    for img in data:
        labels, _ = model.predict_instances(normalize(img))
        out.append(labels)
    
    return out

# creates a pretrained model
print("Spinning up model")
model = StarDist2D.from_pretrained('2D_versatile_he')

if __name__ == "__main__":
    IMG_WIDTH = 1024
    IMG_HEIGHT = 1024
    IMG_CHANNELS = 3

    TRAIN_PATH = '../consep_dataset/CoNSeP/Train/'
    TEST_PATH = '../consep_dataset/CoNSeP/Test/'

    train_ids = [os.listdir(TRAIN_PATH + "Images/")[0]]
    test_ids = [os.listdir(TEST_PATH + "Images/")[0]]

    X_train = np.zeros((len(train_ids)*4, IMG_HEIGHT//2, IMG_WIDTH//2, IMG_CHANNELS), dtype=np.uint8)
    Y_train = np.zeros((len(train_ids)*4, IMG_HEIGHT//2, IMG_WIDTH//2), dtype=np.uint8)

    # patchify(input, (512, 512, 3), step=512)

    print('Resizing training images and masks')
    for n, id_ in tqdm(enumerate(train_ids), total=len(train_ids)):
        img = imread(TRAIN_PATH + "Images/" + id_)[:,:,:IMG_CHANNELS]
        img = resize(img, (IMG_HEIGHT, IMG_WIDTH), mode='constant', preserve_range=True)
        x_patches = patchify(img, (512, 512, 3), step=512)
        for i in range(2):
            for j in range(2):
                X_train[n*4+i*2+j] = x_patches[i,j,:,:]  #Fill empty X_train with values from img
        #X_train[n] = normalize(img)
        mask = np.zeros((IMG_HEIGHT, IMG_WIDTH), dtype=np.uint8)
        mat = scipy.io.loadmat(TRAIN_PATH + 'Labels/' + id_[:-4] + '.mat')
        #print(mat)
        mask_ = mat["inst_map"]#np.asarray([[1 if x > 0 else 0 for x in row] for row in mat["inst_map"]])
        mask_ = resize(mask_, (IMG_HEIGHT, IMG_WIDTH), mode='constant', preserve_range=True)
        mask = np.maximum(mask, mask_)

        y_patches = patchify(mask, (512, 512), step=512)
        for i in range(2):
            for j in range(2):
                Y_train[n*4+i*2+j] = y_patches[i,j,:,:]
        #Y_train[n] = mask

    # test images
    X_test = np.zeros((len(test_ids)*4, IMG_HEIGHT//2, IMG_WIDTH//2, IMG_CHANNELS), dtype=np.uint8)
    Y_test = np.zeros((len(test_ids)*4, IMG_HEIGHT//2, IMG_WIDTH//2), dtype=np.uint8)
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
        #X_test[n] = normalize(img)
        mask = np.zeros((IMG_HEIGHT, IMG_WIDTH), dtype=np.uint8)
        mat = scipy.io.loadmat(TEST_PATH + 'Labels/' + id_[:-4] + '.mat')
        mask_ = np.asarray([[1 if x > 0 else 0 for x in row] for row in mat["inst_map"]])
        mask_ = resize(mask_, (IMG_HEIGHT, IMG_WIDTH), mode='constant', preserve_range=True)
        mask = np.maximum(mask, mask_)
            
        y_patches = patchify(mask, (512, 512), step=512)
        for i in range(2):
            for j in range(2):
                Y_test[n*4+i*2+j] = y_patches[i,j,:,:]
        #Y_test[n] = mask

    print('Done!')

    idx = random.randint(0, len(X_train)-1)

    preds_train = predict_all(model, X_train)
    #preds_val, _ = model.predict_instances(X_train[int(X_train.shape[0]*0.8):], verbose=1)
    preds_test = predict_all(model, X_test)

    #preds_train_t = (preds_train > 0.5).astype(np.uint8)
    #preds_val_t = (preds_val > 0.5).astype(np.uint8)
    #preds_test_t = (preds_test > 0.5).astype(np.uint8)


    # Perform a sanity check on some random training samples
    ix = random.randint(0, len(preds_train)-1)
    imshow(X_train[ix])
    plt.show()
    imshow(np.squeeze(Y_train[ix]))
    plt.show()
    imshow(np.squeeze(preds_train[ix]))
    plt.show()

    # Perform a sanity check on some random validation samples
    #ix = random.randint(0, len(preds_val)-1)
    #imshow(X_train[int(X_train.shape[0]*0.8):][ix])
    #plt.show()
    #imshow(np.squeeze(Y_train[int(Y_train.shape[0]*0.8):][ix]))
    #plt.show()
    #imshow(np.squeeze(preds_val[ix]))
    #plt.show()

    Y_test_bool = [(img > 0.5).astype(np.uint8) for img in Y_test]
    preds_test_bool = [(img > 0.5).astype(np.uint8) for img in preds_test]

    from keras.metrics import MeanIoU
    n_classes = 2
    IOU_keras = MeanIoU(num_classes = n_classes)
    IOU_keras.update_state(Y_test_bool, preds_test_bool)
    print(IOU_keras.result().numpy())
    from stardist.matching import matching_dataset

    print(matching_dataset((Y_train > 0.5).astype(np.uint8), (preds_train > 0.5).astype(np.uint8), show_progress=True).mean_true_score)