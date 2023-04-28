import os
import numpy as np

from tqdm import tqdm

from skimage.io import imread, imshow
from skimage.transform import resize
import scipy
import random
import matplotlib.pyplot as plt
from patchify import patchify, unpatchify

from stardist.models import Config2D, StarDist2D
from stardist.plot import render_label
from csbdeep.utils import normalize
from stardist import random_label_cmap, gputools_available, _draw_polygons
from stardist.matching import matching_dataset

def reconstruct(images, labels, original_dim=(2,2)):
    """
    Unpatchifies the images and corresponding labels

    Arguments-
    images (list):      list of patched images
    labels (list):      list of labels corresponding to patches

    Named Arguments-
    original_dim (tuple=(2,2)):         number of patches horizontally and vertically in original image

    Returns-
    (tuple):
        (list):         reconstructed images
        (list):         stitched labels
    """
    patches_per = (original_dim[0]*original_dim[1])
    if len(images) != len(labels):
        print("Error: images and labels of incompatible sizes")
        return [], []
    if len(images) % patches_per != 0:
        print("Error: uneven number of images for given dimensions")
        return [], []

    patch_size = images[0].shape
    image_size = (patch_size[0]*original_dim[1], patch_size[1]*original_dim[0], patch_size[2])
    images_out = []
    labels_out = []
    print("Unpatchifying...")
    for i in tqdm(range(len(images) // patches_per)):
        patches = []
        label_patches = []
        patch_slice = images[i*patches_per:i*patches_per + patches_per]
        label_slice = labels[i*patches_per:i*patches_per + patches_per]
        for j in range(original_dim[1]):
            patches.append([])
            for k in range(original_dim[0]):
                patches[-1].append([patch_slice[original_dim[0]*j + k]])
        to_add = 0
        for (i, label) in enumerate(label_slice):
            label = np.array(label)
            label_slice[i] = np.where(label == 0, label, label + to_add)
            to_add = np.max(label_slice[i])
        for j in range(original_dim[1]):
            label_patches.append([])
            for k in range(original_dim[0]):
                label_patches[-1].append(label_slice[original_dim[0]*j + k])
        patches = np.array(patches)
        label_patches = np.array(label_patches)
        unpatched = unpatchify(patches, image_size)
        unpatched_labels = unpatchify(label_patches, image_size[:-1])

        images_out.append(unpatched)
        labels_out.append(unpatched_labels)
    
    return images_out, labels_out


def plot_img_label(img, lbl, pred, img_title="image", lbl_title="label", pred_title="predicted", **kwargs):
    fig, (ai,al,ap) = plt.subplots(1,3, figsize=(12,5), gridspec_kw=dict(width_ratios=(1.25,1,1)))
    im = ai.imshow(img, cmap='gray', clim=(0,1))
    ai.set_title(img_title)    
    al.imshow(lbl, cmap=lbl_cmap)
    al.set_title(lbl_title)
    ap.imshow(pred, cmap=lbl_cmap)
    ap.set_title(pred_title)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    IMG_WIDTH = 7680
    IMG_HEIGHT = 4608
    IMG_CHANNELS = 3

    UNPATCH = False

    CRC_SAMPLE = 0

    TRAIN_PATH = '../consep_dataset/CoNSeP/Train/'
    TEST_PATH = '../consep_dataset/CoNSeP/Test/'
    CRC_PATH = 'C:/Users/TheNa/Desktop/CRC_Dataset'

    train_ids = []#os.listdir(TRAIN_PATH + "Images/")#[0]]
    test_ids =  []#os.listdir(TEST_PATH  + "Images/")#[0]]
    crc_ids = [os.listdir(CRC_PATH)[CRC_SAMPLE]]

    X_train = np.zeros((len(train_ids)*4, IMG_HEIGHT//2, IMG_WIDTH//2, IMG_CHANNELS), dtype=np.double)
    Y_train = np.zeros((len(train_ids)*4, IMG_HEIGHT//2, IMG_WIDTH//2), dtype=np.int32)

    # patchify(input, (512, 512, 3), step=512)

    print('Resizing training images and masks')
    for n, id_ in tqdm(enumerate(train_ids), total=len(train_ids)):
        img = imread(TRAIN_PATH + "Images/" + id_)[:,:,:IMG_CHANNELS]
        img = resize(img, (IMG_HEIGHT, IMG_WIDTH), mode='constant', preserve_range=True)
        x_patches = patchify(img, (512, 512, 3), step=512)
        for i in range(2):
            for j in range(2):
                X_train[n*4+i*2+j] = normalize(x_patches[i,j,:,:])  #Fill empty X_train with values from img
        #X_train[n] = normalize(img)
        mask = np.zeros((IMG_HEIGHT, IMG_WIDTH), dtype=np.int32)
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
    X_test = np.zeros((len(test_ids)*4, IMG_HEIGHT//2, IMG_WIDTH//2, IMG_CHANNELS), dtype=np.double)
    Y_test = np.zeros((len(test_ids)*4, IMG_HEIGHT//2, IMG_WIDTH//2), dtype=np.int32)
    sizes_test = []
    print('Resizing test images')
    for n, id_ in tqdm(enumerate(test_ids), total=len(test_ids)):
        img = imread(TEST_PATH + "Images/" + id_)[:,:,:IMG_CHANNELS]
        sizes_test.append([img.shape[0], img.shape[1]])
        img = resize(img, (IMG_HEIGHT, IMG_WIDTH), mode='constant', preserve_range=True)
        x_patches = patchify(img, (512, 512, 3), step=512)
        for i in range(2):
            for j in range(2):
                X_test[n*4+i*2+j] = normalize(x_patches[i,j,:,:])
        #X_test[n] = img
        mask = np.zeros((IMG_HEIGHT, IMG_WIDTH), dtype=np.int32)
        mat = scipy.io.loadmat(TEST_PATH + 'Labels/' + id_[:-4] + '.mat')
        mask_ = mat["inst_map"]#np.asarray([[1 if x > 0 else 0 for x in row] for row in mat["inst_map"]])
        mask_ = resize(mask_, (IMG_HEIGHT, IMG_WIDTH), mode='constant', preserve_range=True)
        mask = np.maximum(mask, mask_)
            
        y_patches = patchify(mask, (512, 512), step=512)
        for i in range(2):
            for j in range(2):
                Y_test[n*4+i*2+j] = y_patches[i,j,:,:]
        #Y_test[n] = mask
    
    patches_x = IMG_WIDTH//512
    patches_y = IMG_HEIGHT//512

    X_crc = np.zeros((len(crc_ids)*patches_x*patches_y, IMG_HEIGHT//patches_y, IMG_WIDTH//patches_x, IMG_CHANNELS), dtype=np.double)
    print('Resizing crc images')
    for n, id_ in tqdm(enumerate(crc_ids), total=len(crc_ids)):
        img = imread(CRC_PATH + "/" + id_)[:,:,:IMG_CHANNELS]
        img = resize(img, (IMG_HEIGHT, IMG_WIDTH), mode='constant', preserve_range=True)
        x_patches = patchify(img, (512, 512, 3), step=512)
        for i in range(patches_y):
            for j in range(patches_x):
                X_crc[n*patches_x*patches_y+i*patches_x+j] = normalize(x_patches[i,j,:,:])

    X_val = X_train[int(len(X_train)*0.8):]
    Y_val = Y_train[int(len(X_train)*0.8):]
    X_train = X_train[:int(len(X_train)*0.8)]
    Y_train = Y_train[:int(len(X_train)*0.8)]

    print('Done!')

    lbl_cmap = random_label_cmap()

    #plt.subplot(121); plt.imshow(X_train[0],cmap='gray');   plt.axis('off'); plt.title('Raw image')
    #plt.subplot(122); plt.imshow(Y_train[0],cmap=lbl_cmap); plt.axis('off'); plt.title('GT labels')
    #plt.show()

    n_rays = 32
    use_gpu = True

    conf = Config2D(
        n_rays       = n_rays,
        use_gpu      = use_gpu and gputools_available(),
        n_channel_in = IMG_CHANNELS,
    )

    #model = StarDist2D(conf, name='stardist_2', basedir='models')
    model = StarDist2D(None, name="stardist_2", basedir="models")

    #model.optimize_thresholds(X_val, Y_val)

    #model.train(X_train[:int(len(X_train)*0.8)], Y_train[:int(len(X_train)*0.8)], validation_data=(X_train[int(len(X_train)*0.8):],Y_train[int(len(X_train)*0.8):]), augmenter=None)

    Y_test_pred = np.array([model.predict_instances(x)[0]#, n_tiles=model._guess_n_tiles(x), show_tile_progress=False)
              for x in tqdm(X_crc)])
    
    """for i in range(len(X_crc)):
        plot_img_label(X_crc[i],Y_test_pred[i], Y_test_pred[i][0])
        labels, details = Y_test_pred[i]
        plt.figure(figsize=(13,10))
        coord, points, prob = details['coord'], details['points'], details['prob']
        plt.subplot(121); plt.imshow(X_crc[i]); plt.axis('off')
        a = plt.axis()
        _draw_polygons(coord, points, prob, show_dist=True)
        plt.axis(a)
        #plt.subplot(122); plt.imshow(Y_test[i]); plt.axis('off')
        plt.imshow(labels, cmap=lbl_cmap, alpha=0.5)
        plt.tight_layout()
        plt.show()"""
    for label in Y_test_pred:
        imshow(label, cmap=lbl_cmap)
        plt.show()
    klsd
    # UNPATCHIFY (needs to be non-static)
    if UNPATCH:
        X_unpatched, Y_unpatched = reconstruct(X_crc, Y_test_pred, original_dim=(patches_x,patches_y))
    else:
        X_unpatched, Y_unpatched = [], []
    """imshow(X_unpatched[0])
    plt.show()
    imshow(Y_unpatched[0])
    plt.show()"""

    from skimage.measure import regionprops_table
    import pandas as pd

    print("Getting nuclei properties...")
    if UNPATCH:
        with open("C:/Users/TheNa/Desktop/properties.csv", "a") as outfile:
            for i, img in tqdm(enumerate(X_unpatched), total=len(X_unpatched)):
                label = Y_unpatched[i]
                df = pd.DataFrame(regionprops_table(label, img, 
                                                    properties=["label", "centroid",
                                                                "area", "eccentricity", "equivalent_diameter_area",
                                                                #"intensity_mean", 
                                                                "solidity"]))
                out = ""

                lines = df.to_csv().split("\n")
                for line in lines[1:]:
                    if len(line) <= 1: continue
                    out += f"{CRC_SAMPLE+i}," + line[line.index(",")+1:]
                outfile.write(out)
    else:
        with open("C:/Users/TheNa/Desktop/properties.csv", "a") as outfile:
            #outfile.write("image,label,centroid-0,centroid-1,area,eccentricity,equivalent_diameter_area,solidity\n")
            patches_per = (patches_x*patches_y)
            if len(X_crc) % patches_per != 0:
                print("Error: uneven number of images for given dimensions")

            patch_size = X_crc[0].shape
            for i in tqdm(range(len(X_crc) // patches_per)):
                patch_slice = X_crc[i*patches_per:i*patches_per + patches_per]
                label_slice = Y_test_pred[i*patches_per:i*patches_per + patches_per]
                to_add = 0
                print("")
                for j in range(patches_y):
                    for k in range(patches_x):
                        if (patches_x*j + k) % 10 == 0:
                            print(f"\tFinding properties for patch {patches_x*j + k}/135")
                        image = patch_slice[patches_x*j + k]
                        label = label_slice[patches_x*j + k]
                        df = pd.DataFrame(regionprops_table(label, image, 
                                                    properties=["label", "centroid",
                                                                "area", "eccentricity", "equivalent_diameter_area",
                                                                #"intensity_mean", 
                                                                "solidity"]))
                        out = ""

                        lines = df.to_csv().split("\n")
                        for line in lines[1:]:
                            if len(line) <= 1: continue
                            items = line.split(",")
                            #           image         	label	                       centroid-0   	                   centroid-1            	area	eccentricity equivalent_diameter_area	solidity
                            out += f"{CRC_SAMPLE + i},{int(items[1]) + to_add},{float(items[2])+patch_size[1]*j},{float(items[3])+patch_size[0]*k},{items[4]},{items[5]},{items[6]},{items[7]}"
                        outfile.write(out)
                        to_add += np.max(label)

    imshow(X_crc[0])
    plt.show()
