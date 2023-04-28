from unet import get_model   #Use normal unet model
from keras.utils import normalize
import os
import cv2
from PIL import Image
import numpy as np
from matplotlib import pyplot as plt
from skimage import measure, color, io
from patchify import patchify


IMG_WIDTH = 1024
IMG_HEIGHT = 1024
IMG_CHANNELS = 3

def init_model():
    return get_model(IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS)

#Load the model and corresponding weights
model = init_model()
model.load_weights('unet_model.h5') #Trained for 50 epochs

#Load and process the test image - image that needs to be segmented. 
test_img = cv2.imread('../consep_dataset/CoNSeP/Test/Images/test_5.png')[:,:,:IMG_CHANNELS]
print(test_img.shape)
patches = patchify(test_img, (512, 512, 3), step=512)
test_img = patches[0][0]
#test_img_norm = np.expand_dims(normalize(np.array(test_img), axis=1),2)
#test_img_norm=test_img_norm[:,:,0][:,:,None]
#test_img_input=np.expand_dims(test_img_norm, 0)

#Predict and threshold for values above 0.5 probability
segmented = (model.predict(test_img)[0,:,:,0] > 0.05).astype(np.uint8)
print(segmented.shape)

plt.figure(figsize=(8, 8))
plt.subplot(221)
plt.title('Testing Image')
plt.imshow(test_img[0], cmap='gray')
plt.subplot(222)
plt.title('Segmented Image')
plt.imshow(segmented, cmap='gray')
plt.show()

plt.imsave('output.jpg', segmented, cmap='gray')

########################################################
#####Watershed

img = cv2.imread('output.jpg')  #Read as color (3 channels)
img_grey = img[:,:,0]

## transform the unet result to binary image
#Threshold image to binary using OTSU. ALl thresholded pixels will be set to 255
ret1, thresh = cv2.threshold(img_grey, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)


# Morphological operations to remove small noise - opening
#To remove holes we can use closing
kernel = np.ones((3,3),np.uint8)
opening = cv2.morphologyEx(thresh,cv2.MORPH_OPEN,kernel, iterations = 2)

sure_bg = cv2.dilate(opening,kernel,iterations=10)

plt.imshow(sure_bg)
plt.plot()

dist_transform = cv2.distanceTransform(opening,cv2.DIST_L2,5)


#Let us threshold the dist transform by starting at 1/2 its max value.
ret2, sure_fg = cv2.threshold(dist_transform, 0.01*dist_transform.max(),255,0)

plt.imshow(sure_fg)
plt.show()

# Unknown ambiguous region is nothing but bkground - foreground
sure_fg = np.uint8(sure_fg)
unknown = cv2.subtract(sure_bg,sure_fg)

ret3, markers = cv2.connectedComponents(sure_fg)

markers = markers+10

# Now, mark the region of unknown with zero
markers[unknown==255] = 0
#plt.imshow(markers, cmap='gray')   #Look at the 3 distinct regions.

#Now we are ready for watershed filling. 
markers = cv2.watershed(img, markers)

#Let us color boundaries in yellow. 
img[markers == -1] = [0,255,255]  

img2 = color.label2rgb(markers, bg_label=0)

cv2.imshow('Overlay on original image', img)
cv2.imshow('Colored Grains', img2)
cv2.waitKey(0)


props = measure.regionprops_table(markers, intensity_image=img_grey, 
                              properties=['label',
                                          'area', 'equivalent_diameter',
                                          'mean_intensity', 'solidity'])
    
import pandas as pd
df = pd.DataFrame(props)
df = df[df.mean_intensity > 100]  #Remove background or other regions that may be counted as objects

print(df.head())



