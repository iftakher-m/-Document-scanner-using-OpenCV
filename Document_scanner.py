# This is a jupyter notebook file exported as a python script and then uploaded to this repo.(hence, ignore the lines having '#In[ ]') 

# import the necessary packages
from pyimagesearch.transform import four_point_transform 
from skimage.filters import threshold_local

import numpy as np
import cv2
import imutils
import matplotlib.pyplot as plt

# making a function to easily show the image inside the kernel
def display_img(im): 
    fig, ax= plt.subplots(1, figsize=(6,8))    
    ax.imshow(im, cmap='gray')   

# this function easily display a resized image using opencv   
def show_img(img):
    cv2.imshow('Image', imutils.resize(img, height=650))
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    


# In[84]:


# load the image and compute the ratio of the old height to the new height
image= cv2.imread('images/receipt.jpg')
ratio= image.shape[0]/ 500.0

orig= image.copy()

#Resize the image maintaining the aspect ratio, (defining the height will set the width accordingly)
image= imutils.resize(image, height=500)
# image= cv2.resize(image,(0,0), image,.2,.2)


# In[85]:


# cv2.imshow('k', image[125:375,125:300]) # would be useful when we got to crop it.
# cv2.waitKey(0)


# In[86]:


gray= cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) # Converting to grayscale
gr_blur= cv2.GaussianBlur(image,(5,5),0 )     # blurring the image
edged= cv2.Canny(gr_blur, 75, 200 )           # finding the edges

# show_img(image)
# show_img(edged)


# In[87]:


# find the contours in the edged image, keeping only the largest ones, and initialize the screen contour
cnts= cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
cnts= imutils.grab_contours(cnts)
cnts= sorted(cnts, key= cv2.contourArea, reverse=True)[:5]


# In[88]:


# loop over the contours
for c in cnts:
    # approximate the contour
    peri= cv2.arcLength(c,True)
    approx= cv2.approxPolyDP(c, 0.02*peri, True)
    
    # if our approximated contour has four points, then we can assume that we have found our screen
    if len(approx)==4:
        screenRnt= approx
        break
        
# show the contour (outline) of the piece of paper       
cv2.drawContours(image, [screenRnt], -1, (0,255,0), 2)

show_img(image)
display_img(image)


# In[89]:


#apply the four point transform to obtain a top-down view of the original curved image
warped= four_point_transform(orig, screenRnt.reshape(4,2)*ratio )

#convert the warped image to grayscale, then threshold it to give it that 'black and white' paper effect
warped= cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
T= threshold_local(warped, 11, offset=10, method='gaussian')

warped= (warped>T).astype('uint8')* 255 # This will produce a white background having black fonts

warped_black= (T>warped).astype('uint8')* 255 # This will produce a black background having white fonts


# In[90]:


print(warped) 


# In[91]:


print(T) 


# In[92]:


display_img(T)


# In[93]:


display_img(warped)


# In[94]:


display_img(warped_black)


# In[95]:


show_img(warped)


# In[96]:


show_img(warped_black)

