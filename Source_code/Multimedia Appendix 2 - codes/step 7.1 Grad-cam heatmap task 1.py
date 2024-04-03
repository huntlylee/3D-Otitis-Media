# -*- coding: utf-8 -*-
"""
Created on Mon Jul 19 20:49:21 2021

@author: liy45
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras

# Display
# from IPython.display import Image, display
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from tensorflow.keras import layers
import os
import re
import cv2

from PIL import Image
from scipy import ndimage

import seaborn as sns






def get_img_array(folder):
    images = []
    images_info = os.listdir(folder)
    images_info.sort(key=lambda f: int(re.sub('\D', '', f)))
    for filename in images_info:
        img = cv2.imread(os.path.join(folder,filename), 0)
        if img is not None:
            images.append(img)
            
    img_3d = np.stack(images, axis = 2)
    images_np = np.expand_dims(img_3d/255, axis=0)
    images_np = np.expand_dims(images_np, axis=-1)
    
    return images_np.astype("float32")



def get_model(width=128, height=128, depth=32):
    """Build a 3D convolutional neural network model."""

    inputs = keras.Input((width, height, depth, 1))
    
    # original kernel size (3, 3, 2)

    x = layers.Conv3D(filters=64, kernel_size = (3, 3, 3), activation="relu", padding='same')(inputs)
    x = layers.MaxPool3D(pool_size=2)(x)
    x = layers.BatchNormalization()(x)

    x = layers.Conv3D(filters=64, kernel_size = (3, 3, 3), activation="relu", padding='same')(x)
    x = layers.MaxPool3D(pool_size=2)(x)
    x = layers.BatchNormalization()(x)

    x = layers.Conv3D(filters=128, kernel_size = (3, 3, 3), activation="relu", padding='same')(x)
    x = layers.MaxPool3D(pool_size=2)(x)
    x = layers.BatchNormalization()(x)

    x = layers.Conv3D(filters=256, kernel_size = (3, 3, 3), activation="relu", padding='same')(x)
    x = layers.MaxPool3D(pool_size=2)(x)
    x = layers.BatchNormalization()(x)

    x = layers.GlobalAveragePooling3D()(x)
    x = layers.Dense(units=64, activation="relu")(x)
    x = layers.Dropout(0.3)(x)

    outputs = layers.Dense(units=2, activation="softmax")(x)

    # Define the model.
    model = keras.Model(inputs, outputs, name="3dcnn")
    return model

model = get_model(width=128, height=128, depth=32)
model.summary()




def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None, resize = True):

    # First, we create a model that maps the input image to the activations
    # of the last conv layer as well as the output predictions
    
    # model.layers[-1].activation = None
   
    grad_model = tf.keras.models.Model(
        [model.inputs], [model.get_layer(last_conv_layer_name).output, model.output]
    )
    
    # Then, we compute the gradient of the top predicted class for our input image
    # with respect to the activations of the last conv layer
    ''' 1st approach'''
    # with tf.GradientTape() as tape:
    #     last_conv_layer_output, preds = grad_model(img_array)
    #     if pred_index is None:
    #         pred_index = tf.argmax(preds[0])
    #     class_channel = preds[:, pred_index]
    
    # # This is the gradient of the output neuron (top predicted or chosen)
    # # with regard to the output feature map of the last conv layer
    # grads = tape.gradient(class_channel, last_conv_layer_output)
    
    # # This is a vector where each entry is the mean intensity of the gradient
    # # over a specific feature map channel
    # pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2, 3))
    
    # # We multiply each channel in the feature map array
    # # by "how important this channel is" with regard to the top predicted class
    # # then sum all the channels to obtain the heatmap class activation
    # last_conv_layer_output = last_conv_layer_output[0]
    # heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    # heatmap = tf.squeeze(heatmap) #Removes dimensions of size 1 from the shape of a tensor.
    
    # # For visualization purpose, we will also normalize the heatmap between 0 & 1
    # heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    
    
    ''' 2nd approach, and resize'''
    if pred_index is None:
        pred_index = 0 
    
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        loss = predictions[:, pred_index]

    # Extract filters and gradients
    output = conv_outputs[0]
    grads = tape.gradient(loss, conv_outputs)[0]
    
    # Average gradients spatially
    weights = tf.reduce_mean(grads, axis=(0, 1, 2))
    # Build a ponderated map of filters according to gradients importance
    cam = np.zeros(output.shape[0:3], dtype=np.float32)
    
    for index, w in enumerate(weights):
        cam += w * output[:, :, :, index]    
        
    if resize:
        
        capi = resize_volume(cam, method = 'grid-constant')
    else:
        capi = cam
        #print(capi.shape)
    capi = np.maximum(capi,0)
    heatmap = (capi - capi.min()) / (capi.max() - capi.min())
           
    return heatmap#.numpy()


# Display heatmap

def display_heatmap(heatmap, rows, columns, cmap):
    
    # num_subplot = heatmap.shape[-1]

    f, axarr = plt.subplots(
        rows,
        columns,
        figsize=(2*columns, 2*rows),
    )

    for i in range(rows):
        
        for j in range(columns):
            axarr[i, j].matshow(heatmap[:, :, i*columns + j], cmap=cmap)
            # axarr[i, j].title.set_text('Layer {}'.format(i*columns + j + 1))
            axarr[i, j].text(0.5, 0.1, str(i*columns + j + 1), transform=axarr[i, j].transAxes, size=14, weight='bold', color = 'w')
            axarr[i, j].axis("off")
    f.tight_layout(pad = 0.3)
    f.subplots_adjust(top=0.92)
    
    plt.show()


def resize_volume(img, method = 'grid-constant'):
    """Resize across z-axis"""
    # Set the desired depth
    desired_depth = 32
    desired_width = 128
    desired_height = 128
    # Get current depth
    current_depth = img.shape[-1]
    current_width = img.shape[1]
    current_height = img.shape[0]
    # Compute depth factor
    depth = current_depth / desired_depth
    width = current_width / desired_width
    height = current_height / desired_height
    depth_factor = 1 / depth
    width_factor = 1 / width
    height_factor = 1 / height
    # Rotate
    # img = ndimage.rotate(img, 90, reshape=False)
    # Resize across z-axis
    img = ndimage.zoom(img, (height_factor, width_factor, depth_factor), order=2, mode = method)
    img = (img - img.min()) / (img.max() - img.min())
    return img





# def superimpose_gradcam(img, heatmap, alpha=0.4):
    
#     bg_img = Image.fromarray(np.uint8(img * 255) , 'L').convert("RGBA") 
#     # bg_img = Image.fromarray(np.uint8(img * 255) , 'L').convert("RGBA") 
#     # bg_img = np.asarray(bg_img)
                  
#     hm_img = Image.fromarray(np.uint8(cm.hot(heatmap)*255), 'RGBA')
#     # hm_img = Image.fromarray(np.uint8(cm.hot(heatmap)*255), 'RGBA')
#     # hm_img = np.asarray(hm_img)
    
#     newimg = Image.blend(bg_img, hm_img, alpha=alpha)
#     # superimposed_img = hm_img * alpha + bg_img
    
#     return newimg
#     # plt.imshow(superimposed_img/255)

# superimpose_gradcam(img_array[0, :, :, 20, 0], heatmap_resize[:, :, 20], 0.6)


# import matplotlib.colors


def display_superimpose_imgs(img_array, heatmap_resize, preds, ch, alpha = 0.5,  rows = 4, columns = 8, 
                             adj_prob = True, show_origin = True, show_heatmap = True, cmap = 'bwr'):
    
    # norm = plt.Normalize(0,1)
    # matplotlib.colors.LinearSegmentedColormap.from_list("", ["white","red"])

    assert rows * columns == img_array.shape[3], "Mismatched size!"
    
    if adj_prob:
        probs = preds[0][1]
    else:
        probs = 1
    
    class_dict ={0: 'Normal',
                1: 'Pathologic'
                }
        
    f, axarr = plt.subplots(
        rows,
        columns,
        figsize=(2*columns, 2*rows),
    )

    for i in range(rows):
        
        for j in range(columns):
            # img_temp  = superimpose_gradcam(np.squeeze(img_array[:, :, :, i*columns + j, :]), np.squeeze(heatmap_resize[:, :, i*columns + j]), alpha=alpha)
            # axarr[i, j].imshow(img_temp)
            # axarr[i, j].axis("off")
            if show_origin:
                axarr[i, j].imshow(np.squeeze(img_array[:, :, :, i*columns + j, :]), interpolation='gaussian', cmap='gray')
            if show_heatmap:
                axarr[i, j].imshow(np.squeeze(heatmap_resize[:, :, i*columns + j])*probs, cmap=cmap, interpolation='gaussian',
                                    vmin=0, vmax=1, alpha = alpha)           
            axarr[i, j].text(0.5, 0.1, str(i*columns + j + 1), transform=axarr[i, j].transAxes, size=14, weight='bold', color = 'w')
            axarr[i, j].axis('off')
            # plt.show()   
    f.suptitle('Prediction: {} ({:.1%})'.format(class_dict[preds[0].argmax()], preds[0].max()), fontsize = 36)

    f.tight_layout(pad = 0.3)
    f.subplots_adjust(top=0.92)
    plt.show()






# Load model
chckpnt_path = r"D:\OneDrive - VUMC\Research\Ongoing Research Projects\My AI projects\AI 3D otitis media\Tasks updated 2022-09-03\Results task 1\fold 1\3d_image_classification_task1_fold1.h5"
model.load_weights(chckpnt_path)
model.summary()

last_conv_layer_name = "conv3d_7"

scan_id = 'p00556360-191224-Right'
dataset = 'Shanghai'



pt_id = scan_id.split('-')[0] 
scan_dt = scan_id.split('-')[1] 
side = scan_id.split('-')[-1] 
img_folder = os.path.join(r'D:\My Data\EENT RDD\Otitis Media\Extracted ROI', dataset, pt_id + '-' + scan_dt, side)
if not os.path.isdir(img_folder):
    img_folder = os.path.join(r'D:\My Data\EENT RDD\Otitis Media\Extracted ROI', dataset, pt_id + '-20' + scan_dt, side)

# Prepare image
img_array = get_img_array(img_folder)

# Print what the top predicted class is
preds = model.predict(img_array)
print("Predicted class: ", preds[0].argmax())

# Generate class activation heatmap
view_ch = 1
heatmap_np = make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index = view_ch, resize = False)


# display_heatmap(heatmap_np, 2, 2)
heatmap_resize = resize_volume(heatmap_np, method = 'constant')

# plt.figure()
# plt.imshow(np.squeeze(heatmap_resize[:, :, 16])*preds[0].max(), cmap='bwr', vmin=-1, vmax=1, interpolation='none')
# plt.imshow(np.squeeze(img_array[:, :, :, 16, :]), interpolation='none', alpha = 0.4)
# plt.axis('off')
# plt.show()

# display_heatmap(heatmap_resize, 4, 8, cmap='coolwarm')
display_superimpose_imgs(img_array, heatmap_resize, preds = preds, ch=view_ch, alpha = 0.4,  rows = 4, columns = 8,
                         adj_prob = True, show_origin = True, show_heatmap = False, cmap = 'coolwarm')

