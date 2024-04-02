# -*- coding: utf-8 -*-
"""
Created on Fri Jun 18 17:12:45 2021

@author: liy45
"""

''' 3D diagnostic system of chronic otitis media based on temporal bone CT scans'''


import os
# import zipfile
import numpy as np
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers
# from PIL import Image
# import cv2
# import random
import matplotlib.pyplot as plt

from scipy import ndimage
# import re
import pandas as pd
# from sklearn.model_selection import StratifiedKFold, train_test_split
# from sklearn.metrics import roc_curve, auc, confusion_matrix
# import seaborn as sn

import glob
import pydicom as dicom

import torch
import io
import os
from contextlib import redirect_stdout

print(torch.__version__) 
torch.cuda.is_available() 
torch.cuda.device_count() 

''' Step 1. extract ROI from original CT scans'''

def get_scan_info(subject_folder): # may refer to wh script if separate reading from each series is required.
    scans = [f.name for f in os.scandir(subject_folder) if f.is_dir()]
    if len(scans) == 1: # most common situation, one scan per folder, so DICOMDIR contains only one modality
        dicomdir_path = os.path.join(subject_folder,'DICOMDIR')
        print(dicomdir_path)
        try:
            dicomdir = dicom.dcmread(dicomdir_path)
            pt_ID = dicomdir.DirectoryRecordSequence[0].PatientID
            pt_DOB = dicomdir.DirectoryRecordSequence[0].PatientBirthDate
            pt_sex = dicomdir.DirectoryRecordSequence[0].PatientSex 
            scan_date = dicomdir.DirectoryRecordSequence[1].StudyDate
            # scan_type = dicomdir.DirectoryRecordSequence[1].ModalitiesInStudy
            ''' to generate a dataframe for all series'''
            
            series_path_list = []
            num_image = [] # How many slides per scan
            modality = [] # CT or MR
            plane = [] # Axial; Coronary; Sagittal 
            image_type = [] # MR: T1 or T2; CT: soft-tissue window (40) and bone window (60,70)
            thickness = [] 
               
            series_name = [f.name for f in os.scandir(os.path.join(subject_folder, scans[0])) if f.is_dir()]   
            for series in series_name:
                dicom_files = []
                file_path = os.path.join(os.path.join(subject_folder, scans[0]), series)
                for fname in glob.glob(file_path + os.sep + '*', recursive=False):
                    if not fname.endswith('VERSION'):
                        dicom_files.append(fname)
                if len(dicom_files) > 3: # rule out those scans with only 1-3 images; 
                    ds_temp = dicom.dcmread(dicom_files[0])
                    if ds_temp.Modality == 'CT':
                        # scan_ID.append(scan)
                        series_path_list.append(file_path)
                        num_image.append(len(dicom_files))
                        modality.append('CT')               
                        try:
                            plane.append(ds_temp[0x07a1, 0x1047].value)                  
                        except: 
                            plane.append('Unknown')                
                        try: 
                            image_type.append(ds_temp[0x0018, 0x1210].value) 
                        except:
                            image_type.append('Unknown') 
                        thickness.append(ds_temp[0x0018, 0x0050].value)

            df_series = pd.DataFrame(list(zip(modality, num_image, plane, image_type,  thickness, series_path_list)), 
                                      columns =['Modality', 'Number of images', 'Plane', 'Type', 'Thickness', 'Path']) 
            
            df_series['ID'] = pt_ID
            df_series['DOB'] = pt_DOB
            df_series['Sex'] = pt_sex
            df_series['Date'] = scan_date
            df_series = df_series[['ID', 'DOB', 'Sex', 'Date', 'Modality', 'Number of images', 'Plane', 'Type', 'Thickness', 'Path']]
            df_read = df_series.loc[(df_series['Thickness'] < 1) & (df_series['Plane'] == 'AXIAL')] 
            return df_read.reset_index(drop=True)
        except:
            print('This case is not parsed.')   
    else:        
        print('This patient has more than one scans.')     

''' Step 2. genearte images by specifying plane, weight, thickness'''

def load_scan(path):
    slices = [dicom.read_file(os.path.join(path, s)) for s in os.listdir(path) if not s.endswith('VERSION')]
    slices.sort(key = lambda x: float(x.ImagePositionPatient[2]))
    return slices 


def get_pixels_hu(slices):
    image = np.stack([s.pixel_array for s in slices], axis = -1)
    # Convert to int16 (from sometimes int16), 
    # should be possible as values should always be low enough (<32k)
    image = image.astype(np.int16)
    # Set outside-of-scan pixels to 0
    # The intercept is usually -1024, so air is approximately 0
    image[image == -2000] = 0    
    # Convert to Hounsfield units (HU)
    for slice_number in range(len(slices)):        
        intercept = slices[slice_number].RescaleIntercept
        slope = slices[slice_number].RescaleSlope        
        if slope != 1:
            image[:,:, slice_number] = slope * image[:,:, slice_number].astype(np.float64)
            image[:,:, slice_number] = image[:,:, slice_number].astype(np.int16)            
        image[:,:, slice_number] += np.int16(intercept)
    
    return np.array(image, dtype=np.int16)

def resample(image, scan, new_spacing=[1,1,1]):
    # Determine current pixel spacing
    spacing = np.array([scan[0].PixelSpacing[0], scan[0].PixelSpacing[1], scan[0].SliceThickness], dtype=np.float32)
 
    resize_factor = spacing / new_spacing
    new_real_shape = image.shape * resize_factor
    new_shape = np.round(new_real_shape)
    real_resize_factor = new_shape / image.shape
    new_spacing = spacing / real_resize_factor
    
    image = ndimage.zoom(image, real_resize_factor, mode='nearest')
    return image, new_spacing


def normalize_ct(image):
    """Normalize the volume"""
    # min = -1000
    # max = 400
    # volume[volume < min] = min
    # volume[volume > max] = max
    image_scaled = (image - image.min()) / (image.max() - image.min()) * 255.
    image_scaled = image_scaled.astype("float32")
    return image_scaled


def process_scan(path):
    """Read and resize volume"""    
    scan = load_scan(path) # Read scan    
    image = get_pixels_hu(scan) # rescale    
    pix_resampled, spacing = resample(image, scan, [0.5, 0.5, 0.5])    # Resize width, height and depth   
    img_scaled = normalize_ct(pix_resampled) # Normalize
    return img_scaled



''' Step 3. detect and generate ROI as numpy arrays'''

def scan_through(dicom_array, model_roi):
    result_list = []   
    for i in range(dicom_array.shape[-1]):
        results = model_roi(dicom_array[...,i])
        result_list.append(results.xywh[0].tolist())
       
    im_size = dicom_array.shape[0]    
    img_list, center_xy, side, obj_class, obj_prob = [], [], [], [], []
    
    for i in range(len(result_list)):
        if not len(result_list[i]) == 0:
            # i = 85
            for j in range(len(result_list[i])):
                center_xy.append(result_list[i][j][:2])
                side.append('left') if result_list[i][j][0] > im_size/2 else side.append('right')
                obj_prob.append(result_list[i][j][4])
                obj_class.append(result_list[i][j][5])
                img_list.append(i)
    
    return pd.DataFrame(list(zip(img_list, center_xy, side, obj_class, obj_prob)),
                        columns=['image', 'xy', 'side', 'object', 'prob'])


def get_center_img_info(df):   
    df_sum = df.groupby(['side', 'image'])['prob'].sum().reset_index()
    df_left = df_sum.loc[df_sum['side'] == 'left']
    center_img_left = df_left.loc[df_left['prob'].idxmax(), 'image']    
    df_right = df_sum.loc[df_sum['side'] == 'right']   
    center_img_right = df_right.loc[df_right['prob'].idxmax(), 'image']
    try:
        center_xy_left = df.loc[(df['side'] == 'left') & (df['image'] == center_img_left) & (df['object'] == 0), 'xy'].item()
    except:
        df_temp = df.loc[(df['side'] == 'left') & (df['image'] == center_img_left) & (df['object'] == 0)]       
        center_xy_left = df_temp.loc[df_temp['prob'] == df_temp['prob'].max(), 'xy'].item()
    try:    
        center_xy_right = df.loc[(df['side'] == 'right') & (df['image'] == center_img_right) & (df['object'] == 0), 'xy'].item()
    except:
        df_temp = df.loc[(df['side'] == 'right') & (df['image'] == center_img_right) & (df['object'] == 0)]     
        center_xy_right = df_temp.loc[df_temp['prob'] == df_temp['prob'].max(), 'xy'].item()

    return center_img_left, center_xy_left, center_img_right, center_xy_right


def crop_ROI(dicom_array, center_img_layer, center_xy, slice_num = 10, box_size = 128):  
    center_xy = [float(i) for i in center_xy]
    ROI_xyxy = [int(center_xy[0]-box_size/2.0), int(center_xy[1]-box_size/2.0), int(center_xy[0]+box_size/2.0), int(center_xy[1]+box_size/2.0)]   
    img_crop_list = []
    for i in range(center_img_layer - slice_num, center_img_layer + slice_num):
        img_crop_list.append(dicom_array[ROI_xyxy[1]:ROI_xyxy[3], ROI_xyxy[0] : ROI_xyxy[2], i])    
    return np.stack(img_crop_list, axis=-1)


''' Optional: rescale and plot'''

def resize_img(img, desired_height = 128, desired_width = 128):
    """Resize across z-axis"""
    # Set the desired depth
    # Get current depth    
    current_width = img.shape[1]
    current_height = img.shape[0]
    # Compute depth factor    
    width = current_width / desired_width
    height = current_height / desired_height    
    width_factor = 1 / width
    height_factor = 1 / height
    # Rotate
    # img = ndimage.rotate(img, 90, reshape=False)
    # Resize across z-axis
    img = ndimage.zoom(img, (height_factor, width_factor), order=1)
    return img

def plot_slices(data, save_fig=True):
    """Plot a montage of CT slices"""    
    rows = 4
    columns = 8
    f, axarr = plt.subplots(rows, columns, figsize=(2*columns, 2*rows))
    for i in range(rows):
        for j in range(columns):
            axarr[i, j].imshow(data[...,i*columns+j], cmap="gray")
            axarr[i, j].axis("off")
    plt.subplots_adjust(wspace=0, hspace=0, left=0, right=1, bottom=0, top=1)
    plt.tight_layout(pad = 0.5)
    if save_fig:
        plt.savefig(r'C:\Users\liy45\Desktop\mygraph.png')
    plt.show()


''' Step 4. predict outcome'''

def get_model(width=128, height=128, depth=32):
    """Build a 3D convolutional neural network model."""
    inputs = keras.Input((width, height, depth, 1))
    
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

    outputs = layers.Dense(units=2, activation="softmax")(x) # Change output number

    # Define the model.
    model = keras.Model(inputs, outputs, name="3dcnn")
    return model


''' Step 5. plot outcome with heatmap'''

def plot_test_slices(data, case_id, y_predict, save_fig=True):
    """Plot a montage of CT slices"""    
    rows = 4
    columns = 8
    f, axarr = plt.subplots(rows, columns, figsize=(2*columns, 2*rows))
    for i in range(rows):
        for j in range(columns):
            axarr[i, j].imshow(data[...,i*columns+j], cmap="gray")
            axarr[i, j].axis("off")
    plt.subplots_adjust(wspace=0, hspace=0, left=0, right=1, bottom=0, top=1)
    pred_txt = 'Non-cholesteatoma' if  y_predict == 0 else 'Cholesteatoma'

    f.suptitle('{} : {}'.format(case_id, pred_txt), fontsize = 28)
    f.tight_layout(pad = 0.3)
    f.subplots_adjust(top=0.92)
    if save_fig:
        plt.savefig(r'C:\Users\liy45\Desktop\mygraph.png')
    plt.show()

def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None, resize = True):
   
    grad_model = tf.keras.models.Model(
        [model.inputs], [model.get_layer(last_conv_layer_name).output, model.output])
    
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
    f, axarr = plt.subplots(rows, columns, figsize=(2*columns, 2*rows))
    for i in range(rows):     
        for j in range(columns):
            axarr[i, j].matshow(heatmap[:, :, i*columns + j], cmap=cmap)
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

def display_superimpose_imgs(out_root_folder,scan_id,target_side,img_array, heatmap_resize, preds, ch, alpha = 0.5,  rows = 4, columns = 8, 
                             show_origin = True, show_heatmap = True, cmap = 'bwr', threshold = 0.45):
    
    assert rows * columns == img_array.shape[3], "Mismatched size!"
    class_dict = {0: 'Non-Cholesteatoma',
                  1: 'Cholesteatoma'}
    probs = preds[0][1]
    # y_pred_lbl = 1 if probs >= optimal_threshold else 0 
    y_pred_lbl = 1 if probs >= threshold else 0               
    f, axarr = plt.subplots(rows, columns, figsize=(2*columns, 2*rows))
    for i in range(rows):        
        for j in range(columns):
            if show_origin:
                axarr[i, j].imshow(np.squeeze(img_array[:, :, :, i*columns + j, :]), interpolation='gaussian', cmap='gray')
            if show_heatmap:
                axarr[i, j].imshow(np.squeeze(heatmap_resize[:, :, i*columns + j])*probs, cmap=cmap, interpolation='gaussian',
                                    vmin=0, vmax=1, alpha = alpha)           
            axarr[i, j].text(0.5, 0.1, str(i*columns + j + 1), transform=axarr[i, j].transAxes, size=14, weight='bold', color = 'w')
            axarr[i, j].axis('off')
            # plt.show()   
    f.suptitle('Prediction: {} ({:.1%})'.format(class_dict[y_pred_lbl], preds[0].max()), fontsize = 36)
    f.tight_layout(pad = 0.3)
    f.subplots_adjust(top=0.92)
    plt.savefig(os.path.join(out_root_folder, '{} {}.png'.format(scan_id, target_side)))
    plt.show()

def get_tensorflow_last_layer(model_3dcom):
    summary_string_io = io.StringIO()
    with redirect_stdout(summary_string_io):
        model_3dcom.summary()

    # Get the summary string and split it into lines
    summary_string = summary_string_io.getvalue()
    summary_lines = summary_string.split('\n')

    # retrieve conv3d_3 layer for heatmap
    tmp_model_last_layer_name = ''
    idx = 0
    for line in summary_lines:
        #print('%d:%s' % (idx,line))
        if idx == 31:
            tmp_model_last_layer_name = line.split(' ')[1]
        idx = idx + 1
    return tmp_model_last_layer_name