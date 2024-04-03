# -*- coding: utf-8 -*-
"""
Created on Mon Jan 11 21:09:20 2021

@author: Yike Li

"""

'''extract dicom files as images and case info'''


import numpy as np

import glob

import os

import pydicom as dicom

import cv2
import pandas as pd
import pathlib
from scipy import ndimage


''' Step 1: to read all scans in a folder and extract key info per scan.
    A scan is defined as a collection of images of the same modality taken from one person at a time
    e.g. If a person takes a CT and an MR on the same day, there are two scans;
    If a person takes a CT in one day and another CT on a different day, they are also two scans.
''' 



out_root_folder = r'D:\My Data\EENT RDD\Otitis Media\Extracted images\Wuhan 022122'

pt_file_path =  r'C:\Users\liy45\Desktop\pt info.csv'

scan_file_path =  r'C:\Users\liy45\Desktop\scan info.csv'



def get_scan_info(folder):
    
    scan_folders = [f.name for f in os.scandir(folder) if f.is_dir()] # get the names of all scan folders

    scanned_folder_list = []
    name_list = []
    pt_ID_dicom = []
    pt_DOB_dicom = []
    pt_sex_dicom = []
    scan_date_dicom = []
    scan_type_dicom = []
    
    for i in range(len(scan_folders)):
        scan_path = os.path.join(scan_root_folder, scan_folders[i])
        scan_subfolder = [f.name for f in os.scandir(scan_path) if f.is_dir()]
        
        ''' series in the same folder may belong to different patients, so only read by series, not dicomdir'''
     
        ''' in some cases, a person takes a CT and an MR on the same day, 
        these two scans can be contained in the same folder, therefore reading the dicomdir does not give all of the info
        '''
        if len(scan_subfolder) > 0:
            for j in range(len(scan_subfolder)): # to read each folder instead
                series_folder_temp = [f.path for f in os.scandir(os.path.join(scan_path, scan_subfolder[j])) if f.is_dir()]
                file_temp = [f.path for f in os.scandir(series_folder_temp[0]) if (f.is_file() & ~(f.name.endswith('VERSION')))]
                # file_temp = [r'D:\My Data\EENT RDD\Otitis Media\Imaging\CT\Wuhan\WH-1\1\2URPHDDC\HPM5WUVT\I1000000']
                try:
                    dicomdir = dicom.dcmread(file_temp[0])
                    
                    scanned_folder_list.append(scan_folders[i])
                    pt_ID_dicom.append(dicomdir.PatientID)
                    name_list.append(dicomdir.PatientName.family_name)
                    pt_DOB_dicom.append(dicomdir.PatientBirthDate)
                    pt_sex_dicom.append(dicomdir.PatientSex) 
                    scan_date_dicom.append(dicomdir.StudyDate) 
                    scan_type_dicom.append(dicomdir.Modality) 
                except:
                    print('Case {} is not parsed'.format(scan_folders[i] + '-' + scan_subfolder[j]))  
        else: # empty folder
            print('Case {} is not parsed'.format(scan_folders[i]))     
    
    df_patient = pd.DataFrame(list(zip(scanned_folder_list, pt_ID_dicom, name_list, scan_date_dicom,  pt_DOB_dicom, pt_sex_dicom, scan_type_dicom)), 
                   columns =[ 'ID_folder', 'ID_dicom', 'Name', 'Date_dicom', 'DOB', 'Sex', 'Modality']) 
    

    ''' to generate a dataframe for all series'''
    
    scan_ID = [] # folder name: e.g.: PtID-YYYYMMDD 
    scan_ID_dicom = []
    scan_date_dicom2 = []
    series_path_list = []
    num_image = [] # How many slides per scan
    modality = [] # CT or MR
    plane = [] # Axial; Coronary; Sagittal 
    image_type = [] # MR: T1 or T2; CT: soft-tissue window (40) and bone window (60,70)
    thickness = [] 
    option = [] # MR: FS or not; CT: Not needed
    
    for scan in scan_folders:
        
        # pt_ID_temp = df_patient.loc[df_patient['ID_folder'] == scan, 'ID_dicom'].values[0]       
        # scan_date_temp = df_patient.loc[df_patient['ID_folder'] == scan, 'Date_dicom'].values[0]
        scan_subfolders = [f.name for f in os.scandir(os.path.join(scan_root_folder, scan)) if f.is_dir()]
        
        for i in range(len(scan_subfolders)):
            series_path = os.path.join(scan_root_folder, scan, scan_subfolders[i])
            series_name = [f.name for f in os.scandir(series_path) if f.is_dir()] # get all series names in a scan
            for series in series_name:
                dicom_files = []
                file_path = os.path.join(series_path, series)
                for fname in glob.glob(file_path + '\\*', recursive=False):
                    if not fname.endswith('VERSION'):
                        dicom_files.append(fname)
                if len(dicom_files) > 3: # rule out those scans with only 1-3 images; 
                    ds_temp = dicom.dcmread(dicom_files[0])
                    if ds_temp.Modality == 'CT':
                        scan_ID.append(scan)
                        scan_ID_dicom.append(ds_temp.PatientID)
                        scan_date_dicom2.append(ds_temp.StudyDate)
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
                        option.append('N/A')
                    elif ds_temp.Modality == 'MR':
                        scan_ID.append(scan)
                        scan_ID_dicom.append(ds_temp.PatientID)
                        scan_date_dicom2.append(ds_temp.StudyDate)
                        series_path_list.append(file_path)
                        num_image.append(len(dicom_files))
                        modality.append('MR')
                        try:
                            plane.append(ds_temp[0x07a1, 0x1047].value)
                            
                        except: 
                            plane.append('Unknown')
                        image_type.append(ds_temp.SeriesDescription.split('_')[0])
                        thickness.append(ds_temp[0x0018, 0x0050].value)
                        if 'FS' in ds_temp[0x0018, 0x0022].value:
                            option.append('FS')
                        else:
                            option.append('')
    
    df_series = pd.DataFrame(list(zip(scan_ID, scan_ID_dicom, scan_date_dicom2, modality, num_image, plane, image_type,  thickness, option, series_path_list)), 
                              columns =['Scan', 'Scan_dicom', 'Date', 'Modality', 'Number of images', 'Plane', 'Type', 'Thickness', 'Option', 'Path']) 
       
    df_patient['ID_folder'] = df_patient['ID_folder'].astype(int)
    df_series['Scan'] = df_series['Scan'].astype(int)
    
    return df_patient.sort_values('ID_folder', ignore_index=True), df_series.sort_values('Scan', ignore_index=True)



''' Step 2: genearte images by specifying plane, weight, thickness'''

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
    # Read scan
    scan = load_scan(path)
    # rescale
    image = get_pixels_hu(scan)

    # Resize width, height and depth
    pix_resampled, spacing = resample(image, scan, [0.5, 0.5, 0.5])
    
    # Normalize
    img_scaled = normalize_ct(pix_resampled)

    return img_scaled


def save_img(img_array, out_folder, jpg = True):
    if not os.path.exists(out_folder):
        pathlib.Path(out_folder).mkdir(parents=True, exist_ok=True)
    for i in range(img_array.shape[-1]):
        pixel_array_numpy = img_array[:, :, i]
        
        if jpg:
            out_fname = 'image' + str(i+1) +'.jpg'
        else:
            out_fname = 'image' + str(i+1) +'.png'

        # Image.fromarray(norm.astype(np.uint8)).save(os.path.join(jpg_folder_path,out_fname)) 

        out_path = os.path.join(out_folder, out_fname)
        if not os.path.exists(out_path):
            cv2.imwrite(out_path, np.uint8(pixel_array_numpy))


def batch_extract_images(df_series, out_root_folder, jpg = True):
          
    df_read = df_series.loc[(df_series['Thickness'] <= 1) & (df_series['Number of images'] >= 25)] # for axial thin slice
    
    scan_list = list(df_read['Scan'].unique())
    
    for scan_id in scan_list:
        
        df_temp = df_read.loc[df_read['Scan'] == scan_id]
        if len(df_temp) == 1:
            read_path = df_temp['Path'].values[0]
            pt_id = df_temp['Scan_dicom'].values[0]
            scan_date = df_temp['Date'].values[0] #-1
            modality = 'CT'
        else:
            read_path = df_temp.loc[df_temp['Thickness'] == df_temp['Thickness'].min(), 'Path'].values[0]
            pt_id = df_temp.loc[df_temp['Thickness'] == df_temp['Thickness'].min(), 'Scan_dicom'].values[0]
            scan_date = df_temp.loc[df_temp['Thickness'] == df_temp['Thickness'].min(), 'Date'].values[0] #-1
            modality = 'CT'
        
        print('Processing {}'.format(pt_id + '-' + scan_date))
                 
        out_folder = os.path.join(out_root_folder, pt_id, modality + '-' + scan_date)
        
        if pathlib.Path(out_folder).is_dir() and len(os.listdir(out_folder)) != 0:
            print('Files already exist!')
            continue

        dicom_array = process_scan(read_path)
        
        save_img(dicom_array, out_folder, jpg = jpg)
       

def preprocessing_flow():

    df_patient, df_series = get_scan_info(scan_root_folder)

    df_patient.to_csv(pt_file_path, mode='a', index = False, header=None)
    df_series.to_csv(scan_file_path, mode='a', index = False, header=None)

    batch_extract_images(df_series, out_root_folder)

scan_root_folder =  r'D:\My Data\EENT RDD\Otitis Media\Imaging\CT\Wuhan\WH-2'
preprocessing_flow()
