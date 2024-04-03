# -*- coding: utf-8 -*-
"""
To search the center image of each side in a CT scan and crop the ROI

by YL 

"""

import torch

print(torch.__version__) 
torch.cuda.is_available()
torch.cuda.device_count() 

import os
import pandas as pd
# import glob
# import shutil
from PIL import Image
from pathlib import Path
    
import re

# Set working directory
os.chdir(r'C:\YL AI projects\YOLO\yolov5')

trained_model_path = r'D:\OneDrive - Personal\OneDrive\Shared with Guo\CT\yolov5-master\runs\train\exp\weights\best.pt' 
model = torch.hub.load('ultralytics/yolov5', 'custom', path = trained_model_path)
model.eval()

pt_folder = r'D:\My Data\EENT RDD\Otitis Media\Extracted images\Wuhan 022122'

target_folder = r'D:\My Data\EENT RDD\Otitis Media\Extracted ROI\Wuhan 220221' # should be a different folder than the original img folder

slice_num = 16

def test_single_img(model, img_path, show = True, save = False):
    
    try:
        results = model(img_path)
        if show:
            results.show() 
        if save:
            results.save(r'C:\Users\liy45\Desktop')

    except:
        print ('Path not exist!')
    return results


def test_img_folder(test_dir):
    
    img_list = os.listdir(test_dir)
    
    im = Image.open(os.path.join(test_dir, img_list[0]))
    original_dimension, _ = im.size
    
    result_list = []
    
    for img in img_list:
        results = test_single_img(model = model, img_path = os.path.join(test_dir, img), show = False, save = False)
        result_list.append(results.xywh[0].tolist())
    
    return (img_list, result_list), original_dimension


def single_img_analysis(coordinate_info, original_dimension):
    
    center_xy = []
    side = []
    obj_class = []
    obj_prob = []
    
    for i in range(len(coordinate_info)):
        center_xy.append(coordinate_info[i][:2])
        side.append('left') if coordinate_info[i][0] > original_dimension/2 else side.append('right')
        obj_prob.append(coordinate_info[i][4])
        obj_class.append(coordinate_info[i][5])
    
    df = pd.DataFrame(list(zip(center_xy, side, obj_class, obj_prob)),
                      columns=['xy', 'side', 'object', 'prob'])
    
    return df


def result_summary(result_tuple, original_dimension):
    df_list = []
    for i in range(len(result_tuple[1])):
        if not len(result_tuple[1][i]) == 0:
            df_img = single_img_analysis(result_tuple[1][i], original_dimension)
            df_img['image'] = result_tuple[0][i]

            df_list.append(df_img)

    df_result = pd.concat(df_list)
        
    return df_result

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

   

def crop_ROI(source_folder, target_folder, center_img, center_xy, side, slice_num = 10, box_size = 128):
    
    images_list = os.listdir(source_folder) 
    images_list.sort(key=lambda f: int(re.sub('\D', '', f)))
    center_img_idx = images_list.index(center_img)
    
    '''crop'''  
    center_xy = [float(i) for i in center_xy]
    ROI_xyxy = [int(center_xy[0]-box_size/2.0), int(center_xy[1]-box_size/2.0), int(center_xy[0]+box_size/2.0), int(center_xy[1]+box_size/2.0)]
    
    target_dir = os.path.join(target_folder, side)
    Path(target_dir).mkdir(parents=True, exist_ok=True)
    
    for i in range(center_img_idx - slice_num, center_img_idx + slice_num):
            img_orgn = Image.open(os.path.join(source_folder,images_list[i]))
            img_crop = img_orgn.crop(ROI_xyxy)
            # img_array = image.img_to_array(img_crop)
            save_path = os.path.join(target_dir, images_list[i])
            img_crop.save(save_path)
    
      

def preprocessing_flow(pt_folder, target_folder):
    
    problem_list = []
    
    for pt_id in os.listdir(pt_folder):
        
        for scan_id in os.listdir(os.path.join(pt_folder, pt_id)):
            
            if scan_id.split('-')[0] == 'CT':
                scan_date = scan_id.split('-')[1]
                print('Processing {}'.format(pt_id + '-' + scan_date))
                
                target_dir = os.path.join(target_folder, pt_id + '-' + scan_date)

                if Path(target_dir).is_dir():
                    print('ROI images already created.')
                    continue                   
                source_dir = os.path.join(pt_folder, pt_id, scan_id)
       
                try:         
                    result_tuple, original_dimension = test_img_folder(source_dir)                
                    df = result_summary(result_tuple, original_dimension)               
                    center_img_left, center_xy_left, center_img_right, center_xy_right = get_center_img_info(df)                               
                    crop_ROI(source_dir, target_dir, center_img_left, center_xy_left, side='Left', slice_num = slice_num, box_size = 128)                                    
                    crop_ROI(source_dir, target_dir, center_img_right, center_xy_right, side='Right', slice_num = slice_num, box_size = 128)                    
                except:                   
                    problem_list.append(pt_id + '-' + scan_date)
                    
    return problem_list
                
problem_list = preprocessing_flow(pt_folder, target_folder)



''' if a single scan needs to be processed'''
            
            
def process_signle_scan(source_folder, target_folder, side, center_img):
    
    pt_id = source_folder.split('\\')[-2]
    scan_date = source_folder.split('\\')[-1].split('-')[-1]
    
    target_dir = os.path.join(target_folder, pt_id + '-' + scan_date)

    Path(target_dir).mkdir(parents=True, exist_ok=True)              
       
    try:         
        result_tuple, original_dimension = test_img_folder(source_folder)                
        df = result_summary(result_tuple, original_dimension)

        # return df              
       
        center_xy = df.loc[(df['image'] == center_img) & (df['side'] == side.lower()) & (df['object'] == 0), 'xy'].item()                                  
        crop_ROI(source_folder, target_dir, center_img, center_xy, side=side, slice_num = slice_num, box_size = 128)       
    except:                   
        print('Problem processing imgs')
              
single_source_folder = r'D:\My Data\EENT RDD\Otitis Media\Extracted images\standard_ct_wh\P4739502\CT-20210315'         
target_folder = r'C:\Users\liy45\Desktop'    
            
# slice_num = 32
process_signle_scan(single_source_folder, target_folder, side = 'Right', center_img ='image180.jpg')
            
            
