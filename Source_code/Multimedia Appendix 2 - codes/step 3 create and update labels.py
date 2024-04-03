# -*- coding: utf-8 -*-
"""
Extract data from an Excel sheet

Created on Fri Jul  9 15:50:07 2021

@author: kakee
"""

# import os
import pandas as pd
import os
# import csv
import numpy as np
# import re
# from datetime import datetime
from sklearn.model_selection import StratifiedKFold, train_test_split



''' read verified label excel file '''

lbl_file_dir = r'D:\OneDrive - VUMC\Research\Ongoing Research Projects\My AI projects\AI 3D otitis media\Updated labels for tast 2'

df_list =[]
for i in range(1,6,1):
    lbl_file_path = os.path.join(lbl_file_dir, 'Result fold{}-已完成.xlsx'.format(i))
    df_temp = pd.read_excel(lbl_file_path, header = 0, usecols = 'A:E, G:I')
    df_temp.columns.values[5] = "Action-Task 1"
    df_temp.columns.values[6] = "Corrected diagnoses"
    df_temp.columns.values[7] = "CT impression"
    df_list.append(df_temp)
df = pd.concat(df_list, ignore_index=True, sort=False)



''' dict label: disease
（0、无 1、慢性化脓性中耳炎 2、中耳胆脂瘤 3、鼓室硬化症 4、中耳胆固醇肉芽肿 5、粘连性中耳炎  6、鼓室成形术后 7、分泌性中耳炎 
8、外耳道胆脂瘤 9、耳硬化症 10、鼓室体瘤 11、面瘫 12、岩尖胆脂瘤 13、听骨链畸形 14、听神经瘤 ）'''



''' 2022-09-03 - read labels after targeted review: Shanghai'''

def gen_new_lbl(row):
    if row['Action-Task 1'] == 1:
        return row['Truth']
    elif row['Action-Task 1'] == 2:
        return 'remove'
    elif row['Action-Task 1'] == 3:
        return abs(row['Truth'] - 1)
    else:
        return row['Truth']

df['Task 1 label 9-3-22'] = df.apply(gen_new_lbl, axis=1)

df.to_excel(r'C:\Users\liy45\Desktop\Labels 2022-09-03 SH CV.xlsx', index=False)
# df_new = df_original.loc[df_original['New label'] != 'remove']




''' read verified label excel file WH 2022-02-22 with newly extracted ROI'''

lbl_file_path_wh = r'D:\OneDrive - VUMC\Research\Ongoing Research Projects\My AI projects\AI 3D otitis media\Task 1 lbl wh 022222 Chen.xlsx'
lbl_file_path_wh_yl = r'D:\OneDrive - VUMC\Research\Ongoing Research Projects\My AI projects\AI 3D otitis media\Task 1 lbl wh 022222.xlsx'

df_wh1 = pd.read_excel(lbl_file_path_wh, header = 0)

df_wh1.columns.values[6] = "Corrected diagnoses"
df_wh1.columns.values[7] = "CT impression"

df_wh2 = pd.read_excel(lbl_file_path_wh_yl, header = 0)


df_wh = df_wh2.merge(df_wh1[['Corrected diagnoses', "CT impression", 'Scan ID', 'Side']], how = 'outer',
                on = ['Scan ID',  'Side'])


def generate_final_lbl_wh(row):
    if not pd.isna(row['Review YL']):
        return row['Review YL']
    else:
        return row['Task 1 label after review']


df_wh['Task 1 label 9-3-22'] = df_wh.apply(generate_final_lbl_wh, axis =1)

df_wh.to_excel(r'C:\Users\liy45\Desktop\Labels 2022-09-03 SH CV.xlsx', index=False)



''' Task 2: generating labels for EENT'''


''' Read updated task 1 labels'''
lbl_file_path = r'D:\OneDrive - VUMC\Research\Ongoing Research Projects\My AI projects\AI 3D otitis media\Tasks updated 2022-09-03\Labels 2022-09-03 SH Task 1.xlsx'

df = pd.read_excel(lbl_file_path, header = 0)

def get_ct_imp(row):
    if pd.isna(row["CT impression"]):
        if row['Task 1 label 9-3-22'] != 'remove':
            if row["Pred score"] >= 0.9:
                return 3
            elif row["Pred score"] <= 0.1:
                return 4
            else: return 'check'
        else:
            return 'remove'
    else:
        return row["CT impression"]

df['CT impression 2'] = df.apply(get_ct_imp, axis = 1)

df[['Scan ID', 'Scan date', 'Side']] = df['ID'].str.split('-',  expand=True)

'''orignal labels'''
lbl_file_path1 = r'D:\OneDrive - VUMC\Research\Ongoing Research Projects\My AI projects\AI 3D otitis media\Tasks updated 2022-09-03\Task 2 labels\Label sheet 20220109 Shanghai reviewed.xlsx'
df1 = pd.read_excel(lbl_file_path1, header = 0)

df1 = pd.melt(df1, id_vars=['Scan ID', 'Scan date'], value_vars=['Right', 'Left'], ignore_index=False, value_name = 'Diagnoses task 2' , var_name = 'Side' )
df1['Scan date'] = df1['Scan date'].astype(str)

df_sh = df.merge(df1, on = ['Scan ID', 'Scan date', 'Side'], how = 'outer')

''' combine orignal labels with checked labels'''
def get_combine_lbl(row):
    if pd.isna(row['Corrected diagnoses']):
        return row['Diagnoses task 2']
    else:
        return str(row['Diagnoses task 2']) + ' + ' + str(row['Corrected diagnoses'])

df_sh['Diagnoses combined'] = df_sh.apply(get_combine_lbl, axis = 1)


''' if cholesteatoma'''
def get_cholesteatoma_lbl(row):
    if '2' in str(row['Diagnoses combined']):
        return 1
    else:
        return 0

df_sh['Cholesteatoma'] = df_sh.apply(get_cholesteatoma_lbl, axis = 1)

df_sh.to_excel(r'C:\Users\liy45\Desktop\Labels 2022-09-03 SH CV.xlsx', index=False)

df_sh_task2 = df_sh.loc[df_sh['CT impression 2'].astype(str).str.contains("3|2")].reset_index(drop = True)


''' assign folds'''
def generate_kfold(df):
    lbl_array = df['Cholesteatoma'].to_list()
    test_dict = {}
    i = 0
    for _, test_index in StratifiedKFold(n_splits=5, random_state=120, shuffle=True).split(np.zeros(len(lbl_array)), lbl_array):
        # train_dict['fold {}'.format(i)] =  train_index
        test_dict['fold {}'.format(i)] =  test_index
        i += 1
    for i in range(len(test_dict)):
        for j in test_dict[list(test_dict)[i]]:
            df.at[j, 'Fold task 2'] = i+1          
    return df

df_sh_task2 = generate_kfold(df_sh_task2) 

df_sh_task2.to_excel(r'C:\Users\liy45\Desktop\Labels 2022-09-04 SH Task 2.xlsx', index=False)



lbl_review_path = r'D:\OneDrive - VUMC\Research\Ongoing Research Projects\My AI projects\AI 3D otitis media\Tasks updated 2022-09-03\Task 2 labels\Shanghai\Labels 2022-09-07 Add missing Chen.xlsx'

df_sh_task2 = pd.read_excel(lbl_review_path, header = 0)
df_sh_task2['CT impression 2'] = df_sh_task2.apply(get_ct_imp, axis = 1)

df_sh_task2['Diagnoses combined'] = df_sh_task2.apply(get_combine_lbl, axis = 1)

df_sh_task2['Cholesteatoma'] = df_sh_task2.apply(get_cholesteatoma_lbl, axis = 1)

df_sh_task2.to_excel(r'C:\Users\liy45\Desktop\Labels 2022-09-08 SH All.xlsx', index=False)

df_sh_task2 = df_sh_task2.loc[df_sh_task2['CT impression 2'].astype(str).str.contains("3|2")].reset_index(drop = True)

df_sh_task2 = generate_kfold(df_sh_task2) 

df_sh_task2.to_excel(r'C:\Users\liy45\Desktop\Labels 2022-09-08 SH Task 2.xlsx', index=False)

print('Cholesteatoma: {}\nOthers: {}'.format(len(df_sh_task2.loc[df_sh_task2['Cholesteatoma'] == 1]), 
                                             len(df_sh_task2.loc[df_sh_task2['Cholesteatoma'] == 0])))




''' 9-20-2022, read and combined reviewed lbls'''

df_sh_task2 = pd.read_excel(r'D:\OneDrive - VUMC\Research\Ongoing Research Projects\My AI projects\AI 3D otitis media\Tasks updated 2022-09-03\Labels 2022-09-08 SH Task 2.xlsx')


def read_review_df():
    df_list = []
    for i in range(1,6):
        df_temp = pd.read_excel(r'D:\OneDrive - VUMC\Research\Ongoing Research Projects\My AI projects\AI 3D otitis media\Tasks updated 2022-09-03\Task 2 labels\Shanghai\Task 2 reviewed\Result_sh_fold{}.xlsx'.format(i))
        df_list.append(df_temp)
    return pd.concat(df_list, ignore_index=True, axis=0)
    
df_sh_task2_review = read_review_df()


df_sh_task2_new = df_sh_task2.merge(df_sh_task2_review[['ID', '复核标签结果']], on = 'ID', how = 'left')
df_sh_task2_new.rename({'复核标签结果': 'Diagnoses task 2 rev Chen'}, axis=1, inplace=True)

def get_combine_lbl_tsk2(row):
    if pd.isna(row['Diagnoses task 2 rev Chen']):
        return row['Diagnoses combined']
    else:
        return str(row['Diagnoses combined']) + ' + ' + str(row['Diagnoses task 2 rev Chen'])


df_sh_task2_new['Diagnoses 9-20-22'] = df_sh_task2_new.apply(get_combine_lbl_tsk2, axis = 1)

def get_cholesteatoma_lbl_tsk2(row):
    if '2' in str(row['Diagnoses combined']):
        return 1
    else:
        return 0

df_sh_task2_new['Cholesteatoma'] = df_sh_task2_new.apply(get_cholesteatoma_lbl_tsk2, axis = 1)


df_sh_task2_new.to_excel(r'C:\Users\liy45\Desktop\Labels 2022-09-20 SH Task 2.xlsx', index=False)

print('Cholesteatoma: {}\nOthers: {}'.format(len(df_sh_task2_new.loc[df_sh_task2['Cholesteatoma'] == 1]), 
                                             len(df_sh_task2_new.loc[df_sh_task2['Cholesteatoma'] == 0])))




''' Task 2: generating labels for WH'''

lbl_file_path_wh = r'D:\OneDrive - VUMC\Research\Ongoing Research Projects\My AI projects\AI 3D otitis media\Tasks updated 2022-09-03\Task 2 labels\Wuhan\WH task 2 lbl review chen 2022-10-06.xlsx'
df_wh = pd.read_excel(lbl_file_path_wh, header = 0)

''' diagnoses'''
def get_combine_lbl_wh(row):
    if pd.isna(row['Corrected diagnoses']):
        return row['Diagnoses combined']
    else:
        return str(row['Corrected diagnoses'])


df_wh['Diagnoses 20221119'] = df_wh.apply(get_combine_lbl_wh, axis = 1)


''' CT impression'''

def get_ct_imp_wh(row):
    if pd.isna(row["CT impression"]):
        return row["CT impression 2"]           
    else:
        return row["CT impression"]

df_wh['CT impression 20221119'] = df_wh.apply(get_ct_imp_wh, axis = 1)




''' if cholesteatoma'''
def get_cholesteatoma_lbl_wh(row):
    if '2' in str(row['Diagnoses 20221119']):
        return 1
    else:
        return 0

df_wh['Cholesteatoma 20221119'] = df_wh.apply(get_cholesteatoma_lbl_wh, axis = 1)

df_wh.to_excel(r'C:\Users\liy45\Desktop\Labels 2022-11-19 WH task 2 All.xlsx', index=False)


''' select cases based on CT impression'''
df_wh_task2 = df_wh.loc[df_wh['CT impression 20221119'].astype(str).str.contains("3|2")].reset_index(drop = True)

df_wh_task2.to_excel(r'C:\Users\liy45\Desktop\Labels 2022-11-19 WH Task 2.xlsx', index=False)

print('Cholesteatoma: {}\nOthers: {}'.format(len(df_wh_task2.loc[df_wh_task2['Cholesteatoma 20221119'] == 1]), 
                                             len(df_wh_task2.loc[df_wh_task2['Cholesteatoma 20221119'] == 0])))



