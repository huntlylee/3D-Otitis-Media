# -*- coding: utf-8 -*-
"""
Created on Thu Aug 26 00:24:44 2021

@author: liy45
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from sklearn.metrics import auc, roc_curve, confusion_matrix, cohen_kappa_score
import seaborn as sn


'''Human'''

def get_cmx(y_pred_lbl, y_true_lbl, task = 1):
    
    if task == 1:
        y_true_lbl[y_true_lbl > 0] = 1
        y_pred_lbl[np.nonzero(y_pred_lbl)] = 1
    elif task == 2:
        y_true_lbl[y_true_lbl != 2] = 0
        y_true_lbl[y_true_lbl == 2] = 1
        y_pred_lbl[y_pred_lbl != 2] = 0
        y_pred_lbl[y_pred_lbl == 2] = 1
        
    cm = confusion_matrix(y_true_lbl, y_pred_lbl)
    tn, fp, fn, tp = cm.ravel()
    accuracy = (tp+tn)/len(y_pred_lbl)

    tpr = tp / (tp+fn)
    tnr = tn / (tn+fp)                   
    ppv = tp / (tp+fp)
    f1 = 2*tp/(2*tp + fp + fn)
    # print(cm)
    if task == 1:
        df_cm = pd.DataFrame(cm, index = ['Normal', 'Abnormal'],
                      columns = ['Normal', 'Abnormal'])
    elif task == 2:
        df_cm = pd.DataFrame(cm, index = ['Non-cholesteatoma', 'Cholesteatoma'],
                      columns = ['Non-cholesteatoma', 'Cholesteatoma'])
    
    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(8,4))
    sn.set(font_scale=1.4) # for label size
    sn.heatmap(df_cm, annot=True, annot_kws={"size": 16}, fmt="d", cmap="Blues", ax = ax1) # font size
    ax1.tick_params(axis="x", rotation = 45) 
    # plt.yticks(rotation=90)
    ax1.set_title('Confusion matrix', fontsize = 20) # title with fontsize 20
    ax1.set_xlabel('Predicted labels', fontsize = 15) # x-axis label with fontsize 15
    ax1.set_ylabel('True labels', fontsize = 15) # y-axis label with fontsize 15    
    ax2.annotate('{}\nAccuracy = {:.1%}\nSensitivity = {:.1%}\nSpecificity = {:.1%}\nPrecision = {:.1%}\nF1 = {:.1%}'.format(phy,accuracy, tpr, tnr, ppv,f1), 
                 (0.25, 0.5), xycoords = 'axes fraction', annotation_clip=False, horizontalalignment='left', fontsize = 16)
    ax2.axis('off')
    plt.show()
    return tn, fp, fn, tp, accuracy, tpr, tnr, ppv, f1


human_result = pd.read_excel(r'D:\OneDrive - VUMC\Research\Ongoing Research Projects\My AI projects\AI 3D otitis media\Tasks updated 2022-09-03\Human\review-20230805\Summary.xlsx',
                             sheet_name=1, usecols='A:B,F:O')

acc_list, tpr_list, tnr_list, ppv_list, f1_list = [], [],  [], [], []
task_list, phy_list, ds_list = [], [], []
tn_list, fp_list, fn_list, tp_list =  [], [],  [], []

for task in [1, 2]:
    for ds in [1, 2]:
        for phy in ['JY', 'YC', 'LH', 'QL', 'MM', 'XQ', 'JX']:
            if task == 1:
                df_human = human_result.loc[(human_result['Ground truth']!= 'Remove') & 
                                            ~(human_result[phy].isna()) &
                                            (human_result['Dataset'] == ds)].reset_index(drop=True)
            else:
                df_human = human_result.loc[((human_result['Ground truth'] == 1) | (human_result['Ground truth'] == 2)) & 
                                            ~(human_result[phy].isna()) &
                                            (human_result['Dataset'] == ds)].reset_index(drop=True)
            tn, fp, fn, tp, accuracy, tpr, tnr, ppv, f1 = get_cmx(df_human[phy].to_numpy(copy = True), 
                                              df_human['Ground truth'].astype(float).to_numpy(copy = True), task=task)
            ds_list.append(ds)
            tn_list.append(tn)
            fp_list.append(fp)
            fn_list.append(fn)
            tp_list.append(tp)
            acc_list.append(accuracy)
            tpr_list.append(tpr)
            tnr_list.append(tnr)
            ppv_list.append(ppv)
            f1_list.append(f1)
            task_list.append(task)
            phy_list.append(phy)

df_result = pd.DataFrame(list(zip(task_list, ds_list, phy_list, tn_list, fp_list, fn_list, tp_list, acc_list, tpr_list, tnr_list, ppv_list, f1_list)),
                         columns = ['Task', 'Dataset', 'Physician', 'True neg', 'False pos', 'False neg', 'True pos',
                                    'Accuracy', 'Sensitivity', 'Specificity', 'Precision', 'F1'])

df_result.to_excel(r'C:\Users\liy45\Desktop\human.xlsx', index=False)


# df_result.groupby(['Task', 'Dataset']).mean()
# df_result.groupby(['Task', 'Dataset']).std()

''' consistency'''

def get_kappa(score_1, score_2, task):        
    score_1 = np.asarray(score_1)
    score_2 = np.asarray(score_2)
    if task == 1:
        score_1[np.nonzero(score_1)] = 1
        score_2[np.nonzero(score_2)] = 1
    elif task == 2:
        score_1[score_1 != 2] = 0
        # score_1[score_1 == 2] = 1
        score_2[score_2 != 2] = 0
        # score_2[score_2 == 2] = 1           
    return cohen_kappa_score(score_1, score_2)

task_list, phy_list, ds_list, kappa_list = [], [], [], []

for task in [1, 2]:
    for ds in [1, 2]:
        for phy in ['JY', 'YC', 'LH', 'QL', 'MM', 'XQ', 'JX']:
            if task == 1:
                df_human = human_result.loc[(human_result['Ground truth']!= 'Remove') & 
                                            ~(human_result[phy].isna()) &
                                            # (human_result['Dataset'] == ds) &
                                            ~(human_result['intra-rater reliability'].isna())].reset_index(drop=True)
            else:
                df_human = human_result.loc[((human_result['Ground truth'] == 1) | (human_result['Ground truth'] == 2)) & 
                                            ~(human_result[phy].isna()) &
                                            # (human_result['Dataset'] == ds) &
                                            ~(human_result['intra-rater reliability'].isna())].reset_index(drop=True)
    
            a = df_human.groupby(by = ['MRN', 'Side'],  group_keys=True, sort=True).apply(lambda x: x)
    
            i = 0
            score_1 = []
            score_2 = []
            while i < len(a):
                if a.index[i][0] == a.index[i+1][0] and a.index[i][1] == a.index[i+1][1]:
                    score_1.append(a.iloc[i][phy])
                    score_2.append(a.iloc[i+1][phy])
                    i += 2
                else:
                    i += 1
                        
            kappa_list.append(get_kappa(score_1, score_2, task))
            ds_list.append(ds)
            task_list.append(task)
            phy_list.append(phy)

    
df_kappa = pd.DataFrame(list(zip(task_list, ds_list, phy_list, kappa_list)),
                         columns = ['Task', 'Dataset', 'Physician', 'Kappa'])

df_kappa.to_excel(r'C:\Users\liy45\Desktop\human_kappa.xlsx', index=False)

# df_kappa.groupby(['Task', 'Dataset']).mean()
# df_kappa.groupby(['Task', 'Dataset']).std()



''' ROC '''

df_result = pd.read_excel(r'D:\OneDrive - VUMC\Research\Ongoing Research Projects\My AI projects\AI 3D otitis media\Tasks updated 2022-09-03\Human\physician performance.xlsx', sheet_name=0)
df_kappa = pd.read_excel(r'D:\OneDrive - VUMC\Research\Ongoing Research Projects\My AI projects\AI 3D otitis media\Tasks updated 2022-09-03\Human\physician performance.xlsx', sheet_name=1)



phy_dict = {'JY': ['Senior otologist', '12Y', 'tab:blue'],
            'YC': ['Senior radiologist', '21Y', 'tab:orange'],
            'LH': ['Attending otologist', '7Y', 'tab:green'],
            'QL': ['Resident', '3Y', 'tab:red'],
            'MM': ['Resident', '3Y', 'tab:purple'],
            'XQ': ['Resident', '2Y', 'tab:brown'],
            'JX': ['Senior otologist', '12Y', 'tab:pink']
            }



def plot_cv_roc_curve(df_dict_3d, df_dict_2d, df_result, ax = None, set_name='EENT', save_fig = True):
    tprs_3d, tprs_2d = [], []
    aucs_3d, aucs_2d = [], []
    l_handle, l_text = [], []
    mean_fpr = np.linspace(0, 1, 100)
    plt.style.use('default')
    
    if ax is None:
        fig, ax = plt.subplots(figsize=(5,5))
    
    for test_fold in range(1, 6, 1):
        
        df = df_dict_3d['Fold' + str(test_fold)]
        y_true = df['Truth']
        y_score = df['Pred score']
    
        fpr, tpr, auc_thresholds = roc_curve(y_true, y_score)
        # ax.plot(fpr, tpr, alpha = 0.3)
        interp_tpr = np.interp(mean_fpr, fpr, tpr)
        interp_tpr[0] = 0.0
        tprs_3d.append(interp_tpr)
        aucs_3d.append(auc(fpr, tpr))
        
        df1 = df_dict_2d['Fold' + str(test_fold)]
        y_true1 = df1['Truth']
        y_score1 = df1['Pred score']
        fpr1, tpr1, auc_thresholds1 = roc_curve(y_true1, y_score1)
        interp_tpr1 = np.interp(mean_fpr, fpr1, tpr1)
        interp_tpr1[0] = 0.0
        tprs_2d.append(interp_tpr1)
        aucs_2d.append(auc(fpr1, tpr1))
    
    ax.plot([0, 1], [0, 1], color='darkblue', linestyle='--')
    
    ''' 3d ribbon'''
    mean_tpr_3d = np.mean(tprs_3d, axis=0)
    mean_tpr_3d[-1] = 1.0
    mean_auc_3d = auc(mean_fpr, mean_tpr_3d)
    std_auc_3d = np.std(aucs_3d)
    lmean_3d, = ax.plot(mean_fpr, mean_tpr_3d, color='purple',
            # label='Model-3D (mean AUC={:.2f}$\pm${:.2f})'.format(mean_auc_3d, std_auc_3d),
            lw=2, alpha=.8)
      
    std_tpr_3d = np.std(tprs_3d, axis=0)
    tprs_upper_3d = np.minimum(mean_tpr_3d + std_tpr_3d, 1)
    tprs_lower_3d = np.maximum(mean_tpr_3d - std_tpr_3d, 0)
    lsigma_3d = ax.fill_between(mean_fpr, tprs_lower_3d, tprs_upper_3d, color='purple', alpha=.2,
                    # label='$\pm$ 1 Standard Deviation: 3D'
                    )
    ''' 2d ribbon'''
    mean_tpr_2d = np.mean(tprs_2d, axis=0)
    mean_tpr_2d[-1] = 1.0
    mean_auc_2d = auc(mean_fpr, mean_tpr_2d)
    std_auc_2d = np.std(aucs_2d)
    lmean_2d, = ax.plot(mean_fpr, mean_tpr_2d, color='darkgreen',
            # label='Model-2D (mean AUC={:.2f}$\pm${:.2f})'.format(mean_auc_2d, std_auc_2d),
            lw=2, alpha=.8)
      
    std_tpr_2d = np.std(tprs_2d, axis=0)
    tprs_upper_2d = np.minimum(mean_tpr_2d + std_tpr_2d, 1)
    tprs_lower_2d = np.maximum(mean_tpr_2d - std_tpr_2d, 0)
    lsigma_2d = ax.fill_between(mean_fpr, tprs_lower_2d, tprs_upper_2d, color='lime', alpha=.2,
                    # label='$\pm$ 1 Standard Deviation: 2D'
                    )
    
    l_handle.append((lsigma_3d, lmean_3d))
    l_text.append('Model-3D (mean AUC={:.2f}$\pm${:.2f})'.format(mean_auc_3d, std_auc_3d))
    l_handle.append((lsigma_2d, lmean_2d))
    l_text.append('Model-2D (mean AUC={:.2f}$\pm${:.2f})'.format(mean_auc_2d, std_auc_2d))
    
    
    ds_temp = 2 if ds == 'wh' else 1    
    for phy in ['JY', 'JX', 'YC', 'LH', 'QL', 'MM', 'XQ']:
        tpr_phy = df_result.loc[(df_result['Physician'] == phy) & 
                                (df_result['Dataset'] == ds_temp) &
                                (df_result['Task'] == task)]['Sensitivity']
        fpr_phy = 1 - df_result.loc[(df_result['Physician'] == phy) & 
                                    (df_result['Dataset'] == ds_temp) &
                                    (df_result['Task'] == task)]['Specificity']  
        
        kappa = df_result.loc[(df_result['Physician'] == phy) & 
                                (df_result['Dataset'] == ds_temp) &
                                (df_result['Task'] == task)]['Kappa'].item()
        
        l_phy, = ax.plot(fpr_phy, tpr_phy, marker = '*',  c=phy_dict[phy][2], 
                         # label = '{}, {}, κ={:.2f}'.format(phy_dict[phy][0], phy_dict[phy][1], kappa),
                         markersize = 8, mfc = 'None', ls ='')
        
        l_handle.append(l_phy)
        l_text.append('{}, {}, κ={:.2f}'.format(phy_dict[phy][0], phy_dict[phy][1], kappa))
        
    tpr_phy_mean = df_result.loc[(df_result['Dataset'] == ds_temp) &
                            (df_result['Task'] == task)]['Sensitivity'].mean()
    fpr_phy_mean = 1 - df_result.loc[(df_result['Dataset'] == ds_temp) &
                                (df_result['Task'] == task)]['Specificity'].mean()  
    
    l_phy_mean, = ax.plot(fpr_phy_mean, tpr_phy_mean, marker = 'o',  c='tab:cyan', 
                         label = 'Physician average',
                         markersize = 9, mfc = 'None', ls ='')
    
    l_handle.append(l_phy_mean)
    l_text.append('Physician average')
    
    ax.set(xlim=[-0.05, 1.05], ylim=[-0.05, 1.05])
    # ax.set_xlabel('False Positive Rate')
    # ax.set_ylabel('True Positive Rate')
    ax.set_title('{}'.format(set_name))
    ax.legend(l_handle, l_text, loc="lower right", fontsize = 8)
    if save_fig:
        plt.savefig(r'C:\Users\liy45\Desktop\{} task{}.png'.format(set_name, task), dpi=300)
    # plt.show()
    return ax

fig, axes = plt.subplots(nrows= 2, ncols =2, figsize=(12,10))

# ax_list = []

for task in [1,2]:

    if task == 2:
        parent_dir_3d = r'D:\OneDrive - VUMC\Research\Ongoing Research Projects\My AI projects\AI 3D otitis media\Tasks updated 2022-09-03\Results task 2 updated 9-21-22'
        parent_dir_2d = r'D:\OneDrive - VUMC\Research\Ongoing Research Projects\My AI projects\AI 3D otitis media\2dCNN\Task 2'
    else:
        parent_dir_3d = r'D:\OneDrive - VUMC\Research\Ongoing Research Projects\My AI projects\AI 3D otitis media\Tasks updated 2022-09-03\Results task 1'
        parent_dir_2d = r'D:\OneDrive - VUMC\Research\Ongoing Research Projects\My AI projects\AI 3D otitis media\2dCNN\Task 1'
        
    for ds_num, ds in enumerate(['sh', 'wh']):
        df_dict_3d = {}
        df_dict_2d = {}
        for test_fold in range(1, 6, 1):
            csv_file_path_3d = os.path.join(parent_dir_3d, 'fold {}'.format(test_fold),  'Result_{}_fold{}.xlsx'.format(ds, test_fold)) 
            csv_file_path_2d = os.path.join(parent_dir_2d, 'Result_{}_T{}_2dCNN.xlsx'.format(ds.upper(), task)) 
            df_dict_3d['Fold' + str(test_fold)] = pd.read_excel(csv_file_path_3d) 
            df_dict_2d['Fold' + str(test_fold)] = pd.read_excel(csv_file_path_2d, sheet_name='fold{}'.format(test_fold)) 
  
        plot_cv_roc_curve(df_dict_3d, df_dict_2d, df_result, axes[task-1][ds_num], set_name='', save_fig = False)

fig.supxlabel('False Positive Rate')
fig.supylabel('True Positive Rate')

cols = ['EENT set', 'WU set'] 
rows  = ['Task 1', 'Task 2']


pad = 5 # in points

for ax, col in zip(axes[0], cols):
    ax.annotate(col, xy=(0.5, 1), xytext=(0, pad),
                xycoords='axes fraction', textcoords='offset points',
                size=16, ha='center', va='baseline')

for ax, row in zip(axes[:,0], rows):
    ax.annotate(row, xy=(0, 0.5), xytext=(-ax.yaxis.labelpad - pad, 0),
                xycoords=ax.yaxis.label, textcoords='offset points',
                size=16, ha='right', va='center')


fig.tight_layout()
fig.savefig(r'C:\Users\liy45\Desktop\Figure 3.png', dpi=300)
plt.show()

