# -*- coding: utf-8 -*-
"""
Created on Mon Nov 13 12:55:04 2023

@author: liy45
"""

import numpy as np
from mlxtend.evaluate import mcnemar_table
import pandas as pd
from mlxtend.evaluate import mcnemar
from scipy import stats

import pingouin as pg 


import matplotlib.pyplot as plt
from sklearn.metrics import auc, roc_curve, confusion_matrix, cohen_kappa_score
import seaborn as sn

''' Benchmark'''
''' Task 2'''
ds = 'WH'

''' 2d cnn'''
result_dict = {}
for fold_num in range(1,6):
    result_dict['fold{}'.format(fold_num)] = pd.read_excel(r'D:\OneDrive - VUMC\Research\Ongoing Research Projects\My AI projects\AI 3D otitis media\2dCNN\Task 2\Result_{}_T2_2dCNN.xlsx'.format(ds), 
                                                           sheet_name = 'fold{}'.format(fold_num), header = 0)
merged_df = pd.concat(result_dict).reset_index(drop = True)

# y_target_2d = merged_df['Truth'].to_numpy()
# y_model_2d = merged_df['Pred lbl'].to_numpy()

''' current model'''
result_dict1 = {}
for fold_num in range(1,6):
    result_dict1['fold{}'.format(fold_num)] = pd.read_excel(r'D:\OneDrive - VUMC\Research\Ongoing Research Projects\My AI projects\AI 3D otitis media\Tasks updated 2022-09-03\Results task 2 updated 9-21-22\fold {}\Result_{}_fold{}.xlsx'.format(fold_num, ds.lower(), fold_num), 
                                                           header = 0)
merged_df1 = pd.concat(result_dict1).reset_index(drop = True)

# y_target_3d = merged_df1['Truth'].to_numpy()
# y_model_3d = merged_df1['Pred lbl'].to_numpy()
df_final = merged_df.merge(merged_df1, on =['ID','Fold'] , suffixes=('_2d', '_3d'), how = 'outer')

''' chi square test'''
# df_final['diff'] = df_final['Truth_2d'] - df_final['Truth_3d']
# len(df_final.loc[df_final['diff'] != 0])

# y_target = df_final['Truth_2d'].to_numpy()
# y_model_2d = df_final['Pred lbl_2d'].to_numpy()
# y_model_3d = df_final['Pred lbl_3d'].to_numpy()

# tb = mcnemar_table(y_target=y_target,  y_model1=y_model_2d, y_model2=y_model_3d)
# print(tb)

# chi2, p = mcnemar(ary=tb, corrected=True)
# print('chi-squared:', chi2)
# print('p-value:', p)

''' pair t test'''
df_final['2d_outcome'] = np.where((df_final['Truth_2d'] == df_final['Pred lbl_2d']), 1, 0)
df_final['3d_outcome'] = np.where((df_final['Truth_3d'] == df_final['Pred lbl_3d']), 1, 0)

stats.ttest_rel(df_final['2d_outcome'], df_final['3d_outcome'])


''' Task 1'''

ds = 'SH'

''' 2d cnn'''
result_dict = {}
for fold_num in range(1,6):
    result_dict['fold{}'.format(fold_num)] = pd.read_excel(r'D:\OneDrive - VUMC\Research\Ongoing Research Projects\My AI projects\AI 3D otitis media\2dCNN\Task 1\Result_{}_T1_2dCNN.xlsx'.format(ds), 
                                                           sheet_name = 'fold{}'.format(fold_num), header = 0)
merged_df = pd.concat(result_dict).reset_index(drop = True)

# y_target_2d = merged_df['Truth'].to_numpy()
# y_model_2d = merged_df['Pred lbl'].to_numpy()

''' current model'''
result_dict1 = {}
for fold_num in range(1,6):
    result_dict1['fold{}'.format(fold_num)] = pd.read_excel(r'D:\OneDrive - VUMC\Research\Ongoing Research Projects\My AI projects\AI 3D otitis media\Tasks updated 2022-09-03\Results task 1\fold {}\Result_{}_fold{}.xlsx'.format(fold_num, ds, fold_num), 
                                                           header = 0)
merged_df1 = pd.concat(result_dict1).reset_index(drop = True)

# y_target_3d = merged_df1['Truth'].to_numpy()
# y_model_3d = merged_df1['Pred lbl'].to_numpy()

df_final = merged_df.merge(merged_df1, on =['ID','Fold'] , suffixes=('_2d', '_3d'), how = 'outer')

''' chi square test'''
# df_final['diff'] = df_final['Truth_2d'] - df_final['Truth_3d']
# len(df_final.loc[df_final['diff'] != 0])

# y_target = df_final['Truth_2d'].to_numpy()
# y_model_2d = df_final['Pred lbl_2d'].to_numpy()
# y_model_3d = df_final['Pred lbl_3d'].to_numpy()

# tb = mcnemar_table(y_target=y_target,  y_model1=y_model_2d, y_model2=y_model_3d)
# print(tb)

# chi2, p = mcnemar(ary=tb, corrected=True)
# print('chi-squared:', chi2)
# print('p-value:', p)


''' pair t test'''
df_final['2d_outcome'] = np.where((df_final['Truth_2d'] == df_final['Pred lbl_2d']), 1, 0)
df_final['3d_outcome'] = np.where((df_final['Truth_3d'] == df_final['Pred lbl_3d']), 1, 0)

stats.ttest_rel(df_final['2d_outcome'], df_final['3d_outcome'])



''' Ablation Task 1'''

model_num = 3
ds = 'SH'

''' model 1-3'''
result_dict = {}
for fold_num in range(1,6):
    result_dict['fold{}'.format(fold_num)] = pd.read_excel(r'D:\OneDrive - VUMC\Research\Ongoing Research Projects\My AI projects\AI 3D otitis media\Ablation study\Task 1\Result_{}_T1_M{}.xlsx'.format(ds, model_num), 
                                                           sheet_name = 'fold{}'.format(fold_num), header = 0)
merged_df = pd.concat(result_dict).reset_index(drop = True)

''' current model'''
result_dict1 = {}
for fold_num in range(1,6):
    result_dict1['fold{}'.format(fold_num)] = pd.read_excel(r'D:\OneDrive - VUMC\Research\Ongoing Research Projects\My AI projects\AI 3D otitis media\Tasks updated 2022-09-03\Results task 1\fold {}\Result_{}_fold{}.xlsx'.format(fold_num, ds, fold_num), 
                                                           header = 0)

merged_df1 = pd.concat(result_dict1).reset_index(drop = True)

df_final = merged_df.merge(merged_df1, on =['ID','Fold'] , suffixes=('_2d', '_3d'), how = 'outer')

''' chi square test'''
# df_final['diff'] = df_final['Truth_2d'] - df_final['Truth_3d']
# len(df_final.loc[df_final['diff'] != 0])

# y_target = df_final['Truth_2d'].to_numpy()
# y_model_2d = df_final['Pred lbl_2d'].to_numpy()
# y_model_3d = df_final['Pred lbl_3d'].to_numpy()

# tb = mcnemar_table(y_target=y_target,  y_model1=y_model_2d, y_model2=y_model_3d)
# print(tb)

# chi2, p = mcnemar(ary=tb, corrected=True)
# print('chi-squared:', chi2)
# print('p-value:', p)

''' pair t test'''
df_final['2d_outcome'] = np.where((df_final['Truth_2d'] == df_final['Pred lbl_2d']), 1, 0)
df_final['3d_outcome'] = np.where((df_final['Truth_3d'] == df_final['Pred lbl_3d']), 1, 0)

stats.ttest_rel(df_final['2d_outcome'], df_final['3d_outcome'])


''' Ablation Task 2'''

model_num = 3
ds = 'WH'

''' model 1-3'''
result_dict = {}
for fold_num in range(1,6):
    result_dict['fold{}'.format(fold_num)] = pd.read_excel(r'D:\OneDrive - VUMC\Research\Ongoing Research Projects\My AI projects\AI 3D otitis media\Ablation study\Task 2\Result_{}_T2_M{}.xlsx'.format(ds, model_num), 
                                                           sheet_name = 'fold{}'.format(fold_num), header = 0)
merged_df = pd.concat(result_dict).reset_index(drop = True)

''' current model'''
result_dict1 = {}
for fold_num in range(1,6):
    result_dict1['fold{}'.format(fold_num)] = pd.read_excel(r'D:\OneDrive - VUMC\Research\Ongoing Research Projects\My AI projects\AI 3D otitis media\Tasks updated 2022-09-03\Results task 2 updated 9-21-22\fold {}\Result_{}_fold{}.xlsx'.format(fold_num, ds.lower(), fold_num), 
                                                           header = 0)

merged_df1 = pd.concat(result_dict1).reset_index(drop = True)

df_final = merged_df.merge(merged_df1, on =['ID','Fold'] , suffixes=('_2d', '_3d'), how = 'outer')

''' chi square test'''
# df_final['diff'] = df_final['Truth_2d'] - df_final['Truth_3d']
# len(df_final.loc[df_final['diff'] != 0])

# y_target = df_final['Truth_2d'].to_numpy()
# y_model_2d = df_final['Pred lbl_2d'].to_numpy()
# y_model_3d = df_final['Pred lbl_3d'].to_numpy()

# tb = mcnemar_table(y_target=y_target,  y_model1=y_model_2d, y_model2=y_model_3d)
# print(tb)

# chi2, p = mcnemar(ary=tb, corrected=True)
# print('chi-squared:', chi2)
# print('p-value:', p)

''' pair t test'''
df_final['2d_outcome'] = np.where((df_final['Truth_2d'] == df_final['Pred lbl_2d']), 1, 0)
df_final['3d_outcome'] = np.where((df_final['Truth_3d'] == df_final['Pred lbl_3d']), 1, 0)

stats.ttest_rel(df_final['2d_outcome'], df_final['3d_outcome'])


''' Human vs AI '''


''' create human result'''
''' Task 1'''
df_human = pd.read_excel(r'D:\OneDrive - VUMC\Research\Ongoing Research Projects\My AI projects\AI 3D otitis media\Tasks updated 2022-09-03\Human\review-20230805\Summary.xlsx',
                             sheet_name=1, usecols='A:B,F:O')

df_human = df_human.loc[df_human['Ground truth'] != 'Remove'].reset_index(drop=True)

     
df_human.loc[(df_human['Ground truth'] == 0), 'GT1'] = 0
df_human.loc[(df_human['Ground truth'] != 0), 'GT1'] = 1

for phy in ['JY', 'YC', 'LH', 'QL', 'MM', 'XQ', 'JX']:   
    df_human[phy+'1'] = df_human[phy].map(lambda x: 0 if x == 0 else (np.nan if pd.isna(x) else 1 ))
for phy in ['JY', 'YC', 'LH', 'QL', 'MM', 'XQ', 'JX']:      
    df_human.loc[pd.isna(df_human[phy+'1']), phy+'_outcome'] = np.nan
    df_human.loc[df_human[phy+'1'] == df_human['GT1'], phy+'_outcome'] = 1
    df_human.loc[~pd.isna(df_human[phy+'1']) & (df_human[phy+'1'] != df_human['GT1']), phy+'_outcome'] = 0

df_human['case ID'] = df_human['MRN'] + '-' + df_human['Side']

df_human.drop(columns = ['Ground truth', 'JY', 'YC', 'LH', 'QL', 'MM', 'XQ', 'JX']).to_excel(r'D:\OneDrive - VUMC\Research\Ongoing Research Projects\My AI projects\AI 3D otitis media\Tasks updated 2022-09-03\Human\review-20230805\Task 1.xlsx', 
                  index=False)

''' Task 2'''
df_human = pd.read_excel(r'D:\OneDrive - VUMC\Research\Ongoing Research Projects\My AI projects\AI 3D otitis media\Tasks updated 2022-09-03\Human\review-20230805\Summary.xlsx',
                             sheet_name=1, usecols='A:B,F:O')

df_human = df_human.loc[((df_human['Ground truth'] == 1) | (df_human['Ground truth'] == 2))].reset_index(drop=True)

 
df_human.loc[(df_human['Ground truth'] == 2), 'GT2'] = 1
df_human.loc[(df_human['Ground truth'] != 2), 'GT2'] = 0

for phy in ['JY', 'YC', 'LH', 'QL', 'MM', 'XQ', 'JX']:   
    df_human[phy+'2'] = df_human[phy].map(lambda x: 1 if x == 2 else (np.nan if pd.isna(x) else 0 ))
for phy in ['JY', 'YC', 'LH', 'QL', 'MM', 'XQ', 'JX']:      
    df_human.loc[pd.isna(df_human[phy+'2']), phy+'_outcome'] = np.nan
    df_human.loc[df_human[phy+'2'] == df_human['GT2'], phy+'_outcome'] = 1
    df_human.loc[~pd.isna(df_human[phy+'2']) & (df_human[phy+'2'] != df_human['GT2']), phy+'_outcome'] = 0

df_human['case ID'] = df_human['MRN'] + '-' + df_human['Side']

df_human.drop(columns = ['Ground truth', 'JY', 'YC', 'LH', 'QL', 'MM', 'XQ', 'JX']).to_excel(r'D:\OneDrive - VUMC\Research\Ongoing Research Projects\My AI projects\AI 3D otitis media\Tasks updated 2022-09-03\Human\review-20230805\Task 2.xlsx', 
                  index=False)


''' Task 1'''

ds = 'SH'

result_dict1 = {}
for fold_num in range(1,6):
    result_dict1['fold{}'.format(fold_num)] = pd.read_excel(r'D:\OneDrive - VUMC\Research\Ongoing Research Projects\My AI projects\AI 3D otitis media\Tasks updated 2022-09-03\Results task 1\fold {}\Result_{}_fold{}.xlsx'.format(fold_num, ds, fold_num), 
                                                           header = 0)
df_ai = pd.concat(result_dict1).reset_index(drop = True)
df_ai['Outcome'] = np.where((df_ai['Pred lbl'] == df_ai['Truth']), 1, 0)

df_ai['Rater'] = 'Model' + df_ai['Fold'].astype(str)
df_ai['Group'] = 'AI'
df_ai = df_ai.rename({'ID': 'case ID'}, axis=1)


df_human = pd.read_excel(r'D:\OneDrive - VUMC\Research\Ongoing Research Projects\My AI projects\AI 3D otitis media\Tasks updated 2022-09-03\Human\review-20230805\Task 1.xlsx')

if ds == 'SH':
    df_human = df_human.loc[df_human['Dataset'] == 1]
else:
    df_human = df_human.loc[df_human['Dataset'] == 2]

df_physician = pd.melt(df_human, id_vars=['case ID'], value_vars=['JY_outcome', 'YC_outcome', 'LH_outcome', 'QL_outcome', 'MM_outcome', 'XQ_outcome', 'JX_outcome'],
                   var_name='Rater', value_name='Outcome')

df_physician=df_physician.dropna(subset = ['Outcome']).reset_index(drop = True)
df_physician['Group'] = 'Human'

df_final = pd.concat([df_physician, df_ai[['case ID', 'Rater', 'Outcome', 'Group']]], ignore_index=True)
df_final = df_final.rename({'case ID': 'ID'}, axis=1)

df_final['case'] = df_final['ID'].astype('category').cat.codes
df_final['case'] = df_final['case'].astype(int)

df_final.to_excel(r'C:\Users\liy45\Desktop\Task 1 {}.xlsx'.format(ds), index=False)


# import statsmodels.api as sm
# import statsmodels.formula.api as smf
# md = smf.mixedlm("Outcome ~ Group", df_final, groups=df_final["Group"])
# mdf = md.fit(method=["lbfgs"])

# print(mdf.summary())
''' Task 2'''

ds = 'WH'

result_dict1 = {}
for fold_num in range(1,6):
    result_dict1['fold{}'.format(fold_num)] = pd.read_excel(r'D:\OneDrive - VUMC\Research\Ongoing Research Projects\My AI projects\AI 3D otitis media\Tasks updated 2022-09-03\Results task 2 updated 9-21-22\fold {}\Result_{}_fold{}.xlsx'.format(fold_num, ds, fold_num), 
                                                           header = 0)
df_ai = pd.concat(result_dict1).reset_index(drop = True)
df_ai['Outcome'] = np.where((df_ai['Pred lbl'] == df_ai['Truth']), 1, 0)

df_ai['Rater'] = 'Model' + df_ai['Fold'].astype(str)
df_ai['Group'] = 'AI'
df_ai = df_ai.rename({'ID': 'case ID'}, axis=1)


df_human = pd.read_excel(r'D:\OneDrive - VUMC\Research\Ongoing Research Projects\My AI projects\AI 3D otitis media\Tasks updated 2022-09-03\Human\review-20230805\Task 2.xlsx')

if ds == 'SH':
    df_human = df_human.loc[df_human['Dataset'] == 1]
else:
    df_human = df_human.loc[df_human['Dataset'] == 2]

df_physician = pd.melt(df_human, id_vars=['case ID'], value_vars=['JY_outcome', 'YC_outcome', 'LH_outcome', 'QL_outcome', 'MM_outcome', 'XQ_outcome', 'JX_outcome'],
                   var_name='Rater', value_name='Outcome')

df_physician=df_physician.dropna(subset = ['Outcome']).reset_index(drop = True)
df_physician['Group'] = 'Human'

df_final = pd.concat([df_physician, df_ai[['case ID', 'Rater', 'Outcome', 'Group']]], ignore_index=True)
df_final = df_final.rename({'case ID': 'ID'}, axis=1)

df_final['case'] = df_final['ID'].astype('category').cat.codes
df_final['case'] = df_final['case'].astype(int)

df_final.to_excel(r'C:\Users\liy45\Desktop\Task 2 {}.xlsx'.format(ds), index=False)



'''read from file'''
df_final = pd.read_excel(r'C:\Users\liy45\Desktop\Task 1 SH.xlsx')
aov = pg.anova(dv='Outcome', between='Group', data=df_final)
aov.round(3)

stats.ttest_ind(df_final.loc[df_final['Group'] == 'Human']['Outcome'], df_final.loc[df_final['Group'] == 'AI']['Outcome'])


''' chi square'''

# tb = [[len(df_final.loc[(df_final['Group'] == 'Human') & (df_final['Outcome'] == 1)]), len(df_final.loc[(df_final['Group'] == 'AI') & (df_final['Outcome'] == 1)])],
#       [len(df_final.loc[(df_final['Group'] == 'Human') & (df_final['Outcome'] == 0)]), len(df_final.loc[(df_final['Group'] == 'AI') & (df_final['Outcome'] == 0)])]
#       ]
# print(np.array(tb))

# chi2, p = mcnemar(ary=np.array(tb), corrected=True)
# print('chi-squared:', chi2)
# print('p-value:', p)


''' multi group'''
df_final = pd.read_excel(r'D:\OneDrive - VUMC\Research\Ongoing Research Projects\My AI projects\AI 3D otitis media\Tasks updated 2022-09-03\Human\stats\Task 2 SH.xlsx')
df_final['Rater2'] = df_final['Rater']
df_final.loc[df_final['Group'] == 'AI', 'Rater2'] = 'AI'
aov = pg.anova(dv='Outcome', between='Rater2', data=df_final)
aov.round(3)
# from statsmodels.stats.multicomp import pairwise_tukeyhsd
# print (pairwise_tukeyhsd(df_final['Outcome'], df_final['Rater2']))
a = pg.pairwise_tests(dv='Outcome', between='Rater2', data=df_final).round(3)
print(a)

''' combined all test cases as one set'''

'''Task 1 or 2'''
df0 = pd.read_excel(r'D:\OneDrive - VUMC\Research\Ongoing Research Projects\My AI projects\AI 3D otitis media\Tasks updated 2022-09-03\Human\stats\Task 1 SH.xlsx')
df1 = pd.read_excel(r'D:\OneDrive - VUMC\Research\Ongoing Research Projects\My AI projects\AI 3D otitis media\Tasks updated 2022-09-03\Human\stats\Task 1 WH.xlsx')

df_final = pd.concat([df0, df1], ignore_index=True)
aov = pg.anova(dv='Outcome', between='Group', data=df_final)
aov.round(3)
stats.ttest_ind(df_final.loc[df_final['Group'] == 'Human']['Outcome'], df_final.loc[df_final['Group'] == 'AI']['Outcome'])

df_final['Rater2'] = df_final['Rater']
df_final.loc[df_final['Group'] == 'AI', 'Rater2'] = 'AI'

aov = pg.anova(dv='Outcome', between='Rater2', data=df_final)
aov.round(3)
a = pg.pairwise_tests(dv='Outcome', between='Rater2', data=df_final).round(3)
print(a)




''' get cmx for ablation study'''

def get_cmx(y_pred_lbl, y_true_lbl, task = 1):
            
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
    ax2.annotate('Model {}\nAccuracy = {:.1%}\nSensitivity = {:.1%}\nSpecificity = {:.1%}\nPrecision = {:.1%}\nF1 = {:.1%}\nTN: {}\nFP: {}\nFN: {}\nTP: {}'.format(model_num,accuracy, tpr, tnr, ppv,f1, tn,fp,fn,tp), 
                 (0.25, 0.3), xycoords = 'axes fraction', annotation_clip=False, horizontalalignment='left', fontsize = 16)
    ax2.axis('off')
    plt.show()
    return tn, fp, fn, tp, accuracy, tpr, tnr, ppv, f1

''' Task 1'''

model_num = 0
ds = 'SH'

''' model 1-3'''
result_dict = {}
for fold_num in range(1,6):
    result_dict['fold{}'.format(fold_num)] = pd.read_excel(r'D:\OneDrive - VUMC\Research\Ongoing Research Projects\My AI projects\AI 3D otitis media\Ablation study\Task 1\Result_{}_T1_M{}.xlsx'.format(ds, model_num), 
                                                           sheet_name = 'fold{}'.format(fold_num), header = 0)
merged_df = pd.concat(result_dict).reset_index(drop = True)

y_pred_lbl = merged_df['Pred lbl'].to_numpy()
y_test = merged_df['Truth'].to_numpy()

get_cmx(y_pred_lbl, y_test)


''' current model'''
result_dict1 = {}
for fold_num in range(1,6):
    result_dict1['fold{}'.format(fold_num)] = pd.read_excel(r'D:\OneDrive - VUMC\Research\Ongoing Research Projects\My AI projects\AI 3D otitis media\Tasks updated 2022-09-03\Results task 1\fold {}\Result_{}_fold{}.xlsx'.format(fold_num, ds, fold_num), 
                                                           header = 0)

merged_df1 = pd.concat(result_dict1).reset_index(drop = True)
y_pred_lbl = merged_df1['Pred lbl'].to_numpy()
y_test = merged_df1['Truth'].to_numpy()

get_cmx(y_pred_lbl, y_test)


''' Ablation Task 2'''

model_num = 0
ds = 'SH'

''' model 1-3'''
result_dict = {}
for fold_num in range(1,6):
    result_dict['fold{}'.format(fold_num)] = pd.read_excel(r'D:\OneDrive - VUMC\Research\Ongoing Research Projects\My AI projects\AI 3D otitis media\Ablation study\Task 2\Result_{}_T2_M{}.xlsx'.format(ds, model_num), 
                                                           sheet_name = 'fold{}'.format(fold_num), header = 0)
merged_df = pd.concat(result_dict).reset_index(drop = True)

y_pred_lbl = merged_df['Pred lbl'].to_numpy()
y_test = merged_df['Truth'].to_numpy()

get_cmx(y_pred_lbl, y_test, task=2)

''' current model'''
result_dict1 = {}
for fold_num in range(1,6):
    result_dict1['fold{}'.format(fold_num)] = pd.read_excel(r'D:\OneDrive - VUMC\Research\Ongoing Research Projects\My AI projects\AI 3D otitis media\Tasks updated 2022-09-03\Results task 2 updated 9-21-22\fold {}\Result_{}_fold{}.xlsx'.format(fold_num, ds.lower(), fold_num), 
                                                           header = 0)

merged_df1 = pd.concat(result_dict1).reset_index(drop = True)
y_pred_lbl = merged_df1['Pred lbl'].to_numpy()
y_test = merged_df1['Truth'].to_numpy()

get_cmx(y_pred_lbl, y_test, task=2)