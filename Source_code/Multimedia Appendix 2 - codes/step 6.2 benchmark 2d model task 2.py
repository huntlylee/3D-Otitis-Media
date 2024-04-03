# -*- coding: utf-8 -*-
"""
Created on Fri Jun 18 17:12:45 2021

@author: liy45
"""

''' 3D classification task from temporal bone CT scans'''

import os
# import zipfile
import numpy as np
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers
# from PIL import Image
import cv2
import random
import matplotlib.pyplot as plt

from scipy import ndimage
import re
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import roc_curve, auc, confusion_matrix
import seaborn as sn

pt_folder_sh = r'D:\My Data\EENT RDD\Otitis Media\Extracted ROI\Shanghai'
pt_folder_wh = r'D:\My Data\EENT RDD\Otitis Media\Extracted ROI\Wuhan 220221'

lbl_file_path_sh = r'D:\OneDrive - VUMC\Research\Ongoing Research Projects\My AI projects\AI 3D otitis media\Tasks updated 2022-09-03\Labels 2022-09-20 SH Task 2.xlsx'
lbl_file_path_wh = r'D:\OneDrive - VUMC\Research\Ongoing Research Projects\My AI projects\AI 3D otitis media\Tasks updated 2022-09-03\Task 2 labels\Wuhan\Labels 2022-11-19 WH Task 2.xlsx'


def read_lbl(lbl_file_path):
    df = pd.read_excel(lbl_file_path, header = 0, usecols = 'A:C, E') 
    df.rename(columns={'Task 1 final': 'Label'}, inplace=True)    
    return df


def generate_kfold(df):
    lbl_array = df['Label'].to_list()
    test_dict = {}
    i = 0
    for _, test_index in StratifiedKFold(n_splits=5, random_state=120, shuffle=True).split(np.zeros(len(lbl_array)), lbl_array):
        # train_dict['fold {}'.format(i)] =  train_index
        test_dict['fold {}'.format(i)] =  test_index
        i += 1
    for i in range(len(test_dict)):
        for j in test_dict[list(test_dict)[i]]:
            df.at[j, 'Test fold'] = i+1          
    return df
    

def read_ROI_imgs(folder):
    # images = []
    images_info = os.listdir(folder)
    images_info.sort(key=lambda f: int(re.sub('\D', '', f)))
    filename = images_info[int(len(images_info)/2)]
    img = cv2.imread(os.path.join(folder,filename), 0)
    if img is not None:
        return img



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


def process_CT_scan_cv(df_lbl, pt_folder, resize_all = False):
    """Read and resize volume"""
    # Read scan    
    images = []
    problem_list = []
    # resize_list = []     
    for i in range(len(df_lbl)):
        
        folder_temp = str(df_lbl.iloc[i]['Scan ID']) + '-' + str(df_lbl.iloc[i]['Scan date'])
        if not os.path.isdir(os.path.join(pt_folder, folder_temp)):
            folder_temp = str(df_lbl.iloc[i]['Scan ID']) + '-20' + str(df_lbl.iloc[i]['Scan date'])        
        path_temp = os.path.join(pt_folder, folder_temp, df_lbl.iloc[i]['Side'])
   
        print('Processing {}...'.format(folder_temp))
        try:
            img_temp = read_ROI_imgs(path_temp)          
            
            # if not img_temp.shape == (128, 128):
            #     img_temp = resize_img(img_temp)
                # resize_list.append(path_temp)
            images.append(img_temp/255)
        except:
            problem_list.append(path_temp)        
    # resize img 
    if resize_all:
        for i in range(len(images)):
            images[i] = resize_img(images[i], 299, 299)           
    images_np = np.stack(images, axis = 0)
    images_np = images_np.astype("float32")   
              
    return images_np, problem_list
# b, problem_list = process_CT_scan_cv(a, pt_folder_sh) 


def plot_slices(data, case_num = 4):
    """Plot a montage of CT slices"""    
    case_list = random.sample(range(data.shape[0]), case_num)  
    rows, columns = int(case_num/2), 2

    f, axarr = plt.subplots(
        rows,
        columns,
        figsize=(2*columns, 2*rows),
    )
    for i in range(rows):
        for j in range(columns):
            case = case_list[i*2+j]
            axarr[i, j].imshow(data[case, :, :], cmap="gray")
            axarr[i, j].axis("off")
    plt.subplots_adjust(wspace=0, hspace=0, left=0, right=1, bottom=0, top=1)
    plt.tight_layout(pad = 0.5)
    plt.show()
    

 
'''Build train and validation datasets
Read the scans from the class directories and assign labels. 
Downsample the scans to have shape of 128x128x64. 
Rescale the raw HU values to the range 0 to 1. 
Lastly, split the dataset into train and validation subsets.'''

# Split data  for training, validation and test.

def train_val_test_split(df_lbl, fold = 1, val_ratio = 0.15):
    
    # problem_dict = {}   
    df_train = df_lbl.loc[df_lbl['Fold task 2'] != fold]
    df_test = df_lbl.loc[df_lbl['Fold task 2'] == fold]
    
    img_train, _ = process_CT_scan_cv(df_train, pt_folder_sh, resize_all=True)
    # problem_dict['train'] = problem_train
    
    img_test, _ = process_CT_scan_cv(df_test, pt_folder_sh, resize_all=True)
    # problem_dict['test'] = problem_test
       
    lbl_train = df_train['Cholesteatoma'].to_numpy()
    lbl_test = df_test['Cholesteatoma'].to_numpy() 
    
    id_train = ['-'.join(i) for i in zip(df_train['Scan ID'], df_train['Scan date'].map(str), df_train['Side'])]
    id_test = ['-'.join(i) for i in zip(df_test['Scan ID'], df_test['Scan date'].map(str), df_test['Side'])]
    
    x_train, x_val, y_train, y_val, pt_train, pt_val = train_test_split(img_train, lbl_train, id_train,
                                                                        test_size=val_ratio, random_state = 123456, shuffle = True, stratify = lbl_train)
       
    return x_train, x_val, img_test, y_train, y_val, lbl_test, pt_train, pt_val, id_test




'''Data augmentation
The CT scans also augmented by rotating at random angles during training. 
Since the data is stored in rank-3 tensors of shape (samples, height, width, depth), 
we add a dimension of size 1 at axis 4 to be able to perform 3D convolutions on the data. 
The new shape is thus (samples, height, width, depth, 1). 
There are different kinds of preprocessing and augmentation techniques out there, 
this example shows a few simple ones to get started.'''


''' augmentation and plot'''


@tf.function
def rotate(volume):
    """Rotate the volume by a few degrees"""

    def scipy_rotate(volume):
        # define some rotation angles
        angles = [-20, -10, -5, 5, 10, 20]
        # pick angles at random
        angle = random.choice(angles)
        # rotate volume
        volume = ndimage.rotate(volume, angle, reshape=False)
        volume[volume < 0] = 0
        volume[volume > 1] = 1
        return volume

    augmented_volume = tf.numpy_function(scipy_rotate, [volume], tf.float32)
    return augmented_volume


def train_preprocessing(volume, label):
    """Process training data by rotating and adding a channel."""
    # Rotate volume
    volume = rotate(volume)   
    volume = tf.expand_dims(volume, axis=-1) 
    converted = tf.image.grayscale_to_rgb(volume)
       
    # image = tf.image.convert_image_dtype(image, tf.float32)
    
    return converted, label


def validation_preprocessing(volume, label):
    """Process validation data by only adding a channel."""
    volume = tf.expand_dims(volume, axis=-1) 
    volume = tf.image.grayscale_to_rgb(volume)
       
    return volume, label

''' examine preprocessed data'''

def plot_tf_data(dataset, num_case = 2):
    data = dataset.take(num_case)
    images, labels = list(data)[0]
    images = images.numpy()
    # image = images[0]
    print("Dimension of the examined CT scan is:", images.shape)
    # plt.imshow(np.squeeze(image[:, :, 5]), cmap="gray")

    plot_slices(images, case_num = num_case)

'''Define a 3D convolutional neural network'''



def get_model(input_shape=(299,299,3), num_classes=2):
    """Build a 3D convolutional neural network model."""

    tf.keras.backend.clear_session()

    input_tensor = keras.layers.Input(shape = input_shape)
    base_model = keras.applications.inception_v3.InceptionV3(input_tensor=input_tensor, weights='imagenet', include_top=False)
    
    # add a global spatial average pooling layer
    x = base_model.output
    x = keras.layers.GlobalAveragePooling2D()(x)
    # let's add a fully-connected layer
    x = keras.layers.Dense(1024, activation='relu')(x)
    # and a logistic layer -- let's say we have 200 classes
    predictions = keras.layers.Dense(num_classes, activation='softmax')(x)
    
    # this is the model we will train
    model = keras.models.Model(inputs=base_model.input, outputs=predictions)
    
    for layer in base_model.layers:
        layer.trainable = True

    return model



def get_roc_curve(y_score, y_true, ds = 'SH'):
    
    # y_score = 1 - y_score[:, 0]  
    plt.style.use('default')
    y_score = y_score[:,1]
    
    y_true = y_true.argmax(axis = 1)
    # y_true[np.nonzero(y_true)] = 1

    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)
    optimal_idx = np.argmax(tpr - fpr)
    optimal_threshold = thresholds[optimal_idx]
    fig, ax = plt.subplots(1, 1, figsize=(4,4))
    lw = 2
    plt.plot(fpr, tpr, color='darkorange',
             lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Task 1: normal vs pathologic {} fold {}'.format(ds, fold_num))
    plt.legend(loc="best")
    plt.tight_layout(pad = 0.2)
    plt.show()
    print('AUROC = {:.3f}'.format(roc_auc))
    return optimal_threshold




def get_cmx(y_pred_lbl, y_true, output_val = False):
    
    y_true = y_true.argmax(axis = 1)
    y_true[np.nonzero(y_true)] = 1
        
    cm = confusion_matrix(y_true, y_pred_lbl)
    tn, fp, fn, tp = cm.ravel()
    accuracy = (tp+tn)/len(y_pred_lbl)

    tpr = tp / (tp+fn)
    tnr = tn / (tn+fp)
                   
    ppv = tp / (tp+fp)
    
    f1 = 2*tp/(2*tp + fp + fn)
    # print(cm)
    df_cm = pd.DataFrame(cm, index = ['Normal', 'Abnormal'],
                  columns = ['Normal', 'Abnormal'])
    
    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(8,4))

    # plt.figure(figsize=(8,4))

    sn.set(font_scale=1.4) # for label size
    sn.heatmap(df_cm, annot=True, annot_kws={"size": 16}, fmt="d", cmap="Blues", ax = ax1) # font size
    ax1.tick_params(axis="x", rotation = 45) 
    # plt.yticks(rotation=90)

    ax1.set_title('Confusion matrix', fontsize = 20) # title with fontsize 20
    ax1.set_xlabel('Predicted labels', fontsize = 15) # x-axis label with fontsize 15
    ax1.set_ylabel('True labels', fontsize = 15) # y-axis label with fontsize 15
    
    ax2.annotate('Accuracy = {:.1%}\nSensitivity = {:.1%}\nSpecificity = {:.1%}\nPrecision = {:.1%}\nF1 = {:.1%}'.format(accuracy, tpr, tnr, ppv, f1), 
                 (0.25, 0.5), xycoords = 'axes fraction', annotation_clip=False, horizontalalignment='left', fontsize = 16)


    ax2.axis('off')
    print('Accuracy = {:.3%}\nSensitivity = {:.3%}\nSpecificity = {:.3%}\nPrecision = {:.3%}\nF1 = {:.3%}'.format(accuracy, tpr, tnr, ppv, f1))

    plt.show()
    if output_val:
        return accuracy, tpr, tnr, ppv, f1






def get_cmx_multiclass(y_test, prediction):
    
    y_true = y_test.argmax(axis = 1)
    y_pred_lbl = prediction.argmax(axis = 1)
    cm = confusion_matrix(y_true, y_pred_lbl)
    
    # print(cm)
    df_cm = pd.DataFrame(cm, index = ['Normal', 'CSOM', 'Cholesteatoma'],
                  columns = ['Normal', 'CSOM', 'Cholesteatoma'])
    
    # fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(8,4))

    # fig = plt.figure(figsize=(8,4))

    sn.set(font_scale=1.4) # for label size
    sn.heatmap(df_cm, annot=True, annot_kws={"size": 16}, fmt="d", cmap="Blues") # font size
    plt.tick_params(axis="x", rotation = 45) 
    # plt.yticks(rotation=90)

    plt.title('Test set multi-class (n={})'.format(len(y_test)), fontsize = 20) # title with fontsize 20
    plt.xlabel('Predicted labels', fontsize = 15) # x-axis label with fontsize 15
    plt.ylabel('True labels', fontsize = 15) # y-axis label with fontsize 15
    

    plt.show()




'''plot test img'''

def plot_test_slices(x_test, df_result, case_selection = None):
    """Plot a montage of CT slices"""    
    if case_selection is None:
        case_selection = random.randrange(len(df_result))
    rows, columns = 4, 8
    y_predict = df_result.iloc[case_selection]['Pred lbl']
    y_true = df_result.iloc[case_selection]['Truth']
    
    case_id = df_result.iloc[case_selection]['ID']

    f, axarr = plt.subplots(
        rows,
        columns,
        figsize=(2*columns, 2*rows),
    )
    for i in range(rows):
        
        for j in range(columns):
            if i*6+j < 32:
                axarr[i, j].imshow(x_test[case_selection, :, :, i*6+j], cmap="gray")
                axarr[i, j].text(0.5, 0.1, str(i*columns + j + 1), transform=axarr[i, j].transAxes, size=14, weight='bold', color = 'w')
            axarr[i, j].axis("off")
    plt.subplots_adjust(wspace=0, hspace=0, left=0, right=1, bottom=0, top=1)
    pred_txt = 'Non-cholesteatoma' if  y_predict == 0 else 'Cholesteatoma'
    true_txt = 'Non-cholesteatoma' if  y_true == 0 else 'Cholesteatoma' 
    txt_color ='red' if pred_txt != true_txt else 'green'

    f.suptitle('Prediction: {}, Truth: {} ({})'.format(pred_txt, true_txt, case_id), fontsize = 28, color = txt_color)
    # plt.annotate('({})'.format(case_id), xy=(0.5, 0.1), xycoords='figure fraction', fontsize = 20, color = 'k')
    f.tight_layout(pad = 0.3)
    f.subplots_adjust(top=0.92)
    plt.show()
    



''' start here if it has not been generated'''

'''Shanghai test set cv execution'''

df_lbl_sh = read_lbl(lbl_file_path_sh)
df_lbl_sh = generate_kfold(df_lbl_sh) 
df_lbl_sh.to_excel(r'D:\OneDrive - VUMC\Research\Ongoing Research Projects\My AI projects\AI 3D otitis media\Label sheet 20220112 Task 1 Shanghai CV.xlsx',
                   index = False)   




''' read from here'''
df_lbl_sh = pd.read_excel(lbl_file_path_sh, header = 0)



''' data gen'''

fold_num = 5
x_train, x_val, x_test, y_train, y_val, y_test, id_train, id_val, id_test  = train_val_test_split(df_lbl_sh, fold = fold_num, val_ratio = 0.15)

y_train = keras.utils.to_categorical(y_train, 2) 
y_val = keras.utils.to_categorical(y_val, 2) 
y_test= keras.utils.to_categorical(y_test, 2) 

print(
    "Number of samples in train, validation and test are %d, %d and %d."
    % (x_train.shape[0], x_val.shape[0], x_test.shape[0]))

# Number of samples in train, validation and test are 1433, 253 and 422.

# plot_slices(x_test, case_num = 4)

train_loader = tf.data.Dataset.from_tensor_slices((x_train, y_train))
validation_loader = tf.data.Dataset.from_tensor_slices((x_val, y_val))

batch_size = 32
# Augment the on the fly during training.
train_dataset = (
    train_loader.shuffle(len(x_train))
    .map(train_preprocessing)
    .batch(batch_size)
    .prefetch(4)
)
# Only rescale.
validation_dataset = (
    validation_loader.shuffle(len(x_val))
    .map(validation_preprocessing)
    .batch(batch_size)
    .prefetch(4)
)

# plot_tf_data(train_dataset, num_case = 4)
# plot_tf_data(validation_dataset, num_case = 4)


''' Build model'''

model = get_model(input_shape = (299,299,3))
model.summary()

'''Train model'''
# Compile model.
initial_learning_rate = 0.0001
lr_schedule = keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate, decay_steps=100000, decay_rate=0.96, staircase=True
)
model.compile(
    loss="categorical_crossentropy",
    optimizer=keras.optimizers.Adam(learning_rate=lr_schedule),
    metrics=["acc"],
)

# Define callbacks.
chckpnt_path = r"C:\Users\liy45\Desktop\Task2_model_2d_fold{}.h5".format(fold_num)
checkpoint_cb = keras.callbacks.ModelCheckpoint(chckpnt_path, save_best_only=True)
early_stopping_cb = keras.callbacks.EarlyStopping(monitor="val_loss", patience=10) #val_loss

# Train the model, doing validation at the end of each epoch
epochs = 150
model.fit(train_dataset,
          validation_data=validation_dataset,
          epochs=epochs,
          shuffle=True,
          verbose=2,
          callbacks=[checkpoint_cb, early_stopping_cb],
          )

# '''load the model'''
# model = keras.models.load_model(chckpnt_path)
'''Visualizing model performance'''

fig, ax = plt.subplots(1, 2, figsize=(20, 3))
ax = ax.ravel()

for i, metric in enumerate(["acc", "loss"]):
    ax[i].plot(model.history.history[metric])
    ax[i].plot(model.history.history["val_" + metric])
    ax[i].set_title("Model {}".format(metric))
    ax[i].set_xlabel("epochs")
    ax[i].set_ylabel(metric)
    ax[i].legend(["train", "val"])
plt.show()

''' repeat training on all other models for the same fold to save time'''




''' Make predictions on the test set'''


model = get_model(input_shape = (299,299,3))
# model.summary()
    
chckpnt_path = r"C:\Users\liy45\Desktop\Task2_model_2d_fold{}.h5".format(fold_num)
model.load_weights(chckpnt_path)

del x_train, train_dataset, train_loader

x_test_newdim, _ = validation_preprocessing(x_test, None)
x_val_newdim, _ = validation_preprocessing(x_val, None)

prediction = model.predict(x_test_newdim, batch_size = 32)
prediction_val = model.predict(x_val_newdim, batch_size = 32)

optimal_threshold = get_roc_curve(prediction_val, y_val)
print(optimal_threshold)


get_roc_curve(prediction, y_test)


y_pred_lbl = [1 if y >= optimal_threshold else 0 for y in prediction[:, 1]]

df_result = pd.DataFrame(list(zip(id_test, prediction[:, 1], y_pred_lbl, np.argmax(y_test, axis = 1))),
                         columns = ['ID', 'Pred score', 'Pred lbl', 'Truth'])

df_result['Fold'] = fold_num

try:
    with pd.ExcelWriter(r'C:\Users\liy45\Desktop\Result_SH_T2_2dCNN.xlsx',  mode='a', if_sheet_exists = 'replace') as writer:  
        df_result.to_excel(writer, sheet_name = 'fold{}'.format(fold_num), index=False)
except:
    df_result.to_excel(r'C:\Users\liy45\Desktop\Result_SH_T2_2dCNN.xlsx', sheet_name = 'fold{}'.format(fold_num), index=False)

get_cmx(y_pred_lbl, y_test)

    
# get_cmx_multiclass(y_test, prediction)
  
plot_test_slices(x_test, df_result, case_selection = None)






''' Wuhan test set cv'''

df_lbl_wh = pd.read_excel(lbl_file_path_wh, header = 0) 
                         

def scandate_tostring(row):
    return row.replace('-','')


'''generate img arrays'''
images_np_wh, problem_list_wh = process_CT_scan_cv(df_lbl_wh, pt_folder_wh, resize_all = True)

''' labels'''
lbl_array_wh = np.array(df_lbl_wh['Cholesteatoma 20221119'])
y_test_wh = keras.utils.to_categorical(lbl_array_wh, 2) 

images_np_wh_tf, _ = validation_preprocessing(images_np_wh, None)




fold_num = 5

chckpnt_path = r"D:\OneDrive - VUMC\Research\Ongoing Research Projects\My AI projects\AI 3D otitis media\2dCNN\Task 2\Task2_model_2d_fold{}.h5".format(fold_num)

model = get_model(input_shape = (299,299,3), num_classes=2)
model.load_weights(chckpnt_path)

prediction_wh = model.predict(images_np_wh_tf, batch_size = 32)
get_roc_curve(prediction_wh, y_test_wh, 'WH')

df_sh = pd.read_excel(r'D:\OneDrive - VUMC\Research\Ongoing Research Projects\My AI projects\AI 3D otitis media\2dCNN\Task 2\Result_SH_T2_2dCNN.xlsx',
                      sheet_name = 'fold{}'.format(fold_num))

optimal_threshold = (df_sh.loc[df_sh['Pred lbl'] == 0]['Pred score'].max() + df_sh.loc[df_sh['Pred lbl'] == 1]['Pred score'].min())/2


# plot_slices(images_np_wh, case_num = 4)

y_pred_lbl_wh = [1 if y >= optimal_threshold else 0 for y in prediction_wh[:, 1]]
get_cmx(y_pred_lbl_wh, y_test_wh)

''' record results'''
id_wh = ['-'.join(i) for i in zip(df_lbl_wh['Scan ID'], df_lbl_wh['Scan date'].map(str),  df_lbl_wh['Side'])]

df_result_wh = pd.DataFrame(list(zip(id_wh, prediction_wh[:, 1], y_pred_lbl_wh, np.argmax(y_test_wh, axis = 1))),
                         columns = ['ID', 'Pred score', 'Pred lbl', 'Truth'])

df_result_wh['Fold'] = fold_num

try:
    with pd.ExcelWriter(r'C:\Users\liy45\Desktop\Result_WH_T2_2dCNN.xlsx',  mode='a', if_sheet_exists = 'replace') as writer:  
        df_result_wh.to_excel(writer, sheet_name = 'fold{}'.format(fold_num), index=False)
except:
    df_result_wh.to_excel(r'C:\Users\liy45\Desktop\Result_WH_T2_2dCNN.xlsx', sheet_name = 'fold{}'.format(fold_num), index=False)


plot_test_slices(images_np_wh, df_result_wh, case_selection = None)


''' plot from files'''


fold_num = 5
ds = 'WH'

df_result = pd.read_excel(r'D:\OneDrive - VUMC\Research\Ongoing Research Projects\My AI projects\AI 3D otitis media\2dCNN\Task 2\Result_{}_T2_2dCNN.xlsx'.format(ds), 
                          sheet_name = 'fold{}'.format(fold_num), header = 0)

# confusion matrix
y_test = df_result['Truth'].to_numpy()
y_test = keras.utils.to_categorical(y_test, 2) 
get_cmx(df_result['Pred lbl'].tolist(), y_test)


# ROC 
y_pred =  df_result['Pred score'].to_numpy()
y_pred_comp = 1 - y_pred
get_roc_curve(np.column_stack((y_pred_comp,y_pred)), y_test, ds)




''' original models'''


fold_num = 5
ds = 'wh'

df_result = pd.read_excel(r'D:\OneDrive - VUMC\Research\Ongoing Research Projects\My AI projects\AI 3D otitis media\Tasks updated 2022-09-03\Results task 2 updated 9-21-22\fold {}\Result_{}_fold{}.xlsx'.format(fold_num, ds, fold_num), 
                          header = 0)

y_test = df_result['Truth'].to_numpy()
y_test = keras.utils.to_categorical(y_test, 2) 
get_cmx(df_result['Pred lbl'].tolist(), y_test)


# ROC 
y_pred =  df_result['Pred score'].to_numpy()
y_pred_comp = 1 - y_pred
get_roc_curve(np.column_stack((y_pred_comp,y_pred)), y_test, ds)
