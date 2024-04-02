import sys

def print_help():
    help_message = """
Usage: python {script_name} [OPTIONS]
Set the path configurations with specified command-line arguments.

Options:
    --out_root_folder=PATH     Path to the output root folder. Default is 'output'.
    --scan_root_folder=PATH    Path to the root folder containing CT images. Default is 'CT_images'.
    --scan_id=ID              Identifier for the scan. Default is 'p00726056-231124'.
    --target_side=SIDE        Target side. Options are 'Left' or 'Right'. Default is 'Left'.
    --view_ch=CHANNEL         Model channel configuration, 1 for cholesteatoma, 0 for non-cholesteatoma. Default is 1.
    --optimal_threshold=VALUE Heatmap threshold for visualization. Default is 0.45.

Example:
    python {script_name} --out_root_folder=output --scan_root_folder=CT_images --scan_id=p00726123 --target_side=Left --view_ch=1 --optimal_threshold=0.45
    """.format(script_name=sys.argv[0])
    print(help_message)

def parse_args(defaults):
    args = defaults.copy()

    for arg in sys.argv[1:]:
        if arg in ['-h', '--help']:
            print_help()
            sys.exit()
        if '=' not in arg:
            print(f"Invalid argument format: {arg}")
            print_help()
            sys.exit(1)
        k, v = arg.split('=', 1)  # Split on the first '='
        k = k.lstrip('-')  # Remove leading dashes
        if k in args:
            if k == 'view_ch' or k == 'optimal_threshold':
                v = float(v) if '.' in v else int(v)
            args[k] = v
        else:
            print(f"Unknown argument: {k}")
            print_help()
            sys.exit(1)

    return args

# Default values setup
defaults = {
    'out_root_folder': r'output',
    'scan_root_folder': r'CT_images',
    'scan_id': 'p00726056-231124',
    'target_side': 'Left',
    'view_ch': 1,
    'optimal_threshold': 0.45
}

if len(sys.argv) == 1:
    print_help()
    sys.exit(1)

# Parse the command line arguments
args = parse_args(defaults)

# Use the arguments as needed
# print(args)  # Just for demonstration

import scripts.full_workflow_utils as workflow
import torch
import tensorflow as tf
import os


''' define paths'''
out_root_folder = r'output'
scan_root_folder = r'CT_images'

''' specify scan'''
scan_id = 'p00726056-231124'

target_side = 'Left'

''' get models'''
roi_model_path = r'Model_weights/YOLO_3DCT.pt' 
model_roi = torch.hub.load('ultralytics/yolov5', 'custom', path = roi_model_path)

model_3dcom_path = r"Model_weights/3d_image_classification_task2.h5"
model_3dcom = workflow.get_model(width=128, height=128, depth=32)
model_3dcom.load_weights(model_3dcom_path)
optimal_threshold = 0.45
view_ch = 1 # 1 for cholesteatoma, 0 for non-cholesteatoma


model_3dcom.summary()

last_conv_layer_name = workflow.get_tensorflow_last_layer(model_3dcom) # "conv3d_3" # if necessary, update the name of the last conv layer by checking it through model.summary()


''' execution'''

''' initial assessment of the dicom file''' 
df_scan = workflow.get_scan_info(os.path.join(scan_root_folder, scan_id))
read_path = df_scan.iloc[0]['Path']
dicom_array = workflow.process_scan(read_path)

''' evaluate central layers and crop ROI'''
df = workflow.scan_through(dicom_array, model_roi)
center_img_left, center_xy_left, center_img_right, center_xy_right = workflow.get_center_img_info(df)

roi_left = workflow.crop_ROI(dicom_array, center_img_left, center_xy_left, slice_num = 16, box_size = 128)
roi_right = workflow.crop_ROI(dicom_array, center_img_right, center_xy_right, slice_num = 16, box_size = 128)

roi_img = roi_left if target_side=='Left' else roi_right

''' preprocess for 3d cnn'''
roi_img = (roi_img/255.).astype("float32")  
img_test = tf.expand_dims(roi_img, axis=-1) 
img_test = tf.expand_dims(img_test, axis=0) 


''' get model prediction'''
prediction = model_3dcom.predict(img_test)

''' plot without heatmap'''
# y_pred_lbl = 1 if prediction[0, 1] >= optimal_threshold else 0 
# plot_test_slices(roi_left, scan_id, y_pred_lbl, False)

''' plot with heatmap'''
heatmap_np = workflow.make_gradcam_heatmap(img_test, model_3dcom, last_conv_layer_name, pred_index = view_ch, resize = False)
heatmap_resize = workflow.resize_volume(heatmap_np, method = 'constant')

workflow.display_superimpose_imgs(out_root_folder,scan_id,target_side,img_test, heatmap_resize, preds = prediction, ch=view_ch, alpha = 0.4,  rows = 4, columns = 8,
                         show_origin = True, show_heatmap = True, cmap = 'coolwarm', threshold = optimal_threshold)
