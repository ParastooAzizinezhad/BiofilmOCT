#!/usr/bin/python39
# FILE: fun.py
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
# load the images from oct to train_images, train_masks
# Developed by Rahmat

Changed the use of temp files and use of previously extracted OCT files

Some functions has been updated
Functions for extracting OCT files.
(500,1024)
"""
import os
import cv2
import pandas as pd
import numpy as np
from scipy.fftpack import fft,ifft
from scipy.interpolate import interp1d
import tempfile
import zipfile
import warnings
import os
import xmltodict
from warnings import warn
import torch
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor
import matplotlib.pyplot as plt


import shutil



def unzip_OCTFile(filename):
    """
    Unzip the OCT file into a (new) temp folder.
    """

    tempdir = tempfile.gettempdir()
    handle = dict()
    handle['filename'] = filename
    handle['path'] = os.path.join(tempdir, 'OCTData')

    temp_oct_data_folder = os.path.join(handle['path'], os.path.basename(filename).split('.oct')[0])
    handle['temp_oct_data_folder'] = temp_oct_data_folder

    # Clear any existing temporary data
    if os.path.exists(temp_oct_data_folder):
        shutil.rmtree(temp_oct_data_folder)

    # Ensure the directory exists
    os.makedirs(temp_oct_data_folder, exist_ok=True)
    print('\nExtracting {} into {}. Please wait.\n'.format(filename, temp_oct_data_folder))

    # Extract the zip file
    with zipfile.ZipFile(file=handle['filename']) as zf:
        zf.extractall(path=temp_oct_data_folder)

    # Read Header.xml
    header_path = os.path.join(temp_oct_data_folder, 'Header.xml')
    if not os.path.exists(header_path):
        raise FileNotFoundError(f"Header.xml not found in extracted contents of {filename}")

    with open(header_path, 'rb') as fid:
        xmldoc = fid.read()

    # Convert Header.xml to dict
    handle_xml = xmltodict.parse(xmldoc)
    handle.update(handle_xml)

    # Return the handle
    return handle


"""
def unzip_OCTFile(filename):
    tempdir = tempfile.gettempdir()
    handle = dict()
    handle['filename'] = filename
    handle['path'] = os.path.join(tempdir, 'OCTData')

    temp_oct_data_folder = os.path.join(handle['path'],os.path.basename(filename).split('.oct')[0])
    handle['temp_oct_data_folder'] = temp_oct_data_folder
    if os.path.exists(temp_oct_data_folder) and os.path.exists(os.path.join(temp_oct_data_folder, 'Header.xml')):
        warn('Reuse data in {}\n'.format(temp_oct_data_folder))
    else:
        print('\nTry to extract {} into {}. Please wait.\n'.format(filename,temp_oct_data_folder))
        if not os.path.exists(handle['path']):
            os.mkdir(handle['path'])
        if not os.path.exists(temp_oct_data_folder):
            os.mkdir(temp_oct_data_folder)

        with zipfile.ZipFile(file=handle['filename']) as zf:
            zf.extractall(path=temp_oct_data_folder)

    # read Header.xml
    with open(os.path.join(temp_oct_data_folder, 'Header.xml'),'rb') as fid:
        up_to_EOF = -1
        xmldoc = fid.read(up_to_EOF)

    # convert Header.xml to dict
    handle_xml = xmltodict.parse(xmldoc)
    handle.update(handle_xml)
    return handle

"""


def get_OCTFileMetaData(handle, data_name):
    """
    The metadata for files are store in a list.
    The artifact 'data\\' stems from windows path separators and may need fixing.
    On mac and linux the file names will have 'data\\' as a name prefix.
    """
    # Check if data_name is available
    data_names_available = [d['#text'] for d in handle['Ocity']['DataFiles']['DataFile']]
    data_name = 'data\\'+data_name+'.data' # check this on windows
    assert data_name in data_names_available, 'Did not find {}.\nAvailable names are: {}'.format(data_name,data_names_available)

    metadatas = handle['Ocity']['DataFiles']['DataFile'] # get list of all data files
    # select the data file matching data_name
    metadata = metadatas[np.argwhere([data_name in h['#text'] for h in handle['Ocity']['DataFiles']['DataFile']]).squeeze()]
    return handle, metadata

def get_OCTIntensityImage(filename):
    """
    Example how to extract Intensity data
    """
    python_dtypes = {'Colored': {'4': np.int32, '2': np.int16},
                     'Real': {'4': np.float32},
                     'Raw': {'signed': {'1': np.int8, '2': np.int16},
                             'unsigned': {'1': np.uint8, '2': np.uint16}}}
    #Load image
    handle = unzip_OCTFile(filename);
    handle.update({'python_dtypes': python_dtypes})


    handle, metadata = get_OCTFileMetaData(handle, data_name='Intensity')
    data_filename = os.path.join(handle['temp_oct_data_folder'], metadata['#text'])
    img_type = metadata['@Type']
    dtype = handle['python_dtypes'][img_type][metadata['@BytesPerPixel']] # This is not consistent! unsigned and signed not distinguished!
    sizeX = int(metadata['@SizeX'])
    sizeZ = int(metadata['@SizeZ'])
    data = (np.fromfile(data_filename, dtype=(dtype, [sizeX,sizeZ])))
    data = np.moveaxis(data, -1, 1)
    data = data[0]
    data = cv2.resize(data, (500, 1024))
    data = data[None,:]
    return data

def get_img_paths(folder_path):
    """
    Get the paths only for images (png, jpg or jpeg) to be used as input for segmentation models.
    for OCT images use get_image_paths
    """
    image_paths = []
    root_path = []
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.lower().endswith(('.png')) or file.lower().endswith(('.jpg')) or file.lower().endswith(('.jpeg')):
                image_paths.append(os.path.join(root, file))
                root_path.append(root)
    return image_paths, set(root_path)

def get_image_paths(folder_path):
    """
    Get the paths for OCT images.
    """
    image_paths = []
    root_path = []
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.lower().endswith(('.oct')):
                image_paths.append(os.path.join(root, file))
                root_path.append(root)
    return image_paths, set(root_path)

# Function to load images into a NumPy array and df
def load_images(folder_path):
    image_paths, root_path = get_image_paths(folder_path)
    print(image_paths)
    df = pd.DataFrame({'FilePath': image_paths})
    images = [get_OCTIntensityImage(path) for path in image_paths]
    return df, np.array(images)

def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)

def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)

def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0,0,0,0), lw=2))

def sam_predict(CHECKPOINT_PATH, img_array, input_point, input_label, multimask_output = False):
    DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    MODEL_TYPE = "vit_h"

    sam = sam_model_registry[MODEL_TYPE](checkpoint=CHECKPOINT_PATH).to(device=DEVICE)
    mask_predictor = SamPredictor(sam)  
    out_img_array = []
    for img_rgb in img_array:
        mask_predictor.set_image(img_rgb) 
        masks, scores, logits = mask_predictor.predict(
            point_coords=input_point,
            point_labels=input_label,
            multimask_output=multimask_output, # False for include only the best mask
        )
        for i, (mask, score) in enumerate(zip(masks, scores)):
            out_img_array.append(mask)
    return np.array(out_img_array)
