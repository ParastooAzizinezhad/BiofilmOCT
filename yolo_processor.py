#!/usr/bin/python39
# FILE: yolo_processor.py
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
YoloProcessor

"""

from ultralytics import YOLO
from PIL import Image
import skimage
import tempfile
import numpy as np
import os

# Load YOLOv8 model
yolo_model = YOLO(r'Models_pretrained/best.pt')


def yolo_inference(image):
    results = yolo_model(image)
    
    if results[0].masks == None:  # numel() gives the total number of elements
    # Create a zero mask with the shape (h, w)
        mask = np.zeros((1,500,1024), dtype=np.uint8)
    else:
    # If not empty, proceed as normal
    #mask = mask_data.cpu().numpy()
        mask = results[0].masks.data.cpu().numpy()
    #mask = results[0].masks.data.cpu().numpy()
    #print(mask.shape)
    mask = Image.fromarray(mask[0,:,:])
    mask = mask.resize((500,1024))
    mask = np.array(mask)
    mask = np.where(mask > 0, 1, 0)
    return mask




def yolo_inference_2(model, image, name):
    
    input_image = Image.open(image)
    grayscale_img = input_image.convert('L')
    rgb_img = grayscale_img.convert('RGB')
    im = np.array(rgb_img)
    print(im.shape)
    h, w, _ = im.shape
    
    #rgb_img.save('grayscale_image.png')
    
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create a temporary file path within the temporary directory
        temp_filename = os.path.join(temp_dir, f'grayscale_{name}.png')
        
        # Save the image to the temporary file
        rgb_img.save(temp_filename)
        print(f"Image saved temporarily at: {temp_filename}")
    
        # Use the temporary file (e.g., open and display)
        #temp_img = Image.open(temp_filename)

        results = yolo_model(temp_filename)
    
    #mask = results[0].masks.data.cpu().numpy()
    if results[0].masks == None:  # numel() gives the total number of elements
    # Create a zero mask with the shape (h, w)
        mask = np.zeros((1, h, w), dtype=np.uint8)
    else:
    # If not empty, proceed as normal
    #mask = mask_data.cpu().numpy()
        mask = results[0].masks.data.cpu().numpy()
        
    # mask = Image.fromarray(mask)
    print(mask.shape)
    # mask_copy = mask.copy()
    # mask_copy.resize((mask_copy.shape[0],h,w))
    # mask = Image.fromarray(mask[0,:,:])
    mask_new = mask[0]
    for ch in range(1,mask.shape[0]):
        mask_new = mask_new + mask[ch]
    mask = np.array(mask_new)
    mask = np.where(mask > 0, 1, 0)
    return mask



def save_image_yolo(thearray ,save_path, name):
    #image_save = Image.fromarray(thearray)
    im = Image.fromarray((thearray * 255).astype(np.uint8))

    #NameCode = f'{code}-Raw'
    fileName = os.path.join(save_path, name +'-mask.png')
    os.makedirs(save_path, exist_ok=True)  # Create if needed, otherwise do nothing
    im.save(fileName)
    
    
    
