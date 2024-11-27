#!/usr/bin/python39
# FILE: main.py

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@author: Parastoo


"""

#from sam_processor import *
from yolo_processor import *
from segformer_processor import *
import pickle
import argparse
from fun import *
import skimage.io
from PIL import Image

from sam_processor import *
import measure as  measurefunc








parser = argparse.ArgumentParser(description = 'Process Biofilm images.')

parser.add_argument('-help', action="help")
parser.add_argument("-pathimage", type = str , help = "Path for OCT image Experiment")
parser.add_argument("-pathsave", type = str , help = "Path to save segmentes OCT images.")
parser.add_argument("-model", type = str ,  choices=['SAM', 'YOLO', 'SEGFORMER', 'sam','yolo','segformer'] , help = "Choose the model you want to use (YOLO or SAM or SEGFORMER).", default='SEGFORMER')
parser.add_argument("-imagetype", type = str, default = 'oct' , choices=['oct', 'img'] , help = "Choose type of image (oct or img)")
# parser.add_argument("-measuresave", type = str, default = '/home/workprojects/Desktop/finaltestimages/MeasurmentsSave', help = "Measurements save directory")


args = parser.parse_args()

experimentpath = args.pathimage
print(experimentpath)
save_path = args.pathsave
print(save_path)
model = args.model
print(model)
imgtype = args.imagetype
print(imgtype)
# measuresave = args.measuresave
# print(measuresave)



# experimentpath = '/home/workprojects/Desktop/OCTmilestone/biofilm-oct/OCTimages'
# print(experimentpath)
# save_path = '/home/workprojects/Desktop/OCTmilestone/biofilm-oct/Final'
# print(save_path)




def YOLO_image_predictor(img_list, save):
    image_path_list, root_path = get_img_paths(img_list)
    print(image_path_list, root_path)
    
    for img_path in image_path_list:
        
        try:
            filename = os.path.basename(img_path).replace('-Raw.png', '')#.split('.')[0]
            
            img=skimage.io.imread(img_path)
            
            dim = img.shape
            
            if img.shape == (1024, 500, 3):
                pred_mask_yolo = yolo_inference(img_path)
                save_image_yolo(pred_mask_yolo ,save, filename + '-YOLO')
                #plt.imshow(pred_mask_yolo)
                
            else:
                errorList.append('Sample number: ' + img_path +', error message: Wrong image')
                pred_mask_yolo = yolo_inference_2(img_path, filename)
                save_image_yolo(pred_mask_yolo ,save, filename + '-YOLO')
                
                
        except Exception as error1:
            errorList.append('Sample number: ' + img_path +', error message: ' + str(error1))











errorList= []

if model == 'SAM' or model == 'sam':
    image_path_list, root_path = get_image_paths(experimentpath)
    
    for root in root_path:
        #temp = root.replace(experimentpath + '/','')
        if root == experimentpath:
            temp = os.path.basename(root)
            newsavepath = os.path.join(save_path, temp + '-SAM')
        else:
            temp = os.path.relpath(root, experimentpath)
            newsavepath = os.path.join(save_path, temp + '-SAM')
        print(root, newsavepath)
        ProcessImages(root, newsavepath)
        measurefunc.measure_csv_in_folder(newsavepath, newsavepath, 'SAM-Model', 'SAM-' + temp)

elif model == 'YOLO' or model == 'yolo':
    #errorList= []

    if imgtype == "oct":
        image_path_list, root_path = get_image_paths(experimentpath)
        
        print(image_path_list, root_path )
        
        for root in root_path:
            if root == experimentpath:
                temp = os.path.basename(root)
                newsavepath = os.path.join(save_path, temp + '-YOLO')
            else:
                temp = root.replace(experimentpath + '/','').replace(experimentpath + '\\','')
                newsavepath = os.path.join(save_path, temp + '-YOLO')
            
            print(root, newsavepath)
            
            RawfromOCTImages(root, newsavepath) #saves OCTs to png
            YOLO_image_predictor(newsavepath, newsavepath)
            measurefunc.measure_csv_in_folder(newsavepath, newsavepath, 'YOLO-Model', 'YOLO-' + temp)
    
    else:
        
        
        image_path_list, root_path = get_img_paths(experimentpath)
        print(image_path_list, root_path)
        
        
        for root in root_path:
            if root == experimentpath:
                temp = os.path.basename(root)
                newsavepath = os.path.join(save_path, temp + '-YOLO')
            else:
                # temp = root.replace(experimentpath + '/','').replace(root.split('/')[-1],'')
                temp = root.replace(experimentpath + '/','').replace(experimentpath + '\\','')
                newsavepath = os.path.join(save_path, temp + '-YOLO')
            
            try:
                YOLO_image_predictor(root, newsavepath)
                measurefunc.measure_csv_in_folder(newsavepath, newsavepath, 'YOLO-Model', 'YOLO-' + temp)
                
                    
            except Exception as error1:
                errorList.append('Sample number: ' + img_path +', error message: ' + str(error1))
        
    with open(save_path + 'errorList.pkl', 'wb') as f:
        pickle.dump(errorList, f)

elif model == 'SEGFORMER' or model == 'segformer':
    
    if imgtype == "oct":
        image_path_list, root_path = get_image_paths(experimentpath)
        
        print(image_path_list, root_path )
        
        for root in root_path:
            if root == experimentpath:
                temp = os.path.basename(root)
                newsavepath = os.path.join(save_path, temp + '-SEGFORMER')
            else:
                temp = root.replace(experimentpath + '/','').replace(experimentpath + '\\','')
                newsavepath = os.path.join(save_path, temp + '-SEGFORMER')
            
            print(root, newsavepath)
            
            RawfromOCTImages(root, newsavepath) #saves OCTs to png
            image_path_list, root_path = get_img_paths(newsavepath)
            for img in image_path_list:
                image = Image.open(img).convert('RGB')  # Open the image using PIL
                # Process the image one by one
                ProcessImagesSegformer(image, img, newsavepath)
            measurefunc.measure_csv_in_folder(newsavepath, newsavepath, 'SEGFORMER-Model', 'SEGFORMER-' + temp)
    else:
        image_path_list, root_path = get_img_paths(experimentpath)
        print(image_path_list, root_path)
        
        
        for root in root_path:
            if root == experimentpath:
                temp = os.path.basename(root)
                newsavepath = os.path.join(save_path, temp + '-SEGFORMER')
            else:
                # temp = root.replace(experimentpath + '/','').replace(root.split('/')[-1],'')
                temp = root.replace(experimentpath + '/','').replace(experimentpath + '\\','')
                newsavepath = os.path.join(save_path, temp + '-SEGFORMER')

            os.makedirs(newsavepath, exist_ok=True)
            image_path_list, root_path = get_img_paths(root)
            for img in image_path_list:
                image = Image.open(img).convert('RGB')  # Open the image using PIL
                # Process the image one by one
                ProcessImagesSegformer(image, img, newsavepath)
            measurefunc.measure_csv_in_folder(newsavepath, newsavepath, 'SEGFORMER-Model', 'SEGFORMER-' + temp)
















            