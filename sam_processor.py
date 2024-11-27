#!/usr/bin/python39
# FILE: sam_processor.py

"""
SAM prosessor (This method is highly dependent on the substrate)

The weight is located in the folder. ocationally, it gets corrupted in the download/upload/extracting process.
If there is any error related just delete the weight and download the new weight directry to the directory using:
    !wget -q https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth

@author: Parastoo
V without saving Box,OTSU
"""


from fun import *
from OctCorrection import *
from ImageProcessing import *
import matplotlib.pyplot as plt
from skimage import morphology
import os
import torch
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor
import numpy as np
from PIL import Image
import argparse
import collections
import matplotlib as p
import pandas as pd
import pickle
import cv2
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline



# CHECKPOINT_PATH = os.path.basename("sam_vit_h_4b8939.pth")
CHECKPOINT_PATH = 'Models_pretrained/sam_vit_h_4b8939.pth'
DEVICE = torch.device('cpu') #('cuda:0' if torch.cuda.is_available() else 'cpu')
MODEL_TYPE = "vit_h"

sam = sam_model_registry[MODEL_TYPE](checkpoint=CHECKPOINT_PATH).to(device=DEVICE)
mask_predictor = SamPredictor(sam)




    






def get_image_path(folder_path):
    image_path_list = get_image_paths(folder_path)
    return image_path_list



def save_image(thearray ,save_path, name):
    image_save = Image.fromarray(thearray)
    #NameCode = f'{code}-Raw'
    fileName = os.path.join(save_path, name +'.png')
    os.makedirs(save_path, exist_ok=True)  # Create if needed, otherwise do nothing
    image_save.save(fileName)
    
    
    
def RawfromOCTImages(sample_path, save_path):
    
    df, image_array = load_images(sample_path)
    no,ch,h,w = image_array.shape
    
    df['filename'] = df['FilePath'].apply(os.path.basename)
    
    for j in range (no):
        try:

            octname = df['filename'][j]
            code = octname.replace(".oct", "")
            repeat_counter = 0
            #save_path = "/home/workprojects/Desktop/OCTDataset/"
            #code = f'OCT_{SampleNo}_{j}'
            #code = f'OCT_{octname}'
            
            #fileName = os.path.join(save_path, NameCode)

            workingimage = image_array[j,0]

            #Raw = Image.fromarray(workingimage).astype(np.uint8)
            #NameCode = f'{code}-Raw'
            #fileName = os.path.join(save_path, NameCode +'.png')
            #os.makedirs(save_path, exist_ok=True)  # Create if needed, otherwise do nothing
            #Raw.save(fileName)
            #p.image.imsave(fileName, workingimage)

            img_rgb = cv2.cvtColor(workingimage, cv2.COLOR_GRAY2RGB).astype(np.uint8) # first image first layer. as far as I knwo, SAM only get RGB image
            #save raw image
            NameCode = f'{code}-Raw'
            save_image(img_rgb ,save_path, NameCode)
        except Exception as error1:
            print(error1)

            errorList.append('Sample number: ' + str(octname) + ', error message: ' + str(error1))
            


def GetBoxPoints(image):
    
    Y, X = image.shape
    BoxXmin = 0
    BoxXmax = X
    
    image[0:100] = 0
        
    MaxValY = image.argmax(axis = 0)
    
    elements_count = collections.Counter(MaxValY)
    elements_count = sorted(elements_count.items(), reverse=True)
    elements_count = dict(elements_count)

    BoxYmax = max(elements_count, key=elements_count.get)
    
    
    upperpart = image[:BoxYmax,:]
    uppermaxval = upperpart.max(axis=1)
    uppermean = uppermaxval.mean()
    #print(uppermean)
    
    
    ForYmin=np.where(uppermaxval<uppermean)

    BoxYmin = ForYmin[0][-1]

    # plt.imshow(upperpart[BoxYmin:,:])
    """
    plt.figure(figsize=(10,10))
    plt.imshow(image)
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0,0,0,0), lw=2))
    """
    return BoxYmin, BoxYmax, BoxXmin, BoxXmax, uppermean


def find_other_boxes(BoxYmin, uppermean, image):
    newBoxYmin = BoxYmin - 25
    check = True
    crop = 10
    upperBox = image[crop:newBoxYmin,:]
    counter = 0
    Boxpoints = []
    crop_check = False
    while check: 
        counter = counter + 1
        area_check, newUpperBox, newLowerBox = check_available_area(upperBox, uppermean)
        if area_check and (len(newUpperBox[0]) == 0 or len(newLowerBox[0]) == 0):
            for crop in [20,30,40,50,60]:
                upperBox = image[crop:newBoxYmin,:]
                area_check, newUpperBox, newLowerBox = check_available_area(upperBox, uppermean)
                if area_check and (len(newUpperBox[0]) == 0 or len(newLowerBox[0]) == 0):
                    continue
                else:
                    crop_check = True
                    break
        else:
            crop_check = True
        
        if area_check and crop_check:
            #plt.imshow(upperBox[newUpperBox[0][-1]-25:newLowerBox[0][-1]+25])
            #plt.imshow(upperBox)
            Boxpoints.append(np.array([0,newUpperBox[0][-1]-25,500,newLowerBox[0][-1]+25]))
            newBoxYmin = newUpperBox[0][-1]
            #print(newBoxYmin)
            upperBox = image[crop:newBoxYmin,:]
            #print(upperBox.shape)
            if upperBox.shape[0] < 50:
                check = False
        else:
            check = False
    return Boxpoints


def show_masknew(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([0/255, 180/255, 255/255, 0.2])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)



def check_available_area(upperBox, uppermean):
    #print('******')
    #MaxValY = upperBox.max(axis = 1)
    #uppermean = uppermaxval.mean()
    #print(uppermean)
    counterRowData = upperBox>uppermean
    counterRowData1 = np.where(counterRowData == False, 0, 1)
    counterRowData2 = np.sum(counterRowData1, axis=1)
    #print(counterRowData2)

    elements_count = collections.Counter(counterRowData2)
    elements_count = dict(elements_count)
    #for key, value in elements_count.items():
    #   print(f"{key}: {value}")

    newBoxCounter = 0
    for i in elements_count.keys():
        if i > uppermean:
            newBoxCounter = newBoxCounter + elements_count[i]
    if newBoxCounter > 20:
        threshold_data = counterRowData2.max() - counterRowData2.mean()
        counterRowData2Mean = counterRowData2.mean()
        #print(counterRowData2Mean)
        lowerMean = np.where(counterRowData2>counterRowData2Mean)
        #print(threshold_data)
        newUpperMean = np.where(counterRowData2>threshold_data)
        #print(newUpperMean)
        newUpperMin = np.where(counterRowData2[:newUpperMean[0][0]] < counterRowData2Mean)
        #print(newUpperMin)
        #print(np.mean(upperBox[newUpperMin[0][-1]:]))

        newLowerMean = np.where(counterRowData2>5)
        #print(newLowerMean)
        return True, newUpperMin, newLowerMean
    
    return False, 0, 0


def calculate_error(raw_img, out_img, Boxpoints, df_error, file_name, uppermean):
    needToRunAgain = False
    boxImages = []
    boxMask = []
    for point in Boxpoints:
        boxImages.append(raw_img[point[1]-25:point[3]])
        boxMask.append(out_img[point[1]-25:point[3]])
    
    rawImgMaxNo = []
    outImgMaxNo = []
    for boxNo in range(len(boxImages)):
        mean = np.mean(np.max(boxImages[boxNo], axis=1))
        counterRowData = np.sum(np.where(boxImages[boxNo]>uppermean, 1, 0), axis=1)
        #print(counterRowData)
        rawImgMaxNo.append(counterRowData)
        outImgMaxNo.append(np.sum(boxMask[boxNo], axis=1))
        
    rawMean = []
    outMean = []
    for no in range(len(rawImgMaxNo)):
        rawMean.append(np.where(rawImgMaxNo[no] > np.mean(rawImgMaxNo[no]), 1, 0))
        outMean.append(np.where(outImgMaxNo[no] > np.mean(outImgMaxNo[no]), 1, 0))
        
    correctBoxNo = {}
    correctBoxPercent = {}
    for meanNo in range(len(rawMean)):
        correctNo = 0
        idxNo = 0
        for idxNo in range(len(rawMean[meanNo])):
            if rawMean[meanNo][idxNo] != outMean[meanNo][idxNo]:
                correctNo = correctNo + 1
        correctBoxNo[meanNo] = correctNo / (idxNo+1)
        correctBoxPercent[meanNo] = correctNo
        
        new_row = {'File Name': file_name, 'Box Position': Boxpoints[meanNo], 'Fault No':correctNo, 'Percent':correctNo/(idxNo+1)}
        df_error.loc[len(df_error)] = new_row
        if correctBoxNo[meanNo] > 0.70:
            Boxpoints[meanNo][1] = Boxpoints[meanNo][1] + (np.where(rawMean[meanNo]==1)[0][0] / 2)
            Boxpoints[meanNo][3] = Boxpoints[meanNo][3] + (np.where(rawMean[meanNo]==1)[0][0])
            needToRunAgain = True
        
    return correctBoxNo, correctBoxPercent, Boxpoints, needToRunAgain, df_error


def remove_under_substrate(original_image):
    original_image[0:100] = 0
    Y = original_image.argmax(axis = 0)

    X = np.arange(len(Y))
    X_reshaped = X.reshape(-1, 1)

    degree = 2

    # Create a pipeline for polynomial regression
    model = make_pipeline(PolynomialFeatures(degree), LinearRegression())
    model.fit(X_reshaped, Y)

    # Generate predictions for the curve
    #x_curve = np.linspace(min(X), max(X), 100).reshape(-1, 1)
    #y_curve = model.predict(x_curve)
    y_predicted = model.predict(X_reshaped)
    
    Ymax = y_predicted.max()
    
    convertimage = cv2.convertScaleAbs(original_image)
    _, binary_image = cv2.threshold(convertimage, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    binary_rgb = cv2.cvtColor(binary_image, cv2.COLOR_GRAY2RGB).astype(np.uint8)
    
    
    #indy, indx = original_image.shape
    #background_color = np.array([1.0])

    for x in range(len(y_predicted)):
        y= int(y_predicted[x])+10
        #background_color = np.zeros_like(theimage[x,y:])
        #print(xs,y)
        binary_rgb[y:,x] = 255


 
    #plt.scatter(X,Y, label='Data points')
    #plt.imshow(theimage)
    #plt.plot(X, y_predicted, label=f'Polynomial Regression (Degree {degree})', color='red')
    #plt.xlabel('X-axis')
    #plt.ylabel('Y-axis')
    #plt.legend()
    #plt.show()
    
    return binary_rgb, Ymax










# Get the boxes and create images


def ProcessImages(sample_path, save_path):
    
    df_error = pd.DataFrame(columns = ['File Name', 'Box Position', 'Fault No', 'Percent'])
    
    errorList = []
    df, image_array = load_images(sample_path)
    no,ch,h,w = image_array.shape
    
    df['filename'] = df['FilePath'].apply(os.path.basename)
    
    for j in range (no):
        try:

            octname = df['filename'][j]
            code = octname.replace(".oct", "")
            repeat_counter = 0
            #save_path = "/home/workprojects/Desktop/OCTDataset/"
            #code = f'OCT_{SampleNo}_{j}'
            #code = f'OCT_{octname}'

            #fileName = os.path.join(save_path, NameCode)

            workingimage = image_array[j,0]

            #Raw = Image.fromarray(workingimage).astype(np.uint8)
            #NameCode = f'{code}-Raw'
            #fileName = os.path.join(save_path, NameCode +'.png')
            #os.makedirs(save_path, exist_ok=True)  # Create if needed, otherwise do nothing
            #Raw.save(fileName)
            #p.image.imsave(fileName, workingimage)

            img_rgb = cv2.cvtColor(workingimage, cv2.COLOR_GRAY2RGB).astype(np.uint8) # first image first layer. as far as I knwo, SAM only get RGB image
            #save raw image
            NameCode = f'{code}-Raw'
            save_image(img_rgb ,save_path, NameCode)




            #convert images
            convertimage = cv2.convertScaleAbs(workingimage)
            _, binary_image = cv2.threshold(convertimage, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

            #save threshold results
            # NameCode = f'{code}-OTSU'
            # save_image(binary_image ,save_path, NameCode)


            binary_rgb = cv2.cvtColor(binary_image, cv2.COLOR_GRAY2RGB).astype(np.uint8) # first image first layer. as far as I knwo, SAM only get RGB image

            removed_binary , regressionYmax = remove_under_substrate(workingimage)
            # print(regressionYmax)

            #Get the first box positions for SAM
            BoxYmin, BoxYmax, BoxXmin, BoxXmax, uppermean = GetBoxPoints(workingimage)
            #Find if there is anything above the fist box and create the next boxes
            Boxpoints = find_other_boxes(BoxYmin, uppermean, workingimage)



            mask_predictor.set_image(removed_binary) #replace with img_rgb for raw image, binary_rgb for OTSU results and removed_binary
            box_point = np.array([0,BoxYmin-25,500,int(regressionYmax)+25]) # prompt points # replaced BoxYmax+25 with regression results for now
            Boxpoints.append(box_point)

            AllMasks = []
            AllScores =[]
            AllLogits =[]

            #input_label = np.array([1,0,0]) #corresponding label: 0 negative label, 1 positive label

            #run sam Box for all boxes calculated
            for Box in Boxpoints: 
                masks, scores, logits = mask_predictor.predict(
                        box=Box,
                        #point_labels=input_label,
                        multimask_output=False, # False for include only the best mask
                )
                AllMasks.append(masks)
                AllScores.append(scores)
                AllLogits.append(logits)
            #add all boxs' masks to a single mask
            masktokeep = 0
            for i in range(len(AllMasks)):
                #print(i)
                for k, (mask, score) in enumerate(zip(AllMasks[i], AllScores[i])):
                    masktokeep = masktokeep + mask
            masktokeep = np.where(masktokeep >= 1, 1, 0)
            #print('Save image 1')
            # plt.figure(figsize=(10,10))
            # plt.imshow(img_rgb)
            # show_masknew(masktokeep, plt.gca())
            # for i in range(len(AllMasks)):
            #     show_box(Boxpoints[i], plt.gca())
            # NameCode = f'{code}-Box'
            # fileName = os.path.join(save_path, NameCode)
            # plt.savefig(fileName+'.png',bbox_inches = 'tight')


            #print('Save image 2')
            NameCode = f'{code}-mask'
            fileName = os.path.join(save_path, NameCode)
            #print('stage 0')
            FinalMask = Image.fromarray((masktokeep*255).astype(np.uint8))

            #print('stage 1')
            os.makedirs(save_path, exist_ok=True)  # Create if needed, otherwise do nothing
            FinalMask.save(fileName+'.png')
            #p.image.imsave(fileName + '.png', masktokeep)
            #print('stage 2')

            #os.makedirs(csvpath, exist_ok=True)  # Create if needed, otherwise do nothing
            # np.savetxt(fileName+".csv", masktokeep, delimiter=",")
            #print('stage 3')
            #plt.savefig(f'figure{NameCode}.png')
            #plt.show()
            #calculatemeasurements(masktokeep,code)
            # with open(fileName + '.txt', 'a') as f:
            #         for Box in Boxpoints:
            #             f.writelines(str(Box) + '\n')
            #         f.close()
            #check verification
            calculated_error_per, calculated_error, Boxpoints, needToRunAgain, df_error = calculate_error(workingimage, masktokeep, Boxpoints, df_error, code, uppermean)



            if needToRunAgain:
                AllMasks = []
                AllScores =[]
                AllLogits =[]
                #input_label = np.array([1,0,0]) #corresponding label: 0 negative label, 1 positive label
                for Box in Boxpoints: 
                    masks, scores, logits = mask_predictor.predict(
                            box=Box,
                            #point_labels=input_label,
                            multimask_output=False, # False for include only the best mask
                    )
                    AllMasks.append(masks)
                    AllScores.append(scores)
                    AllLogits.append(logits)

                masktokeep = 0
                for i in range(len(AllMasks)):
                    #print(i)
                    for k, (mask, score) in enumerate(zip(AllMasks[i], AllScores[i])):
                        masktokeep = masktokeep + mask
                masktokeep = np.where(masktokeep >= 1, 1, 0)
                #print('Save image 1')
                # plt.figure(figsize=(10,10))
                # plt.imshow(img_rgb)
                # show_masknew(masktokeep, plt.gca())
                # for i in range(len(AllMasks)):
                #     show_box(Boxpoints[i], plt.gca())
                # NameCode = f'{code}-Box-repeated'
                # fileName = os.path.join(save_path, NameCode)
                # plt.savefig(fileName+'.png',bbox_inches = 'tight')


                #print('Save image 2')
                NameCode = f'{code}-repeated-mask'
                #fileName = os.path.join(save_path, NameCode)
                #print('stage 0')
                FinalMask = (masktokeep*255).astype(np.uint8)
                #FinalMask = Image.fromarray((masktokeep*255).astype(np.uint8))
                #print('stage 1')
                #os.makedirs(save_path, exist_ok=True)  # Create if needed, otherwise do nothing
                #FinalMask.save(fileName+'.png')
                save_image(FinalMask, save_path, NameCode)
                #p.image.imsave(fileName + '.png', masktokeep)
                #print('stage 2')

                #os.makedirs(csvpath, exist_ok=True)  # Create if needed, otherwise do nothing
                # np.savetxt(fileName+f"-repeated.csv", masktokeep, delimiter=",")
                #print('stage 3')
                #plt.savefig(f'figure{NameCode}.png')
                #plt.show()
                #repeat_counter = repeat_counter + 1
                #calculatemeasurements(masktokeep,code)
                # with open(fileName + '-repeated.txt', 'a') as f:
                #     for Box in Boxpoints:
                #         f.writelines(str(Box) + '\n')
                #     f.close()
                calculated_error_per, calculated_error, Boxpoints, needToRunAgain,df_error = calculate_error(workingimage, masktokeep, Boxpoints, df_error, code+'-repeated', uppermean)


        except Exception as error1:
            print(error1)

            errorList.append('Sample number: ' + str(octname) + ', error message: ' + str(error1))
    
    print(f"Processing {sample_path} Completed.")
    
    # df_error.to_csv(save_path + 'errors.csv', index=False)
    print(f"Processing Completed.")
    
    # if len(errorList) > 0:
    #     for i in errorList:
    #         print(i)

    with open(save_path + 'errorList.pkl', 'wb') as f:
        pickle.dump(errorList, f)
