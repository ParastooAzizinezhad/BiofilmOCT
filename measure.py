#!/usr/bin/python39
# FILE: measure.py
"""
V modified V2cluster+imageprocessingV3 from cluster
"""
import os
import pandas as pd
import numpy as np
from fun import *
from OctCorrection import *
from ImageProcessing import *
import sys
from PIL import Image
import argparse


def load_png_images(folder_path):
    image_array = []
    file_paths = []

    # Loop through all files in the folder
    print("loading please wait")
    
    for filename in os.listdir(folder_path):
        if filename.endswith('-mask.png'):
            # Construct the full path to the CSV file
            #print(filename)
            file_path = os.path.join(folder_path, filename)
            file_paths.append(os.path.splitext(filename)[0])

            # Read Image to NumPy array
            png_image = Image.open(file_path)
            image_data = np.array(png_image)
            #print(image_data)

            # Append to the image array
            image_array.append(image_data)

    df_paths = pd.DataFrame({'FilePath': file_paths})
    return np.array(image_array), df_paths
    
def measure_csv_in_folder(folder_path, save_path, plate_name, csv_name):
    
    annotated_images, df1 = load_png_images(folder_path)
    print("Shape of the loaded annotated images array:", annotated_images.shape)
    
    thr=0    
    small_obj=30  
    cnames=['model','volume','thickness','Rq', 'Ra', 'density dist.', 'pixel-based density'] # palte is renamed to model
    df2 = pd.DataFrame(columns =cnames )
    for idx, image in enumerate(annotated_images):
        try:
            fig, V,H,Rq,Ra, SL, density=CalcZMap(image[None,:],thr,small_obj) 
            df2=pd.concat([df2, pd.DataFrame([[plate_name ,V,H,Rq,Ra, SL, density]], columns = cnames)])   #append volume, height and roughness results in a dataframe
            file_name = df1.loc[idx, 'FilePath']
            #print(file_name)
            #directory = os.path.dirname(path)
            full_path = os.path.join(save_path, file_name.replace('-mask', '-Line.png'))
            fig.savefig(full_path)
        except Exception as error:
            print('error on index:', idx, 'file: ',file_name, error)
    df2.reset_index(inplace=True, drop=True)
    df = pd.concat([df1, df2], axis=1)
    csv_path = os.path.join(save_path, csv_name+'.csv')
    df.to_csv(csv_path, index=False)

# plate = sys.argv[1]
# folder_path = "/home/rh22708/darpa/dataset/annotated/Plate "+plate+'/'
# measure_csv_in_folder(folder_path)





#print("loading please wait")
#measure_csv_in_folder(folder_path)


"""
# For measurement only


parser = argparse.ArgumentParser(description = 'Calculate Biofilm images.')

parser.add_argument("-pathimage", type = str , help = "Path for OCT csv segmented image Plates.")
parser.add_argument("-pathsave", type = str , help = "Path for OCT image Plates.")
parser.add_argument("-plate", type = str , help = "name for this plate.")
parser.add_argument("-csvname", type = str , help = "name for the csv output file.")

args = parser.parse_args()

folder_path = args.pathimage
save_path = args.pathsave
plate_name = args.plate
csv_name = args.csvname
print(folder_path)
print(save_path)

os.makedirs(save_path, exist_ok=True)  # Create if needed, otherwise do nothing

f = open("errors.txt", "w")
f.writelines(f"running {folder_path}.\n")
f.close()

measure_csv_in_folder(folder_path, save_path, plate_name, csv_name)

f.close()

print("Done")
"""

