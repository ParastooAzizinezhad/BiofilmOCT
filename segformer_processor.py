#!/usr/bin/python39
# FILE: segformer_processor.py


from transformers import AutoImageProcessor, AutoModelForSemanticSegmentation
from PIL import Image
import torch
import torch.nn as nn
import os
import matplotlib.pyplot as plt
import argparse 

def ProcessImagesSegformer(image, file_path, save_path):
    # Prepare the image for the model
    inputs = image_processor(images=image, return_tensors="pt")

    # Run the model on the image
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits

    # Rescale logits to original image size
    logits = nn.functional.interpolate(logits.detach().cpu(),
                                        size=image.size[::-1], # (height, width)
                                        mode='bilinear',
                                        align_corners=False)

    # Get the predicted segmentation map
    predicted = logits.argmax(1).squeeze().numpy()
    predicted = (predicted * 255).astype('uint8')
    # print(predicted[predicted!=0])
    # Save the predicted segmentation map
    # file_name = os.path.basename(file_path)
    # directory = os.path.dirname(file_path)
    full_path = os.path.join(save_path, file_path.split('/')[-1].replace('-Raw.png', '-SEGFORMER-mask.png'))
    # plt.imsave(full_path, predicted)
    img = Image.fromarray(predicted, mode='L')
    img.save(full_path)


model = AutoModelForSemanticSegmentation.from_pretrained('Models_pretrained/Segformer/')
model.eval()
image_processor = AutoImageProcessor.from_pretrained('Models_pretrained/Segformer/')
