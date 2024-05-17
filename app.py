from imutils import paths
from detecto.core import Model
import cv2
import numpy as np
import os
import datetime
from imutils import paths
from pathlib import Path
import keras_ocr
import warnings
warnings.filterwarnings('ignore')
import logging
import json





keras_pipeline = keras_ocr.pipeline.Pipeline()
model =Model.load('model/bib_large2_weights.pth', ['bib'])


def bib(image):
    
    image1=image
    labels, boxes, scores = model.predict(image1)
    loc=boxes
    loc=loc.tolist()  
    acc=scores
    acc=acc.tolist()
    box=[]
    for i,j in zip(acc,loc):
        if i>=0.80: 
            box.append(j)
            
    number=[]
    try:
        
        
        for i in box:
            xmin, ymin = (round(i[0]),round(i[1]))
            xmax, ymax = (round(i[2]),round(i[3]))
            imsub = image1[ ymin:ymax, xmin:xmax ]
    
            image2 = keras_ocr.tools.read(imsub)        
            gray1 = image2.astype(np.uint8)
            boxes = keras_pipeline.recognize([gray1])[0]
            for text ,i in boxes:
                x=xmin
                y=ymin
                w=xmax-xmin
                h=ymax-ymin
                t=text
                bib=(x,y,w,h,t)
                number.append(bib)
        
    except Exception as e:
        number=[]

    return number





imagePaths = list(paths.list_images('test'))


for i in imagePaths:
    basename=i
    head, tail = os.path.split(basename)
    image1=cv2.imread(basename)

    r=bib(image1)
    print(r)


        



