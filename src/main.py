import argparse
import cv2
import os

import preprocessing
from segmentation import segmentation
import vectorization
from heterogeneity import get_heterogeneity

def pipeline(args):
    
    path = args.get('path')
    
    img = cv2.imread(path)
    
    matiz, _, _ = preprocessing.transform_hsv(img)
    
    median = preprocessing.filter_median(matiz)
    
    segmented = segmentation(median)
    
    contours = vectorization.generate_image(img, segmented)
    
    get_heterogeneity(contours)
        


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='Segment Trees')
    
    parser.add_argument('--path', type=str, default='./sample1.tif', help='Path to img')
    
    os.makedirs("./preprocessed/", exist_ok=True)
    os.makedirs("./segmentation_result/", exist_ok=True)
    os.makedirs("./result/", exist_ok=True)
    

    args = parser.parse_args()
    kwargs = vars(args)
    
    pipeline(kwargs)