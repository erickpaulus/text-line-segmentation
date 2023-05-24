#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Several codes are adopted from  https://github.com/ajgallego/document-image-binarization
import os
import re
import sys
import time
import random
import numpy as np
from keras import backend as K
import csv
import cv2
from  PIL import Image
from skimage import  util as ut
import PIL 

#------------------------------------------------------------------------------
def init():
    random.seed(1337)
    np.set_printoptions(threshold=sys.maxsize) 
    np.random.seed(1337)
    sys.setrecursionlimit(40000)

# ----------------------------------------------------------------------------
def print_error(str):
    print('\033[91m' + str + '\033[0m')

# ----------------------------------------------------------------------------
def LOG(fname, str):
    with open(fname, "a") as f:
        f.write(str+"\n")

# ----------------------------------------------------------------------------
def mkdirp(directory):
    if not os.path.isdir(directory):
        os.makedirs(directory)

# ----------------------------------------------------------------------------
# Replace lasts ocurrences of
# Example: rreplace(str, oldStr, newStr, 1):
def rreplace(s, old, new, occurrence):
    li = s.rsplit(old, occurrence)
    return new.join(li)

# ----------------------------------------------------------------------------
# Return the list of files in folder
def list_dirs(directory):
    return [os.path.join(directory, f) for f in os.listdir(directory)
            if os.path.isdir(os.path.join(directory, f))]

# ----------------------------------------------------------------------------
# Return the list of files in folder
# ext param is optional. For example: 'jpg' or 'jpg|jpeg|bmp|png'
def list_files(directory, ext=None):
    return [os.path.join(directory, f) for f in os.listdir(directory)
            if os.path.isfile(os.path.join(directory, f)) and ( ext==None or re.match('([\w_-]+\.(?:' + ext + '))', f) )]

# -----------------------------------------------------------------------------
# Returns an array of paths to the files to process
def load_array_of_files(basepath, folders_in_fold):
    X = []
    print()
    for folder in folders_in_fold:
        full_path = os.path.join(basepath, folder)
        array_of_files = list_files(full_path, ext='jpg|jpeg|bmp|tif|tiff|png')

        for fname_x in array_of_files:
            X.append(fname_x)

    return np.asarray(X)

# ----------------------------------------------------------------------------
def micro_fm(y_true, y_pred):
    beta = 1.0
    beta2 = beta**2.0
    top = K.sum(y_true * y_pred)
    bot = beta2 * K.sum(y_true) + K.sum(y_pred)
    return -(1.0 + beta2) * top / bot
#------------------------------------------------------------------------------
# save seam computation in csv file
def save_data(arr,name):
    with open(name, "w") as f:
        writer = csv.writer(f)
        writer.writerows(arr)
#------------------------------------------------------------------------------
# read seam computation in csv file
def read_csv(csv_file):
    # reading data from a csv file 'Data.csv'
    with open(csv_file, newline='') as file:
        reader = csv.reader(file, delimiter = ',')
        # output list to store all rows
        Output = []
        for row in reader:
            Output.append( [float(i) for i in row[:]])
    return Output
#------------------------------------------------------------------------------
# INTERERSTING FINDING : GREEN COLOR make a better result on line segmentation
def util_alpha_channel(img,opt):
    new_img = np.copy(img)
    if img.ndim == 3 :
        if img.shape[2] == 4:    
            # if there is alpha channel with white color, we need to remove them and apply the same color
            mask = img[:,:,3]
            mask[mask==0] = 0
            
            kernel = np.ones((10,10),np.uint8)
            mask = cv2.erode(mask,kernel,iterations = 1)
            
            img = img[:,:,:3]
            img = cv2.bitwise_and(img, img, mask=mask)
            mask_inv = 255- mask
            
            new_img = np.copy(img)
            average = img.mean(axis=0).mean(axis=0)
            if opt == 'a' :
                for i in range(3) :            
                    new_img[:,:,i] = new_img[:,:,i] + (mask_inv/255 * average[i])
            elif opt == 'w' :
                for i in range(3) :            
                    new_img[:,:,i] = new_img[:,:,i] + mask_inv
            elif opt == 'r' :
                new_img[:,:,2] = new_img[:,:,2] + mask_inv
            elif opt == 'g' :
                new_img[:,:,1] = new_img[:,:,1] + mask_inv
            elif opt == 'b':
                new_img[:,:,0] = new_img[:,:,0] + mask_inv
    return new_img
#------------------------------------------------------------------------------

def segment_line_from_image(path_in,filename, sep_seams, path_out):
    folder_in = os.path.join(path_in, filename)
    print(folder_in)
    image = cv2.imread(folder_in,cv2.IMREAD_UNCHANGED)
    r , c = image.shape
    
    if np.max(image)<=1.0 :
        image=image*255
    result = np.zeros((r,c));
        
    if len(sep_seams)==0 or sep_seams.size == 0 :
        img_seg = np.copy(image)
        result[img_seg < 255] = 1
        check = ut.img_as_bool(img_seg)
        img_seg = PIL.Image.fromarray(check)
        out_fn = os.path.join(path_out, filename[:-4] + '_line_0' + '.bmp')
        img_seg.save(out_fn)
    else:
        sep_seams = np.transpose(sep_seams) # depend on output from previous step        
        
        lines, cols = sep_seams.shape    

        for line in range(lines + 1):
            img_seg = np.copy(image)

            for col in range(cols):
                if line == 0 :                
                    img_seg[sep_seams[line, col]:r, col] = 255
                elif line >= lines :
                    img_seg[0:sep_seams[line - 1, col], col] = 255
                else :                
                    img_seg[0:sep_seams[line - 1, col], col] = 255            
                    img_seg[sep_seams[line, col]:r, col] = 255
            result[img_seg < 255] = line + 1;

            check = ut.img_as_bool(img_seg)
            img_seg = PIL.Image.fromarray(check)
            out_fn = os.path.join(path_out, filename[:-4] + '_line_' + f'{line+1}' + '.bmp')
            img_seg.save(out_fn)
    
    # convert 2D to 1D
    result = result.flatten()
    result = np.transpose(result).astype('uint')
    filename_dat = os.path.join(path_out, filename)
    
    with open(filename_dat + '.dat', "wb") as f:
        f.write(result)     
        print(filename_dat + '.dat')
    return result