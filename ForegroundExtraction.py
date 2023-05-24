#!/usr/bin/python
# -*- coding: utf-8 -*-
# Several codes are adopted from  https://github.com/ajgallego/document-image-binarization

from __future__ import print_function
import sys, os
import time
import argparse
import cv2
import warnings
import numpy as np
from keras import backend as K
__file__ = os.getcwd()
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import util, utilDataGenerator, utilModelDC5REDNet
from skimage.measure import label, regionprops, regionprops_table

util.init()
warnings.filterwarnings('ignore')
K.set_image_data_format('channels_last')

if K.backend() == 'tensorflow':
    import tensorflow as tf    # Memory control with Tensorflow
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth=True
    sess = tf.compat.v1.Session(config=config)
from pathlib import Path
# ----------------------------------------------------------------------------
def build_SAE_network(config):    
    autoencoder, encoder, decoder = utilModelDC5REDNet.build_DCREDNet(
                                            config.nb_layers,
                                            config.window, config.nb_filters,
                                            config.kernel, config.dropout,
                                            config.stride, config.every)

    autoencoder.compile(optimizer='adam', loss=util.micro_fm, metrics=['mse'])
    
    pkg_models = os.listdir( os.path.join(Path().absolute(), 'MODELS') )
    if config.modelpath.replace('MODELS/', '') in pkg_models:
        config.modelpath = os.path.join(Path().absolute(), config.modelpath)

    autoencoder.load_weights(config.modelpath)
    return autoencoder
# ----------------------------------------------------------------------------
def load_and_prepare_input_image(config):

    img = cv2.imread(config.imgpath, 0)
    assert img is not None, 'Empty file'
    img = np.asarray(img)

    original_rows = img.shape[0]
    origina_cols = img.shape[1]
    if img.shape[0] < config.window or img.shape[1] < config.window:  # Scale approach
        new_rows = config.window if img.shape[0] < config.window else img.shape[0]
        new_cols = config.window if img.shape[1] < config.window else img.shape[1]
        img = cv2.resize(img, (new_cols, new_rows), interpolation = cv2.INTER_CUBIC)

    img = np.asarray(img).astype('float32')
    img = 255. - img

    return img, original_rows, origina_cols
# ----------------------------------------------------------------------------
def denoising(args,activate_demo_opt=False, run_demo=False):        
    #print('output :',args.outFilename)
    if run_demo:
        args.demo = True

    autoencoder = build_SAE_network(args)

    img, rows, cols = load_and_prepare_input_image(args)

    finalImg = img.copy()

    start_time = time.time()

    for (x, y, window) in utilDataGenerator.sliding_window(img, stepSize=args.step, windowSize=(args.window, args.window)):
            if window.shape[0] != args.window or window.shape[1] != args.window:
                continue

            roi = img[y:(y + args.window), x:(x + args.window)].copy()
            roi = roi.reshape(1, args.window, args.window, 1)
            roi = roi.astype('float32')

            prediction = autoencoder.predict(roi)
            prediction = (prediction > args.threshold)

            finalImg[y:(y + args.window), x:(x + args.window)] = prediction[0].reshape(args.window, args.window)

            if args.demo == True:
                demo_time = time.time()
                clone = finalImg.copy()
                clone = 1 - clone
                clone *= 255
                clone = clone.astype('uint8')

                try:
                    cv2.rectangle(clone, (x, y), (x + args.window, y + args.window), (255, 255, 255), 2)
                    cv2.namedWindow('Demo', cv2.WINDOW_NORMAL)
                    cv2.imshow('Demo', clone)
                    cv2.waitKey(1)
                except:
                    print('There was an error displaying the demo. You may need to '
                                  'install the full opencv version. Please run: '
                                  'pip install opencv-python==4.*')
                time.sleep( 0.5 )
                start_time += time.time() - demo_time

    print( 'Time: {:.3f} seconds'.format( time.time() - start_time ) )

    finalImg = 1. - finalImg
    finalImg *= 255.
    finalImg = finalImg.astype('uint8')

    if finalImg.shape[0] != rows or finalImg.shape[1] != cols:
        finalImg = cv2.resize(finalImg, (cols, rows), interpolation = cv2.INTER_CUBIC)

    if args.demo == True:
        try:
            cv2.namedWindow('Demo', cv2.WINDOW_NORMAL)
            cv2.imshow('Demo', finalImg)
            cv2.waitKey(0)
        except:
            print('There was an error displaying the demo. You may need to '
                          'install the full opencv version. Please run: '
                          'pip install opencv-python==4.*')
    
    if args.outFilename != None :
        
        cv2.imwrite(args.outFilename, finalImg)
#     print(finalImg.shape)
    return finalImg
# ----------------------------------------------------------------------------
import matplotlib.pyplot as plt
def remove_tall_object(binarized_img, height):
    label_im = label(binarized_img)
    regions = regionprops(label_im)

    # Create a mask to remove objects based on specific criteria
    mask = np.ones_like(binarized_img, dtype=bool)  # Initialize the mask as all True

    # Iterate over the regions and check the desired criteria
    for region in regions:
        # Check your desired criteria here
        h = region.bbox[2] - region.bbox[0]
        if h <= height * 0.25:
            # Add the region to the mask if it is smaller than the maximum size
            minr, minc, maxr, maxc = region.bbox
            mask[minr:maxr, minc:maxc] = False

    # Apply the mask to remove the objects from the image
    denoised_img = binarized_img.copy()
    denoised_img[~mask] = 0
    denoised_img = (denoised_img*255).astype(np.uint8)

    
    print(np.max(denoised_img))
    
    cv2.imwrite('result/denoised_img.png', denoised_img)
    return (denoised_img)

def remove_big_object(binarized_img):
    label_im = label(binarized_img)
    regions = regionprops(label_im)

    
    # Compute the total height and count of objects
    total_area = 0
    object_count = 0

    # Iterate over the regions and calculate the height
    for region in regions:        
        total_area +=  region.area
        object_count += 1

    # Calculate the average height
    if object_count > 0:
        average_area = total_area / object_count
    else:
        average_area = 0
    print("average_area",average_area)
    masks = []
    bbox = []
    list_of_index = []
    for num, x in enumerate(regions):
        area = x.area
        convex_area = x.convex_area
        
        if (num!=0 and (area >= 2* average_area)):
            masks.append(regions[num].convex_image)
            bbox.append(regions[num].bbox)   
            list_of_index.append(num)
    count = len(masks)
#     print(count)

    for box, mask in zip( bbox, masks):        
        binarized_img[box[0]:box[2], box[1]:box[3]] = 0
    denoised_img = 255-(255*binarized_img)
    cv2.imwrite('result/denoised_img.png', denoised_img)
    return (denoised_img)

