"""
@author: Erick Paulus, 2022
"""

#!/usr/bin/python
# -*- coding: utf-8 -*-
from __future__ import print_function
import sys, os
import fnmatch
import glob
import pathlib
import cv2
import argparse
import numpy as np
import warnings
import step1
import SeamCarving
import util
from skimage.filters import sobel,scharr,prewitt
# from keras import backend as K


# import tensorflow as tf


sys.path.append(os.path.dirname(os.path.abspath(__file__)))
# import util, utilFit, utilDataGenerator, utilModelREDNet

# util.init()
# warnings.filterwarnings('ignore')
# K.set_image_data_format('channels_last')

# if K.backend() == 'tensorflow':
#     import tensorflow as tf    # Memory control with Tensorflow
#     config = tf.compat.v1.ConfigProto()
#     config.gpu_options.allow_growth=True
#     sess = tf.compat.v1.Session(config=config)




# ----------------------------------------------------------------------------
def parse_menu():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('-fname',        default='CB-3-18-90-12.tif',       help='base path to datasets')
    parser.add_argument('-path',        default='datasets/Exp1-mixedPLM-GR',       help='base path to datasets')
    parser.add_argument('-db',          default=0 ,  choices=['palm','dibco'],  help='Database name')
    parser.add_argument('-dbp',                                                    help='Database dependent parameters [dibco fold, palm gt]')
    parser.add_argument('-modelpath', default='MODELS/model_weights_DC5_palm_6_nb15_320x320_s128_drop0.2_f16_k5_s2_se1_e200_b10_esp.h5', type=str,   help='Weights filename for image denoising')
    parser.add_argument('-smooth',     default=40, help='smoothing for projection profile matching')
    parser.add_argument('-s',          default=6, help='number of image slices for projection profile matching')
    parser.add_argument('-sigma',      default=3, help='standard deviation for gaussian smoothing')
    parser.add_argument('--aug',        action='store_true', help='Load augmentation folders')
    parser.add_argument('-w',           default=320,    dest='window',              type=int,   help='window size')
    parser.add_argument('-step',        default=128,     dest='step',                type=int,   help='step size. -1 to use window size')
    parser.add_argument('-f',           default=16,     dest='nb_filters',          type=int,   help='nb_filters')
    parser.add_argument('-k',           default=5,      dest='kernel',              type=int,   help='kernel size')
    parser.add_argument('-drop',        default=0.2,    dest='dropout',           type=float, help='dropout value')
    parser.add_argument('-page',        default=-1,                                 type=int,   help='Page size to divide the training set. -1 to load all')
    parser.add_argument('-start_from',  default=0,                                  type=int,   help='Start from this page')
    parser.add_argument('-super',       default=1,      dest='nb_super_epoch',      type=int,   help='nb_super_epoch')
    parser.add_argument('-th',          default=0.5,    dest='threshold',           type=float, help='threshold. -1 to test from 0 to 1')
    parser.add_argument('-e',           default=200,    dest='nb_epoch',            type=int,   help='nb_epoch')
    parser.add_argument('-b',           default=10,     dest='batch',               type=int,   help='batch size')
    parser.add_argument('-esmode',      default='p',    dest='early_stopping_mode',             help="early_stopping_mode. g='global', p='per page'")
    parser.add_argument('-espat',       default=10,     dest='early_stopping_patience',type=int,help="early_stopping_patience")
    parser.add_argument('-verbose',     default=1,                                  type=int,   help='1=show batch increment, other=mute')
    parser.add_argument('-stride',      default=2,      type=int,   help='Conv-Net stride')
    parser.add_argument('-every',       default=1,      type=int,   help='Conv-Net skip conection every x layers')
    parser.add_argument('--test',       action='store_true', help='Only run test')
    parser.add_argument('-denoised',    default=True,   type=bool,   help='implement denoising')
    parser.add_argument('-nb_layers',   default=15,     type=int,    help='number of conv. layer')
    parser.add_argument('-demo',        default=False,   type=bool )
    parser.add_argument('-same',        default=False,   type=bool )
    parser.add_argument('-outFilename',        default='out.png', type=str )
    parser.add_argument('-off',        default=5, type=int )

    args = parser.parse_args()

    if args.step == -1:
        args.step = args.window

    return args

def main(args=None):
    # args = parse_menu()

    # List all files in a directory using os.listdir
    # List all files in a directory using os.listdir
    # basepath = 'datasets/Exp1-mixedPLM-GR'
    # gt_path = 'datasets/Exp1-mixedPLM-GT'
    # # bin_path = 'result/Exp1-mixedPLM-Binarized' 
    # result_path = 'results/result_Exp1-mixedPLM-GR'
    # result_dat ='results/result_Exp1-mixedPLM-Dat'

    # basepath = 'ChallengeB-TrackMixed_GR_Sundanese'
    # gt_path = 'ChallengeB-TrackMixed_GT_Sundanese'
    # bin_path = 'ChallengeB-TrackMixed_Binarized_Sundanese' 
    # result_path = 'result_ChallengeB-TrackMixed_GR_Sundanese'
    # result_dat ='result_ChallengeB-TrackMixed_Dat_Sundanese'

    # basepath = 'ChallengeB-TrackMixed_GR_Balinese'
    # gt_path = 'ChallengeB-TrackMixed_GT_Balinese'
    # bin_path = 'ChallengeB-TrackMixed_Binarized_Balinese' 
    # result_path = 'result_ChallengeB-TrackMixed_GR_Balinese'
    # result_dat ='result_ChallengeB-TrackMixed_Dat_Balinese'

    # basepath = 'ChallengeB-TrackMixed_GR_efeo'
    # gt_path = 'ChallengeB-TrackMixed_GT_efeo'
    # bin_path = 'ChallengeB-TrackMixed_Binarized_efeo' 
    # result_path = 'result_ChallengeB-TrackMixed_GR_efeo'
    # result_dat ='result_ChallengeB-TrackMixed_Dat_efeo'

    # basepath = 'ChallengeB-TrackMixed_GR_Khmer'
    # gt_path = 'ChallengeB-TrackMixed_GT_Khmer'
    # bin_path = 'ChallengeB-TrackMixed_Binarized_Khmer' 
    # result_path = 'result_ChallengeB-TrackMixed_GR_Khmer'
    # result_dat ='result_ChallengeB-TrackMixed_Dat_Khmer'

    # basepath = 'PNRI_86_L620_GR'
    # gt_path = 'PNRI_86_L620_GT'
    # result_path = 'result_PNRI_86_L620_GR'
    # result_dat ='result_PNRI_86_L620_Dat'

    # basepath = 'Sundansese_61_GR'
    # gt_path = 'Sundansese_61_GT'
    # result_path = 'result_Sundansese_61_GR'
    # result_dat ='result_Sundansese_61_Dat'

    # basepath = 'Sundansese_12_GR'
    # gt_path = 'Sundansese_12_GT'
    # result_path = 'result_Sundansese_12_GR'
    # result_dat ='result_Sundansese_12_Dat'

    # basepath = 'datasets/ICDAR2013-HSC-Testing_GR'
    # gt_path = 'datasets/ICDAR2013-HSC-Testing_GR'
    # bin_path = 'result/Exp1-mixedPLM-Binarized' 
    # result_path = 'results/result_ICDAR2013-HSC-Testing_GR'
    # result_dat ='results/result_ICDAR2013-HSC-Testing_Dat'

    basepath = args.path
    # result_path = basepath.replace("datasets","result")
    result_path = basepath.replace("datasets","result")
    result_dat = result_path.replace("_GR","_Dat")
    pathlib.Path(result_path).mkdir(parents=True, exist_ok=True) 
    pathlib.Path(result_dat).mkdir(parents=True, exist_ok=True) 

    temp = 0
    # for filename in os.listdir(basepath): 
    temp +=1
    filename = args.fname
    print("File-",temp," : ",filename)
    folder=os.path.join(basepath, filename)
#     folder_bin=os.path.join(bin_path, filename)[:-4]+'.bmp'
#     folder_gt=os.path.join(gt_path, filename)
    
    if os.path.isfile(folder):
        if folder.endswith(('.jpg','.jpeg','.tif','.png')):        
            print("\n",filename)
            print("\n",folder)
            img = cv2.imread(folder,-1)     
            
            #denoising task
            # args = Args(folder, modelpath,window,nb_layers,step,nb_filters,kernel,dropout,stride,every,threshold,outFilename) #v17
            args.imgpath = folder
            if args.denoised :
                denoised_image = step1.denoising(args)
                img_bool = denoised_image <  0.55
                # denoised_image = step1.remove_big_object_height(img_bool)
                denoised_image = step1.remove_big_object(img_bool)
                # denoised_image = step1.remove_tall_object(img_bool,img.shape[0])
            else:
                denoised_image = cv2.imread(folder,0)
                denoised_image = cv2.GaussianBlur(denoised_image, (5, 5), 0)                    
                denoised_image = 255 - sobel(denoised_image)*10000;
                # denoised_image = 255 - sobel(denoised_image)*10000;
                            
            cv2.imwrite(os.path.join(result_path, filename[:-4] + 'denoised_img.bmp'), denoised_image)
            
            img_medial_seams, sep_seams, img_sep_seams, img_final = SeamCarving.extract_text_lines(img,denoised_image, args.smooth, args.s, args.sigma, args.off, args.denoised)
            #check whether the compute_medial_seams is succeed or not
#             if len(sep_seams)==0 or sep_seams.size == 0 :
#                 print("compute_medial_seams is failed")
#             else:
                
            util.save_data(sep_seams,os.path.join(result_path, filename[:-4] + 'seams.csv'))
            cv2.imwrite(os.path.join(result_path, filename[:-4] + 'seams.bmp'), img_sep_seams)
            cv2.imwrite(os.path.join(result_path, filename[:-4] + 'final.bmp'), img_final)
            if args.same:
                filename = os.path.basename(folder)
                folder_gt = os.path.dirname(folder)
            else:
                filename = os.path.basename(folder).split(".")[0]+".bmp"
                folder_gt = os.path.dirname(folder).replace("GR","GT")
            
#                 print(folder_gt, filename)
#             if not same:                
#                 filename = filename.split(".")[0] + '.bmp'
            
#             segment_line_from_image(gt_path, filename, sep_seams, result_dat)
            util.segment_line_from_image(folder_gt, filename, sep_seams, result_dat)
                

    print("succeed")

from functools import wraps
import time


def timeit(func):
    @wraps(func)
    def timeit_wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        total_time = end_time - start_time
        # first item in the args, ie `args[0]` is `self`
        print(f'Function {func.__name__}{args} {kwargs} Took {total_time:.4f} seconds')
        with open("time_calculation_single.txt", "a+") as myfile:            
            myfile.write("\n")
            myfile.write(f'Function {func.__name__}{args} {kwargs} Took {total_time:.4f} seconds')
            
        return result
    return timeit_wrapper

class TimeCalculator:
    @timeit
    def calculate_base_on_slice(self, args, slice):
        """
        an example function that returns sum of all numbers up to the square of num
        """
        args.s = slice
        main(args)
        

    def __repr__(self):        
        return f'calc_object:{id(self)}'

if __name__ == "__main__":
    args = parse_menu()

    #without time computation
    # main(args) # default

    # with time computation
    calc = TimeCalculator()    
    # calc.calculate_base_on_slice(args,3)
    # calc.calculate_base_on_slice(args,4)
    # calc.calculate_base_on_slice(args,5)
    calc.calculate_base_on_slice(args,6)
    # calc.calculate_base_on_slice(args,7)
    # calc.calculate_base_on_slice(args,8)
    

