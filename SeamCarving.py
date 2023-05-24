from skimage.color import rgb2gray
from skimage.filters import sobel,scharr,prewitt
import numpy as np
import matplotlib.pyplot as plt
from skimage import feature
import cv2
import os
import skimage.filters
from skimage.morphology import disk
import util

def extract_text_lines(img,img_b, smooth, s, sigma, off, denoised): #v13
    img= util.util_alpha_channel(img,'g')
    if len(img.shape) > 2:                                
        img_gray = rgb2gray(img);        
    else:
        img_gray = img;
    
    cv2.imwrite('img_gray.bmp', img_gray)
    if denoised:
        # with denoising
        local_max, medial_seams_inds, L, img_medial_seams= compute_medial_seams(img, img_b, smooth, s, off) # apply Sobel edge detection to deal with the object thickness variation from picture and text , for case IIIB-12-306-P1.jpg
        print("denoising True")
    else :
        # no denoising
        local_max, medial_seams_inds, L, img_medial_seams= compute_medial_seams(img, img_b, smooth, s, off) 
        print("denoising false")

#     sep_seams, img_sep_seams = compute_separating_seams(img, img_gray,img_edge, L, sigma, off);
    sep_seams, img_sep_seams = compute_separating_seams(img, img_gray, L, sigma, off);
    img_final=overlay_medial_seams(img_sep_seams, medial_seams_inds, off)

    return img_medial_seams, sep_seams, img_sep_seams, img_final

import numpy as np
def _get_backward_seam(energy_map: np.ndarray,img,L) -> np.ndarray:
    """Compute the minimum vertical seam from the backward energy map"""
    assert energy_map.size > 0 and energy_map.ndim == 2
 
    n, m = energy_map.shape[:2]
    
    # number of text lines in the page
    l = len(L)   
    # print("number of text lines:",l)

    # copy the energy map for the dynamic programming part 
    new_energy = np.copy(energy_map);
    
    # initialize seam coordinates    
    sep_seams = np.zeros((n,l-1));

    ## apply constrained seam carving for each pair of text lines
    for i in range(l-1) :
        # upper and lower medial seams
        # print(i)
        L_a = L[i];
        L_b = L[i+1];
        # length of the medial seam (maybe they do not extend through the whole page width)
        l_a = len(L_a);
        l_b = len(L_b);
        
        min_l = min(l_a,l_b);   # minimum of the two text line lengths
        
        ## --Succeed --
        ## compute minimum energy separating seam using dynamic programming
        for row in range(1,min_l) :            
            for col in range(L_a[row],L_b[row]+1):
                # find previous row's neighbors for the cumulative matrix computation and take care not to overstep the boundaries
                left = max(col-1,L_a[row-1]);
                right = min(col+1,L_b[row-1]);

                # find the minimum value of the previous row's neighbors 
                if len(new_energy[row-1,left:right+1])==0:                    
                    # many pixels difference, discontinuity
                    if (col > left-1) :
                        new_energy[row,col] = new_energy[row-1,right-1];
                    elif (col < right-1) :
                        new_energy[row,col] = new_energy[row-1,left-1];                    
                else :    # one pixel difference, no discontinuity
                    minpath = np.min(new_energy[row-1,left:right+1])
                    new_energy[row,col] = energy_map[row,col] + minpath;
                
        
        min_index = np.argmin(new_energy[min_l-1,L_a[min_l-1]:L_b[min_l-1]])            
       
        min_index = L_a[min_l-1] + min_index; # correct index in the entire image
        
        # backtrack through the energy map from bottom to top to determine the minimum energy seam
        for row in range(min_l-2,0,-1) :
            # the column of the minimum energy seam            
            j = min_index;
            # update the seam map
            sep_seams[row+1,i] = j;

            # find next row's neighbors inside two textlines or between L_a[row] and L_b[row]
            left = max(j-1,L_a[row]);
            right = min(j+1,L_b[row]);
                       

            if len(new_energy[row,left:right+1])==0:
                if (j > left -1 ) :
                    min_index = right;
                elif (j < right -1) :
                    min_index = left;
                
            else :    # one pixel difference, no discontinuity  
                min_index = np.argmin(new_energy[row,left:right+1])
                min_index = min_index + left ;   # correct index in the entire image
            


        # first position of the optimal seam        
        sep_seams[0,i] = min_index;

        # reset the energy values
        new_energy = np.copy(energy_map); 
    
    return energy_map, sep_seams.astype(int)
# ----------------------------------------------------------------

def _get_backward_energy(gray: np.ndarray) -> np.ndarray:
    """Get backward energy map from the source image"""
    assert gray.ndim == 2

    gray = gray.astype(np.float32)
    grad_x = sobel(gray, axis=1)
    grad_y = sobel(gray, axis=0)
    energy = np.abs(grad_x) + np.abs(grad_y)
#     energy = sobel(gray)
#     energy = prewitt(gray)
#     energy = np.sqrt((grad_x ** 2) + (grad_y ** 2))
#     energy = feature.canny(gray, sigma=1)
    return energy
# ----------------------------------------------------------------

import numpy as np
import math
from scipy.signal import find_peaks
from skimage.measure import label, regionprops

def compute_medial_seams(img, img_modif, smooth, s, off):
    # medial seam computation with projection profile matching
    print('Calculating the estimated text line....');
    # parameter initialization 
    n,m = img_modif.shape[:2];
    print("n,m:",n,m)

    # binary image that will contain the piece-wise text line approximations
    img_bin = np.zeros(n*m);    
   
    denoised_img = (img_modif < 128) * 1
    
    #unskewness process
#     skew_angle = float(skew_angle_hough_transform(img_edge))
#     print(skew_angle)
#     img_edge = rotate(img_edge, skew_angle, cval=1)
#     cv2.imwrite('imgOut_0_unskew.png', img_edge*10000)
    
    # compute horizontal projection profiles of all edge image slices 
    # and find their local maxima
    local_max = [];   # locations of local maxima
    
#   apply an adaptive slicing by checking the  length average of text, anticipate for rare text in a page
    horiz_proj = horizontal_projections(denoised_img);
    
    if np.average(horiz_proj) < m/50 :
        s = 1
        # print("change slice = 1")
    
    # width of each image slice
    w = math.floor(m / s);
 
    k = 0;  # help counter for the slices
    for i in range(s):     # for each slice
        # compute histogram of sum of edges
        horiz_proj = horizontal_projections(denoised_img[:,k:k+w-1]);

        # single smooth it with cubic splines
        # horiz_proj_smooth = get_smooth(horiz_proj,smooth)
        
        # triple smooth it with cubic splines
        horiz_proj_smooth = get_smooth(horiz_proj,smooth)
        horiz_proj_smooth = get_smooth(horiz_proj_smooth,int(smooth/2))
        horiz_proj_smooth = get_smooth(horiz_proj_smooth,int(smooth/4))
        
        # find peaks of the profile
        #height = (np.max(horiz_proj_smooth)-np.min(horiz_proj_smooth))/2 # THIS IS not MIDRANGE
        
        # height1 = (np.max(horiz_proj_smooth)+np.min(horiz_proj_smooth))/2 # THIS IS MIDRANGE
        # height2 = np.average(horiz_proj_smooth)/2  # for case Balinese/MB-TaruPramana-P32.jpg or IIIB-12-306-P1.jpg
        # min_shpp= np.min(horiz_proj_smooth)
        # if height1-min_shpp < height2-min_shpp:
        #     height = height1
        #     print("height1")
        # else:
        #     height = height2
        #     print("height2")
        height = np.average(horiz_proj_smooth)/2
        peaks, _ = find_peaks(horiz_proj_smooth, height)
        # print("peaks: ",peaks)
        local_max.append(peaks)

        # next slice
        k = k + w;
        
    # half of slice width 
    k = int(np.floor(w/2))    # the medial colomn of sub image. It is used if s > 2 
    medial_seams_inds = []
    
    # if s==1 then  local maxima is the row of the medial seam
    if s == 1:
        for j in range(len(local_max[0])):  # for each left local maximum
            points = np.ones(m, dtype=int)*local_max[0][j]
            # linear indices on the entire image                        
            arr = np.array([points,np.arange(0, m, dtype=int)])
            inds=np.ravel_multi_index(arr, (n,m))
            medial_seams_inds = np.append(medial_seams_inds,inds)
            # update the binary image with the text line approximation
            img_bin[inds] = 1;
    elif s == 2:
        # s == 2
        ## match local maxima of the projection profiles between two consecutive image slices        
        for i in range(s-1):        
            # if there is no local max, continue to the next pair
            if len(local_max[i])==0: 
                continue;
            # matches from left to right and from right to left
            matches_lr = np.zeros(len(local_max[i]));
            matches_rl = np.zeros(len(local_max[i+1]));

            # find matches from left slice to right slice
            for j in range(len(local_max[i])):  # for each left local maximum
                # compute distances from the right slice and sort them
                dists = np.abs(local_max[i][j] - local_max[i+1]);            
                ind_sort = np.argsort(dists);
                # the match is the right maximum that is closest to the left one
                matches_lr[j] = ind_sort[0];            

            # find matches from right slice to left slice
            for j in range(len(local_max[i+1])):    # for each right local maximum
                # compute distances from the left slice and sort them
                dists = np.abs(local_max[i+1][j] - local_max[i]);
                ind_sort = np.argsort(dists);

                # the match is the left maximum that is closest to the right one
                matches_rl[j] = ind_sort[0];
                
            # match profile maxima that agree from both sides
            # (left to right and right to left)
            for j in range(len(local_max[i])):
                for o in range(len(local_max[i+1])):
                    # maximum match exists
                    if (matches_lr[j] == o and matches_rl[o] == j):
                        # calculate rounded row coordinates of lines between matched maxima
                        
                        # rounded row coordinates for the first pair of slices                        
                        points = np.round(np.linspace(local_max[i][j],local_max[i+1][o],num=int(m))).astype(int)

                        # linear indices on the entire image                        
                        arr = np.array([points,np.arange(0, m, dtype=int)])
                        inds=np.ravel_multi_index(arr, (n,m))
                        medial_seams_inds = np.append(medial_seams_inds,inds)

                        # update the binary image with the text line approximation
                        img_bin[inds] = 1;
            
    else:
        # s > 2
        ## match local maxima of the projection profiles 
        ## between two consecutive image slices
        for i in range(s-1):        
            # if there is no local max, continue to the next pair            
            if len(local_max[i])==0 or len(local_max[i+1])==0 : 
                continue;

            # matches from left to right and from right to left
            matches_lr = np.zeros(len(local_max[i]));
            matches_rl = np.zeros(len(local_max[i+1]));

            # find matches from left slice to right slice
            for j in range(len(local_max[i])):  # for each left local maximum
                # compute distances from the right slice and sort them
                dists = np.abs(local_max[i][j] - local_max[i+1]);            
                ind_sort = np.argsort(dists);

                # the match is the right maximum that is closest to the left one
                matches_lr[j] = ind_sort[0];  
                
            # find matches from right slice to left slice
            for j in range(len(local_max[i+1])):    # for each right local maximum
                # compute distances from the left slice and sort them
                dists = np.abs(local_max[i+1][j] - local_max[i]);
                ind_sort = np.argsort(dists);

                # the match is the left maximum that is closest to the right one
                matches_rl[j] = ind_sort[0];

            for j in range(len(local_max[i])):
                for o in range(len(local_max[i+1])):
                    # maximum match exists
                    if (matches_lr[j] == o and matches_rl[o] == j):
                        # calculate rounded row coordinates of lines between matched maxima
                        
                        if (i == 0) : # start the first histogram profile from the beginning of the image
                            # rounded row coordinates for the first pair of slices                        
                            points = np.round(np.linspace(local_max[i][j],local_max[i+1][o],num=int(k))).astype(int)

                            # linear indices on the entire image                        
                            arr = np.array([points,np.arange(0, k, dtype=int)])
                            inds=np.ravel_multi_index(arr, (n,m))
                            medial_seams_inds = np.append(medial_seams_inds,inds)
                        elif (i == s-2) :# end the last histogram profile in the end of the image
                            #lenght = len(k:m);
                            length =m-k+1

                            # rounded row coordinates for the last pair of slices
                            points = np.round(np.linspace(local_max[i][j],local_max[i+1][o],num=length)).astype(int);

                            # linear indices on the entire image                            
                            arr = np.array([points,np.arange(k-1, m, dtype=int)])
                            inds=np.ravel_multi_index(arr, (n,m))

                            medial_seams_inds = np.append(medial_seams_inds,inds)
                        else :    # intermediate pairs of slices
                            # rounded row coordinates for an intermediate pair of slices

                            points = np.round(np.linspace(local_max[i][j],local_max[i+1][o],num=w)).astype(int)

                            # linear indices on the entire image
                            arr = np.array([points,np.arange(k-1, k+w-1, dtype=int)])                        

                            inds=np.ravel_multi_index(arr, (n,m))
                            medial_seams_inds = np.append(medial_seams_inds,inds)

                        # update the binary image with the text line approximation
                        img_bin[inds] = 1;

            # update the help counter 
            if (i > 0 and i < s-2) :
                k = k + w;
    # reshape the binary image of medial seams
    img_bin = img_bin.reshape(n,m);
    cv2.imwrite('result/imgOut_2_proj_profile.png', img_bin*255)
    # cv2.imwrite('result/imgOut_2_proj_profile_part.png', img_bin[:,:2000]*255)
    
    
    
    # unique linear indices
    medial_seams_inds = np.unique(medial_seams_inds);
    medial_seams_inds=medial_seams_inds.astype(np.int64)
#     cv2.imwrite('result/imgOut_0_part.png', overlay_medial_seams(img, medial_seams_inds, off))

    img_bin,medial_seams_inds = postProc1(img_bin,medial_seams_inds)
    img_bin,medial_seams_inds = postProc2(img_bin,medial_seams_inds,s)
    img_bin,medial_seams_inds = postProc3(img_bin,medial_seams_inds)
    img_bin,medial_seams_inds = postProc4(img_bin,medial_seams_inds)
    img_bin,medial_seams_inds = postProc5(img_bin,medial_seams_inds)


    
    # connected component analysis of the final binary image containing  the medial seams
    label_im = label(img_bin)
    regions = regionprops(label_im)    
    
    # number of connected components
    num_cc = len(regions);
    L = []
    arr = np.zeros(m).astype(int)
    for c in range(num_cc):        
        # column coordinates of the c-th connected component
        inds_cc  = regions[c].coords #PixelIdxList
        
        if (len(inds_cc[:,0]) < 10) :  
            continue
        I=inds_cc[:,0]
        J=inds_cc[:,1]
        
        for jj in range(len(J)):            
            arr[J[jj]]=I[jj]
        
        L.append(arr.tolist())
            
    # overlay the medial seams on the original image
    img_medial_seams = overlay_medial_seams(img, medial_seams_inds, off);

    return local_max, medial_seams_inds, L, img_medial_seams

def compute_separating_seams(img, img_gray, L, sigma, off):
    # blur the image with gaussian filter    
    img_blur = skimage.filters.gaussian(img_gray, sigma=(sigma, sigma), truncate=3.5, channel_axis=-1)
    cv2.imwrite('result/imgOut_8_img_blur.png', img_blur*100)
    img_sep_seams = np.copy(img)
    
    # compute the energy map    
    energy_map = _get_backward_energy(img_blur)

    cv2.imwrite('result/imgOut_8_energyMap.png', energy_map*1000)
    
    # transpose the energy map, because we compute vertical seams    
    energy_map = np.transpose(energy_map)        
    
    util.save_data(energy_map,"result/energy_map.csv")
    new_energy_map,  sep_seams = _get_backward_seam(energy_map,img,L)
    
    util.save_data(sep_seams,"result/sep_seams.csv")
    util.save_data(new_energy_map,"result/new_energy_map.csv")
    
    img_sep_seams = overlay_separating_seams(img_sep_seams, sep_seams.astype(int), off);
    
    return sep_seams, img_sep_seams





import numpy as np
def overlay_medial_seams(img, medial_seams_inds, off):
    ##  function that overlays the medial_seams on the original image. It considers whether the image is in color or grayscale.
    ##  Input:
    ##      img: original image, either grayscale or color
    ##      medial_seams_inds: linear indices of the medial seams
    ##      off: the thickness of seam
    ##  Output:
    ##      img_medial_seams: original image overlaid with the text line approximations
    ##  Parameter tuning:
    ##  - "off" corresponds to seam thickness when the separating seams are presented in a figure.
    ##  This code is adopted from Nikolaos Arvanitopoulos (nick.arvanitopoulos@epfl.ch), 2014
    ##
    n=img.shape[0]
    m=img.shape[1]
    
    # initialize text lines image
    img_medial_seams = np.copy(img)

    # compute (x,y) coordinates of the linear text line approximation indices    
    [I,J]=np.unravel_index(medial_seams_inds, (n,m))

    # grayscale image
    if (len(img.shape)==2):
        for ii in range(len(I)):
            # use black color
            if (I[ii] <= off ) :  # row coordinates very near the first image row
                img_medial_seams[I[ii]:I[ii]+off,J[ii]] = 0;
            elif (I[ii] > n-off) :  # row coordinates very near the last image row
                img_medial_seams[I[ii]-off:I[ii],J[ii]] = 0;
            else  :   # intermediate row coordinates
                img_medial_seams[I[ii]-off:I[ii]+off,J[ii]] = 0;
            
    else :   # color image
        for ii in range(len(I)):
            # use blue color
            if (I[ii] <= off ) :  # row coordinates very near the first image row
                img_medial_seams[I[ii]:I[ii]+off,J[ii],0] = 0;
                img_medial_seams[I[ii]:I[ii]+off,J[ii],1] = 0;
                img_medial_seams[I[ii]:I[ii]+off,J[ii],2] = 255;
            elif (I[ii] > n-off) :  # row coordinates very near the last image row
                img_medial_seams[I[ii]-off:I[ii],J[ii],0] = 0;
                img_medial_seams[I[ii]-off:I[ii],J[ii],1] = 0;
                img_medial_seams[I[ii]-off:I[ii],J[ii],2] = 255;
            else  :  # intermediate row coordinates
                img_medial_seams[I[ii]-off:I[ii]+off,J[ii],0] = 0;
                img_medial_seams[I[ii]-off:I[ii]+off,J[ii],1] = 0;
                img_medial_seams[I[ii]-off:I[ii]+off,J[ii],2] = 255;
            
    return img_medial_seams



def overlay_separating_seams(img, sep_seams, off) :
    ##  function that overlays estimated seperator seams on the original image. It considers whether the image is in color or grayscale.
    ##
    ##  Input:
    ##      img: original image (grayscale or color)
    ##      sep_seams: seam coordinates, each column contains one seam
    ##      off: the thickness of seam
    ##
    ##  Output:
    ##      img_sep_seams: original image overlaid with the computed seams
    ##
    ##  Parameter tuning:
    ##      - "off" corresponds to seam thickness when the separating seams are presented in a figure.

    ##  This code is adopted from Nikolaos Arvanitopoulos (nick.arvanitopoulos@epfl.ch), 2014
    ##
    # initialize seam image
    img_sep_seams = np.copy(img);    

    # number of seams    
    le,l = sep_seams.shape[:2]    
    
    # grayscale image
    if (len(img.shape)==2):
        # for each seam
        for j in range(l) :
            if (np.sum(sep_seams[:,j]) > 0):
                for i in range(off,le-off) :
                    # use black color for the seam
                    if (sep_seams[i,j] > 0) :
                        img_sep_seams[sep_seams[i,j],i-off:i+off] = 0; 
                    
    else :    # color image
        # for each seam
        for j in range(l) :     
            if (np.sum(sep_seams[:,j]) > 0) :
                for i in range(off,le-off) :
                    # use red color for the seam
                    if (sep_seams[i,j] > 0) :                        
                        img_sep_seams[sep_seams[i,j],i-off:i+off,0] = 255; 
                        img_sep_seams[sep_seams[i,j],i-off:i+off,1] = 0;
                        img_sep_seams[sep_seams[i,j],i-off:i+off,2] = 0;
    return img_sep_seams

def horizontal_projections(image):
    return np.sum(image, axis=1)
# ----------------------------------------------------------------------------
def get_smooth(y, box_pts):
    box = np.ones(box_pts)/box_pts
    y_smooth = np.convolve(y, box, mode='same')
    return y_smooth


## Region is sorted by the minumum of row
def postProc1a(img_bin,medial_seams_inds):    
    # post-processing procedure to eliminate lines that begin in an intermediate column of the image connected component analysis of the binary image having piece-wise medial seams
    
    # -------------------------
    # Post-processing 1 : 
    # pre-requariment
    # Merge two medial seams which has a squence coloum but the row is not connected. 
    # The requirements is one medial seam named left seam is connected the the first column and 
    # the other one named right seam is touched the end of column. This post-processing will delete the right edge of left seam as long as the row difference between left seam and right seam.
    n,m = img_bin.shape[:2];
    label_im = label(img_bin)
    regions = regionprops(label_im)    
    
    # number of connected components
    num_cc = len(regions);
    # print("len(regions)",len(regions))
    skip = False
    img_bin = img_bin.reshape(n*m);
    agregat = []
    avgI = []
    for c in range(num_cc):
        inds_cc  = regions[c].coords #PixelIdxList
        I=inds_cc[:,0] # row
        J=inds_cc[:,1]
        #print("debug")
        agregat.append([np.min(I),np.max(I),np.min(J),np.max(J)])
        avgI.append(np.average(I))
#         print(",np.min(I),np.max(I),np.min(J),np.max(J)",np.min(I),np.max(I),np.min(J),np.max(J))
#     print("avgI",avgI)
    diffs = np.diff(avgI)
    if len(diffs) > 0:
        avgDiff = sum(diffs)/len(diffs) 
#     print("avgDiff",avgDiff)
    
    for i in range(len(agregat)):
        
        distance_row = 99999
        left = 99999
        right = 99999
        if (agregat[i][2] == 0 and agregat[i][3] != m-1) or (agregat[i][2] != 0 and agregat[i][3] == m-1):
            for j in range(len(agregat)):                
                if np.abs(agregat[i][3] - agregat[j][2]) <=1 :                    
                    # extract the left CC
                    inds_cc  = regions[i].coords #PixelIdxList
                    I=inds_cc[:,0] # row
                    J=inds_cc[:,1]
                    
                    # extract the right CC
                    next_cc  = regions[j].coords #PixelIdxList
                    Inext=next_cc[:,0]
                    Jnext=next_cc[:,1]
                    diff = np.abs(I[np.where(J == agregat[i][3])].item() - Inext[np.where(Jnext == agregat[j][2])].item())
                    if diff <= distance_row :
                        distance_row = diff 
                        left = i
                        right = j
                   
            if left != 99999 and right !=99999 :                
                # extract the left CC
                inds_cc1  = regions[left].coords #PixelIdxList
                I=inds_cc1[:,0] # row
                J=inds_cc1[:,1]

                # extract the right CC
                next_cc1  = regions[right].coords #PixelIdxList
                Inext=next_cc1[:,0]
                Jnext=next_cc1[:,1]
                
                start_col = np.max(J);
                end_col = np.min(Jnext);                

                start_row = I[np.where(J == start_col)].item()
                end_row = Inext[np.where(Jnext == end_col)].item()

                t = np.abs(end_row - start_row) 
                if t > 2 * avgDiff:
                    continue
                if start_col-t < 0:
                    start_row_reduce = I[np.where(J == 0)].item()
                else:
                    start_row_reduce = I[np.where(J == start_col-t)].item()
                
                if start_row_reduce != start_row :
                    t = np.abs(end_row - start_row_reduce) 
                    start_row = start_row_reduce

                if start_row < end_row :
                    row1 = start_row
                    row2 = end_row
                else :
                    row2 = start_row
                    row1 = end_row
                # remove the t-element of next medial seam and connet the current medial seam
                rows =[]
                cols =[]                
                
                row = list(range(row1,row2+1))   
                for temp in range(t):
                    rows = rows +row                

                col = list(range(start_col-t,start_col))
                for temp in range(row2-row1+1):
                    cols = cols + col
                
                
                arr_del = np.array([rows,cols],dtype=np.int32)

                ind_del = np.ravel_multi_index(arr_del, (n,m))

                img_bin[ind_del]=0;   # update the binary image
                f = np.isin(medial_seams_inds,ind_del);
                medial_seams_inds = np.delete(medial_seams_inds,f);  # remove the indices
                
                # rebuild the indices    
                
                diff = end_col + 1 - (start_col - t) 
#                 print("STEP 1a: start_row,end_row",start_row,end_row)
                if start_row < end_row :
                    inds_ext = np.floor(np.linspace(start_row-2,end_row+1,num =diff+2)).astype(int)  
                else:
                    inds_ext = np.floor(np.linspace(start_row+2,end_row+1,num =diff+2)).astype(int)  
                
                arr = np.array([inds_ext,np.arange(start_col-t-1, end_col+2, dtype=int)])  
                inds_ext=np.ravel_multi_index(arr, (n,m))
             
                # add the extended indices to the full indices array                
                medial_seams_inds = np.append(medial_seams_inds,inds_ext)
                 
        
    img_bin[medial_seams_inds] = 1;    
    img_bin = img_bin.reshape(n,m);
    
    cv2.imwrite('result/ImgOut_3_PostProc1_connectTheBrokenMedialSeam.png', img_bin*255)
    return(img_bin,medial_seams_inds)


## Region is sorted by the minumum of row
def postProc1b(img_bin,medial_seams_inds):    
    # post-processing procedure to eliminate lines that begin in an intermediate column of the image connected component analysis of the binary image having piece-wise medial seams
    
    # -------------------------
    # Post-processing 1 : 
    # pre-requariment
    # Merge two medial seams which has a squence coloum but the row is not connected. 
    # The requirements is one medial seam named left seam is connected the the first column and 
    # the other one named right seam is touched the end of column. This post-processing will delete the right edge of left seam as long as the row difference between left seam and right seam.
    n,m = img_bin.shape[:2];
    label_im = label(img_bin)
    regions = regionprops(label_im)    
    print("-----------1b---------");
    # number of connected components
    num_cc = len(regions);
    # print("len(regions)",len(regions))
    skip = False
    img_bin = img_bin.reshape(n*m);
    agregat = []
    avgI = []
    for c in range(num_cc):
        inds_cc  = regions[c].coords #PixelIdxList
        I=inds_cc[:,0] # row
        J=inds_cc[:,1]
        #print("debug")
        if len(J) >= int(m/s):
            agregat.append([np.min(I),np.max(I),np.min(J),np.max(J)])
        avgI.append(np.average(I))
        print("np.min(I),np.max(I),np.min(J),np.max(J),len(J)",np.min(I),np.max(I),np.min(J),np.max(J),len(J))
#     print("avgI",avgI)
    diffs = np.diff(avgI)
    avgDiff = sum(diffs)/len(diffs)
#     print("avgDiff",avgDiff)
    
    for i in range(len(agregat)):
        
        distance_row = 99999
        left = 99999
        right = 99999
        if (agregat[i][2] != 0 and agregat[i][3] != m-1):
            for j in range(len(agregat)):                
                if np.abs(agregat[i][3] - agregat[j][2]) <=1 :                    
                    # extract the left CC
                    inds_cc  = regions[i].coords #PixelIdxList
                    I=inds_cc[:,0] # row
                    J=inds_cc[:,1]
                    
                    # extract the right CC
                    next_cc  = regions[j].coords #PixelIdxList
                    Inext=next_cc[:,0]
                    Jnext=next_cc[:,1]
                    print("np.min(I),np.max(I),np.min(J),np.max(J),len(J)",np.min(I),np.max(I),np.min(J),np.max(J),len(J))
                    print("agregat[i][3],agregat[j][2]",agregat[i][3],agregat[j][2])
                    print("I[np.where(J == agregat[i][3])].item() , Inext[np.where(Jnext == agregat[j][2])].item()",I[np.where(J == agregat[i][3])].item() , Inext[np.where(Jnext == agregat[j][2])].item())
                    diff = np.abs(I[np.where(J == agregat[i][3])].item() - Inext[np.where(Jnext == agregat[j][2])].item())
                    
                    if diff <= distance_row :
                        distance_row = diff 
                        left = i
                        right = j
                   
            if left != 99999 and right !=99999 :                
                # extract the left CC
                inds_cc1  = regions[left].coords #PixelIdxList
                I=inds_cc1[:,0] # row
                J=inds_cc1[:,1]

                # extract the right CC
                next_cc1  = regions[right].coords #PixelIdxList
                Inext=next_cc1[:,0]
                Jnext=next_cc1[:,1]
                if len(Jnext) >= int(m/s):
                    continue
                print("np.min(I),np.max(I),np.min(J),np.max(J)",np.min(I),np.max(I),np.min(J),np.max(J))
                print("np.min(Inext),np.max(Inext),np.min(Jnext),np.max(Jnext)",np.min(Inext),np.max(Inext),np.min(Jnext),np.max(Jnext))
                start_col = np.max(J);
                end_col = np.min(Jnext);                

                start_row = I[np.where(J == start_col)].item()
                end_row = Inext[np.where(Jnext == end_col)].item()

                t = np.abs(end_row - start_row) 
                
                if t > 2 * avgDiff:
                    continue
                if start_col-t <= 0:
                    start_row_reduce = I[np.where(J == 0)].item()
                else:
                    start_row_reduce = I[np.where(J == start_col-t)].item()
                print("t_awal",t)
                print("start_row,end_row,start_row_reduce",start_row,end_row,start_row_reduce)
                if start_row_reduce != start_row :
                    t = np.abs(end_row - start_row_reduce) 
                    start_row = start_row_reduce
                print("t",t)
                print("start_row,end_row,start_row_reduce",start_row,end_row,start_row_reduce)
                if start_row < end_row :
                    row1 = start_row
                    row2 = end_row
                else :
                    row2 = start_row
                    row1 = end_row
                # remove the t-element of next medial seam and connet the current medial seam
                rows =[]
                cols =[]                
                
                row = list(range(row1,row2+1))   
                for temp in range(t):
                    rows = rows +row                

                col = list(range(start_col-t,start_col))
                for temp in range(row2-row1+1):
                    cols = cols + col
                
                
                arr_del = np.array([rows,cols],dtype=np.int32)
#                 print(arr_del)
                ind_del = np.ravel_multi_index(arr_del, (n,m))
    #             print("ind_del",ind_del)
                img_bin[ind_del]=0;   # update the binary image
                f = np.isin(medial_seams_inds,ind_del);
                medial_seams_inds = np.delete(medial_seams_inds,f);  # remove the indices
                
                # rebuild the indices    
                
                diff = end_col + 1 - (start_col - t) 
                print("STEP 1a: start_row,end_row",start_row,end_row)
                if start_row < end_row :
                    inds_ext = np.floor(np.linspace(start_row-2,end_row,num =diff+1)).astype(int)  
                else:
                    inds_ext = np.floor(np.linspace(start_row+2,end_row,num =diff+1)).astype(int)  
                
                arr = np.array([inds_ext,np.arange(start_col-t-1, end_col+1, dtype=int)])  
                inds_ext=np.ravel_multi_index(arr, (n,m))
             
                # add the extended indices to the full indices array                
                medial_seams_inds = np.append(medial_seams_inds,inds_ext)
                 
        
    img_bin[medial_seams_inds] = 1;    
    img_bin = img_bin.reshape(n,m);
    
    cv2.imwrite('result/ImgOut_3_PostProc1b_connectTheBrokenMedialSeam.png', img_bin*255)
    return(img_bin,medial_seams_inds)

def postProc2a(img_bin,medial_seams_inds):
    # -------------------------
    # Post-processing 2 : 
    # remove medial seams that do not start from the beginning of the image
    # Add condition, remove the medial seam if the end of medial seam is not touch the end of image colomn
    n,m = img_bin.shape[:2];
    label_im = label(img_bin)
    regions = regionprops(label_im)    
    
    # number of connected components
    num_cc = len(regions);
    img_bin = img_bin.reshape(n*m);
    for c in range(num_cc):        
        # column coordinates of the c-th connected component
        inds_cc  = regions[c].coords #PixelIdxList
        I=inds_cc[:,0]
        J=inds_cc[:,1]
        
        arr = np.array([I,J])         
        inds = np.ravel_multi_index(arr, (n, m))
        
        # if the connected component does not start from the beginning of the image, remove it from the computed indices        
        if (np.min(J) != 0 and np.max(J) < m-1 and len(J) <= int(m/s)) :
            img_bin[inds]=0;   # update the binary image
            f = np.isin(medial_seams_inds,inds);
            medial_seams_inds = np.delete(medial_seams_inds,f);  # remove the indices

    img_bin = img_bin.reshape(n,m);    
    cv2.imwrite('result/ImgOut_2b_PostProc2a_delIfnotBeginning.png', img_bin*255)
    return(img_bin,medial_seams_inds)

# post-processing step to remove lines that start from some intermediate column of the image
# connected component analysis of the binary image containing the piece-wise medial seams

def postProc1(img_bin,medial_seams_inds):    
    # -------------------------
    # Post-processing 1 : 
    # pre
    # Merge two medial seams which has a squence coloum but the row is not connected. 
    # The requirements is one medial seam named left seam is connected the the first column and 
    # the other one named right seam is touched the end of column. This post-processing will extend 
    # the left seam toward to the beginning of column with row equal to the average of the right seam.
    n,m = img_bin.shape[:2];
    label_im = label(img_bin)
    regions = regionprops(label_im)    

    # number of connected components
    num_cc = len(regions);
    # print("len(regions)",len(regions))
    skip = False
    img_bin = img_bin.reshape(n*m);
    for c in range(num_cc-1):    
        if skip :
            skip = False
            continue 

        # column coordinates of the c-th connected component
        inds_cc  = regions[c].coords #PixelIdxList
        I=inds_cc[:,0] # row
        J=inds_cc[:,1] #coloum
        IAvg = int(np.average(I))


        # extract the below (next) CC
        next_cc  = regions[c+1].coords #PixelIdxList
        Inext=next_cc[:,0]
        Jnext=next_cc[:,1]
        InextAvg = int(np.average(Inext))
    #         print("c",c)
    #         print("np.min(I),np.max(I),np.min(Inext),np.max(Inext),np.min(J),np.max(J),np.min(Jnext),np.max(Jnext)",np.min(I),np.max(I),np.min(Inext),np.max(Inext),np.min(J),np.max(J),np.min(Jnext),np.max(Jnext))
        arr = np.array([I,J])         
        inds = np.ravel_multi_index(arr, (n, m))

        arrNext = np.array([Inext,Jnext])         
        indsNext = np.ravel_multi_index(arrNext, (n, m))


        if np.min(J) != 0 and np.max(J) == m-1 : # to be continue
            # begin

            if np.min(J) > np.max(Jnext)  and np.min(Jnext) == 0:           
                
                # rebuild the indices                
                start_col = np.max(Jnext);
                end_col = np.min(J);                

                inds_ext = np.floor(np.linspace(Inext[np.where(Jnext == start_col)].item(),I[np.where(J == end_col)].item(),num = end_col - start_col)).astype(int)  #v15

                arr = np.array([inds_ext,np.arange(start_col, end_col, dtype=int)])  
    
                inds_ext=np.ravel_multi_index(arr, (n,m))

                # add the extended indices to the full indices array                
                medial_seams_inds = np.append(medial_seams_inds,inds_ext)
                skip = True
            elif np.min(J) == np.max(Jnext)  and np.min(Jnext) == 0:           
                start_col = np.max(Jnext);
                end_col = np.min(J);                

                start_row = Inext[np.where(Jnext == start_col)].item()
                end_row = I[np.where(J == end_col)].item()

                if start_row < end_row :
                    row1 = start_row
                    row2 = end_row
                else :
                    row2 = start_row
                    row1 = end_row

                t = np.abs(end_row - start_row)   

                

                # remove the t-element of next medial seam and connet the current medial seam
                rows =[]
                cols =[]                
                
                row = list(range(row1,row2+1))   
                
                for temp in range(t):
                    rows = rows +row                
                col = list(range(start_col-t+1,end_col+1))
                for temp in range(row2-row1+1):
                    cols = cols + col
                # print(len(rows),len(cols))
                arr_del = np.array([rows,cols],dtype=np.int32)
                ind_del = np.ravel_multi_index(arr_del, (n,m))
                
                img_bin[ind_del]=0;   # update the binary image
                f = np.isin(medial_seams_inds,ind_del);
                medial_seams_inds = np.delete(medial_seams_inds,f);  # remove the indices
                
                # rebuild the indices                

                inds_ext = np.floor(np.linspace(Inext[np.where(Jnext == start_col-t)].item(),end_row,num =t+1)).astype(int)  #v16

                arr = np.array([inds_ext,np.arange(start_col-t, end_col+1, dtype=int)])  
    
                inds_ext=np.ravel_multi_index(arr, (n,m))

                # add the extended indices to the full indices array                
                medial_seams_inds = np.append(medial_seams_inds,inds_ext)
                skip = True
        elif np.min(J) == 0 and  np.max(J) != m-1 : # to be continue
            if np.max(J) < np.min(Jnext)  and np.max(Jnext) == m-1:           
                    
                # rebuild the indices                
                start_col = np.max(J);
                end_col = np.min(Jnext)

                inds_ext = np.floor(np.linspace(I[np.where(J == start_col)].item(),Inext[np.where(Jnext == end_col)].item(),num = end_col - start_col)).astype(int)  #v15

                arr = np.array([inds_ext,np.arange(start_col, end_col, dtype=int)])  

                inds_ext=np.ravel_multi_index(arr, (n,m))

                # add the extended indices to the full indices array                
                medial_seams_inds = np.append(medial_seams_inds,inds_ext)
                skip = True
            elif np.max(J) == np.min(Jnext)  and np.max(Jnext) == m-1:    

                start_col = np.max(J);
                end_col = np.min(Jnext)

                start_row = I[np.where(J == start_col)].item()
                end_row = Inext[np.where(Jnext == end_col)].item()
                t = np.abs(end_row - start_row) 

                if start_row < end_row :
                    row1 = start_row -1 
                    row2 = end_row
                else :
                    row2 = start_row -1
                    row1 = end_row
                # print(row1,row2,start_col-t,end_col)
                # remove the t-element of next medial seam and connet the current medial seam
                rows =[]
                cols =[]                
                row = list(range(row1,row2+1))   
                # print("end_col - start_col-t",end_col - start_col-t)
                for temp in range(t):
                    rows = rows +row                
                col = list(range(start_col-t+1,end_col+1))
                for temp in range(row2-row1+1):
                    cols = cols + col

                arr_del = np.array([rows,cols],dtype=np.int)
                ind_del = np.ravel_multi_index(arr_del, (n,m))
                # print(ind_del)
                img_bin[ind_del]=0;   # update the binary image
                f = np.isin(medial_seams_inds,ind_del);
                medial_seams_inds = np.delete(medial_seams_inds,f);  # remove the indices
                
                # rebuild the indices 
                inds_ext = np.floor(np.linspace(I[np.where(J == start_col-t)].item(),end_row,num =t+2)).astype(int)  #v15
                print(I[np.where(J == start_col-t)].item(),end_row,t+1)
                arr = np.array([inds_ext,np.arange(start_col-t, end_col+2, dtype=int)])  

                inds_ext=np.ravel_multi_index(arr, (n,m))

                # add the extended indices to the full indices array                
                medial_seams_inds = np.append(medial_seams_inds,inds_ext)
                skip = True


    img_bin[medial_seams_inds] = 1;    
    img_bin = img_bin.reshape(n,m);
        
    cv2.imwrite('result/ImgOut_3_PostProc1_connectTheBrokenMedialSeam.png', img_bin*255)
    return(img_bin,medial_seams_inds)

def postProc2(img_bin,medial_seams_inds,s):
    # -------------------------
    # Post-processing 2 : 
    # remove medial seams that do not start from the beginning of the image
    # Add condition, remove the medial seam if the end of medial seam is not touch the end of image colomn
    n,m = img_bin.shape[:2];
    label_im = label(img_bin)
    regions = regionprops(label_im)    
    
    # number of connected components
    num_cc = len(regions);
#     print(len(regions))
    
    img_bin = img_bin.reshape(n*m);
    for c in range(num_cc):        
        # column coordinates of the c-th connected component
        inds_cc  = regions[c].coords #PixelIdxList
        I=inds_cc[:,0]
        J=inds_cc[:,1]
        
        arr = np.array([I,J])         
        inds = np.ravel_multi_index(arr, (n, m))
        
        # if the connected component does not start from the beginning of the 
        # image, remove it from the computed indices
        # we need to add a condition if slice is bigger than 4, because mostly text is wriiten not in the beginning of coloumn
        
        if (np.min(J) != 0 and np.max(J) < m-1 and len(J) < int(m/s)) :
            img_bin[inds]=0;   # update the binary image
            f = np.isin(medial_seams_inds,inds);
            medial_seams_inds = np.delete(medial_seams_inds,f);  # remove the indices

                
        
    img_bin = img_bin.reshape(n,m);  
    cv2.imwrite('result/ImgOut_4_PostProc2_delIfnotBeginning.png', img_bin*255)
    return(img_bin,medial_seams_inds)

def postProc3(img_bin,medial_seams_inds):        
    # -------------------------
    # post-processing 3 :
    # if the medial seams is not ends in the last column of the  image, then extend it towards the end
    n,m = img_bin.shape[:2];
    label_im = label(img_bin)
    regions = regionprops(label_im)    
    
    # number of connected components
    num_cc = len(regions);
#     print("proc2",num_cc)
    for c in range(num_cc):   
#         print("c:",c)
        inds_cc  = regions[c].coords #PixelIdxList
        I=inds_cc[:,0]
        J=inds_cc[:,1]

        
        # check if this connected component start from the first column of the  image, if not, extend it towards the beginning
        if (np.min(J) != 0) :
            # last column of the CC
            start_col = np.min(J);
            inds_ext = np.floor(np.linspace(I[-1],I[np.where(J == start_col)].item(),num=start_col)).astype(int) 
            arr = np.array([inds_ext,np.arange(0,start_col, dtype=int)])   

            inds_ext=np.ravel_multi_index(arr, (n,m))

            # add the extended indices to the full indices array                
            medial_seams_inds = np.append(medial_seams_inds,inds_ext)

        # check if this connected component ends in the last column of the  image, if not, extend it towards the end
        if (np.max(J) != m-1) :
            # last column of the CC
            end_col = np.max(J);
            
            if (c == 0) :# the top connected component
                
                # extract the lower (next) CC
                next_cc  = regions[c+1].coords #PixelIdxList
                Inext=next_cc[:,0]
                Jnext=next_cc[:,1]
                I_n=np.copy(Inext)
                
                inds_ext = np.floor(np.linspace(I[np.where(J == end_col)].item(),int(np.average(I)),num=m-end_col)).astype(int)  #v15
                
                [inter_n,in1,in2] = np.intersect1d(inds_ext,I_n,return_indices=True);
                if (len(inter_n)!=0) : # intersection with the lower CC 
                    inds_ext = np.floor(np.linspace(I[np.where(J == end_col)].item(),int(np.min(Inext)- 10) if np.min(Inext) > 10 else n-10 ,num=m-end_col)).astype(int)
                    
                
                arr = np.array([inds_ext,np.arange(end_col,m, dtype=int)])   
                
                inds_ext=np.ravel_multi_index(arr, (n,m))

                # add the extended indices to the full indices array                
                medial_seams_inds = np.append(medial_seams_inds,inds_ext)
                
            elif (c == num_cc-1)  :  # the bottom connected component
                # extract the upper (next) CC
                prev_cc  = regions[c-1].coords #PixelIdxList
                Iprev=prev_cc[:,0] # row
                Jprev=prev_cc[:,1] # column

                #prev_cc = CC.PixelIdxList{c-1};
                I_p=np.copy(Iprev)
                
                inds_ext = np.floor(np.linspace(I[np.where(J == end_col)].item(),int(np.average(I)),num=m-end_col)).astype(int)
                
                [inter_p,ip1,ip2] = np.intersect1d(inds_ext,I_p,return_indices=True);
                if (len(inter_p)!=0) : # intersection with the lower CC 
                    inds_ext = np.floor(np.linspace(I[np.where(J == end_col)].item(),int(np.max(Iprev)+ 10) if (n - np.max(Iprev)) > 10 else 5 ,num=m-end_col)).astype(int)
                    

                arr = np.array([inds_ext,np.arange(end_col,m, dtype=int)])                        
                inds_ext=np.ravel_multi_index(arr, (n,m))

                # add the extended indices to the full indices array 
                medial_seams_inds = np.append(medial_seams_inds,inds_ext)
                
            else :   # intermediate connected component
                # extract the lower (previous) CC               

                prev_cc  = regions[c-1].coords #PixelIdxList
                Iprev=prev_cc[:,0] # row
                Jprev=prev_cc[:,1] # column

                I_p=np.copy(Iprev)

                 # extract the lower (next) CC
                next_cc  = regions[c+1].coords #PixelIdxList
                Inext=next_cc[:,0]
                Jnext=next_cc[:,1]
                I_n=np.copy(Inext)
                

                # extend the CC in the middle between the row coordinates 
                # of the upper and lower CC's until an intersection is found

                middle = np.floor((np.max(I_p)+np.min(I_n))/2);  # middle end point
               

                # extend towards the end                
                inds_ext = np.floor(np.linspace(I[np.where(J == end_col)].item(),middle,num=m-end_col)).astype(int)
                
                arr = np.array([inds_ext,np.arange(end_col,m, dtype=int)])                        
                inds_ext=np.ravel_multi_index(arr, (n,m))

                # add the extended indices to the full indices array 
                medial_seams_inds = np.append(medial_seams_inds,inds_ext)
                
    # binary image with only the medial seams    
    img_bin = np.zeros(n*m);
    img_bin[medial_seams_inds] = 1;    
    img_bin = img_bin.reshape(n,m);
    #import cv2
    cv2.imwrite('result/imgOut_5_PostProc3_extendToEnd.png', img_bin*255)
    return(img_bin,medial_seams_inds)

def postProc4(img_bin,medial_seams_inds):
    # -------------------------
    # post-processing 4 
    # remove medial seams that end in top row or below row or the average is near the top or below row (the number of row is divided by 8 )
    n,m = img_bin.shape[:2];
    label_im = label(img_bin)
    regions = regionprops(label_im)    
    
    # number of connected components
    num_cc = len(regions);
    
    img_bin = img_bin.reshape(n*m);
    # margin_top_bottom = n//10 if n//10 < 100 else 100
    margin_top_bottom = n//10 if n < m else n//20
    for c in range(num_cc):        
        # column coordinates of the c-th connected component
        inds_cc  = regions[c].coords #PixelIdxList
        
        I=inds_cc[:,0]
        J=inds_cc[:,1]
        
        arr = np.array([I,J])         
        inds = np.ravel_multi_index(arr, (n, m))
        
        
        
        if (np.average(I) < margin_top_bottom or np.average(I) > n - margin_top_bottom ) : # depond on document and the number of line and text width for example A4 format 
            img_bin[inds]=0;   # update the binary image
            f = np.isin(medial_seams_inds,inds);
            medial_seams_inds = np.delete(medial_seams_inds,f);  # remove the indices
            
    img_bin = img_bin.reshape(n,m);
    #import cv2
    cv2.imwrite('result/imgOut_6_postproc4_removeLine_top_below.png', img_bin*255)
    return(img_bin,medial_seams_inds)

def postProc5(img_bin,medial_seams_inds):
    # -------------------------
    # post-processing 5 
    # remove medial seams that close between the above  or below of textline 
    # remove text line which has inconsistent spacing between medial seam
    n,m = img_bin.shape[:2];
    label_im = label(img_bin)
    regions = regionprops(label_im)    
    print(len(regions))
    # number of connected components
    num_cc = len(regions);
    
    img_bin = img_bin.reshape(n*m);
    c_avg = []
    c_del = []
    
    for c in range(num_cc):                
        inds_cc  = regions[c].coords #PixelIdxList
        c_avg.append(int(np.average(inds_cc[:,0])))
    
    first_diff = np.diff(c_avg)
    diff_mean = np.average(first_diff)*0.5# good for MB-TaruPramana-P34.jpg
    

 
    
#     find index of the medial seams which close each other
    indxDiff = np.where(first_diff < diff_mean)[0]
    second_diff = np.diff(indxDiff)
    idx_second_diff = np.where(second_diff < 2 )[0]

    
    if len(indxDiff) > 1 :
        c_del =[1 if x==0 else x+2 for x in idx_second_diff]
                
#         print(c_del)
        
    else :
        c_del = indxDiff    
        

    for c in c_del:        
        # column coordinates of the c-th connected component
#         print("c:",c)
        inds_cc  = regions[c].coords #PixelIdxList
        
        I=inds_cc[:,0]
        J=inds_cc[:,1] 
        
        arr = np.array([I,J])         
        inds = np.ravel_multi_index(arr, (n, m))
        
        img_bin[inds]=0;   # update the binary image
        f = np.isin(medial_seams_inds,inds);
        medial_seams_inds = np.delete(medial_seams_inds,f);  # remove the indices
            
    img_bin = img_bin.reshape(n,m);
    
    cv2.imwrite('result/imgOut_7_postproc5_removeLine_closeEachOther.png', img_bin*255)
    return(img_bin,medial_seams_inds)
    #------------------------

def postProc3a(img_bin,medial_seams_inds):        
    # -------------------------
    # post-processing 3 :
    # if the medial seams is not ends in the last column of the  image, then extend it towards the end
    n,m = img_bin.shape[:2];
    label_im = label(img_bin)
    regions = regionprops(label_im)    
    
    # number of connected components
    num_cc = len(regions);
    

    for c in range(num_cc):   
        inds_cc  = regions[c].coords #PixelIdxList
        I=inds_cc[:,0]
        J=inds_cc[:,1]
        
        [y2,y1,x2,x1]=[I[np.where(J == np.min(J))].item(),I[np.where(J == np.max(J))].item(),np.min(J),np.max(J)] 

        print("np.min(I),np.max(I),np.min(J),np.max(J)",np.min(I),np.max(I),np.min(J),np.max(J))
        # check if this connected component start from the first column of the  image, if not, extend it towards the beginning
        if (np.min(J) != 0) :            
            # last column of the CC            
            
            start_col = np.min(J);
            orientation = (y2-y1)/(x2-x1)
            print("orientation", orientation )
            print(y2,y1,x2,x1)
            new_start = int(y1-(x1-0)*orientation)
            if new_start <= 0 :
                new_start = 1
            if new_start >= n -1 :
                new_start = n-2
            
            
            # create linear indices between the first row and the row coordinates of the connected component                
#             inds_ext = np.floor(np.linspace(I[-1],I[np.where(J == start_col)].item(),num=start_col)).astype(int)
            
            inds_ext = np.floor(np.linspace(new_start,y2,num=start_col)).astype(int) #average I[half of len(I)]
            arr = np.array([inds_ext,np.arange(0,start_col, dtype=int)])   
            inds_ext=np.ravel_multi_index(arr, (n,m))

            # add the extended indices to the full indices array                
            medial_seams_inds = np.append(medial_seams_inds,inds_ext)
            
        
        # check if this connected component ends in the last column of the  image, if not, extend it towards the end
        if (np.max(J) != m-1) :
            # last column of the CC
            end_col = np.max(J);
            
            if (c == 0) :# the top connected component
                # extract the lower (next) CC
                next_cc  = regions[c+1].coords #PixelIdxList
                Inext=next_cc[:,0]
                Jnext=next_cc[:,1]
                I_n=np.copy(Inext)
                
                inds_ext = np.floor(np.linspace(I[np.where(J == end_col)].item(),int(np.average(I)),num=m-end_col)).astype(int)  #v15
                
                [inter_n,in1,in2] = np.intersect1d(inds_ext,I_n,return_indices=True);
                if (len(inter_n)!=0) : # intersection with the lower CC 
                    inds_ext = np.floor(np.linspace(I[np.where(J == end_col)].item(),int(np.min(Inext)- 10) if np.min(Inext) > 10 else n-5 ,num=m-end_col)).astype(int)
                
                arr = np.array([inds_ext,np.arange(end_col,m, dtype=int)])   
                
                inds_ext=np.ravel_multi_index(arr, (n,m))

                # add the extended indices to the full indices array                
                medial_seams_inds = np.append(medial_seams_inds,inds_ext)
                
            elif (c == num_cc-1)  :  # the bottom connected component
                # extract the upper (next) CC
                prev_cc  = regions[c-1].coords #PixelIdxList
                Iprev=prev_cc[:,0] # row
                Jprev=prev_cc[:,1] # column

                #prev_cc = CC.PixelIdxList{c-1};
                I_p=np.copy(Iprev)
                
#                 inds_ext = np.floor(np.linspace(I[np.where(J == end_col)].item(),int(np.average(I[int(len(I)/2):-1])),num=m-end_col)).astype(int)
                inds_ext = np.floor(np.linspace(I[np.where(J == end_col)].item(),int(np.average(I)),num=m-end_col)).astype(int)
                
                [inter_p,ip1,ip2] = np.intersect1d(inds_ext,I_p,return_indices=True);
               
                if (len(inter_p)!=0) : # intersection with the lower CC 
#                     print("n,np.max(Iprev), n-np.max(Iprev)",n,np.max(Iprev),n-np.max(Iprev))
                    print("np.max(Iprev), int(n-np.max(Iprev))/2",np.max(Iprev), int(n-np.max(Iprev))/2)
                    inds_ext = np.floor(np.linspace(I[np.where(J == end_col)].item(),int(np.max(Iprev)+ int(n-np.max(Iprev))/2)  ,num=m-end_col)).astype(int)
#                     inds_ext = np.floor(np.linspace(I[np.where(J == end_col)].item(),int(np.max(Iprev)+ 2*off) if (n - np.max(Iprev)) > 2*off else off ,num=m-end_col)).astype(int)
                
                arr = np.array([inds_ext,np.arange(end_col,m, dtype=int)])                        
                inds_ext=np.ravel_multi_index(arr, (n,m))

                # add the extended indices to the full indices array 
                medial_seams_inds = np.append(medial_seams_inds,inds_ext)
                
            else :   # intermediate connected component
                # extract the lower (previous) CC               

                prev_cc  = regions[c-1].coords #PixelIdxList
                Iprev=prev_cc[:,0] # row
                Jprev=prev_cc[:,1] # column

                I_p=np.copy(Iprev)

                 # extract the lower (next) CC
                next_cc  = regions[c+1].coords #PixelIdxList
                Inext=next_cc[:,0]
                Jnext=next_cc[:,1]
                I_n=np.copy(Inext)

                # extend the CC in the middle between the row coordinates of the upper and lower CC's until an intersection is found
                middle = np.floor((np.max(I_p)+np.min(I_n))/2);  # middle end point
               

                # extend towards the end                
                inds_ext = np.floor(np.linspace(I[np.where(J == end_col)].item(),middle,num=m-end_col)).astype(int)
                
                arr = np.array([inds_ext,np.arange(end_col,m, dtype=int)])                        
                inds_ext=np.ravel_multi_index(arr, (n,m))

                # add the extended indices to the full indices array 
                medial_seams_inds = np.append(medial_seams_inds,inds_ext)
                
    # binary image with only the medial seams    
    img_bin = np.zeros(n*m);
    img_bin[medial_seams_inds] = 1;    
    img_bin = img_bin.reshape(n,m);
    #import cv2
    cv2.imwrite('result/imgOut_5_PostProc3_extendToEnd.png', img_bin*255)
    return(img_bin,medial_seams_inds)