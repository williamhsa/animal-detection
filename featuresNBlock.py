from skimage.util import view_as_windows
import numpy as np
from sklearn import cluster
from skimage import data
from skimage.color import rgb2gray
from skimage.feature import local_binary_pattern
from skimage.feature import hog
from skimage.util import img_as_float
from skimage.filters import gabor_kernel
from scipy import ndimage as nd
from auxfunc import *
    
def extractImageFeatures(img_rgb, img_gt, kmeans, fparams):
     
    n_features, total_features = calculateFeatureSize(fparams)
    
    f_imgs = generateFeaturesImages(img_rgb, fparams)
    
    img_gt = fparams['problem'].image2class(img_gt)
    
    blockSize = fparams['pathConfig']['blockSize']
    
    rangeV = int( float(img_rgb.shape[0]) /  blockSize )
    rangeH = int( float(img_rgb.shape[1]) /  blockSize )
    
    blocks_features = np.zeros( (rangeV, rangeH, n_features), dtype = np.float)
    
    for i in range(rangeV-1, -1, -1):
        for j in range(rangeH):      
            initBV = i * blockSize
            initBH = j * blockSize
            endBV = initBV + blockSize
            endBH = initBH + blockSize 
            
            blocks_features[i,j] = extractBlockFeatures(initBV, endBV, initBH, endBH, f_imgs, n_features, fparams)
    
    pos_i_center = blocks_features.shape[0] - 1
    pos_j_center = blocks_features.shape[1] / 2
    bottom_block1 = blocks_features[pos_i_center ,  pos_j_center + 1]
    bottom_block2 = blocks_features[pos_i_center ,  pos_j_center - 1]
    
    features = []
    n_area = fparams['pathConfig']['n_area']
    
    for i in range(rangeV - n_area -1, n_area -1, -1):
        for j in range(n_area, rangeH - n_area): 
            
            initBV = i * blockSize
            initBH = j * blockSize
            endBV = initBV + blockSize
            endBH = initBH + blockSize
            
            classRates = fparams['problem'].pathClassRate(img_gt[initBV:endBV,initBH:endBH])
            
            if np.max(classRates) >= fparams['perc']:   #if this block have all of the same class
                
                # block features
                feat = blocks_features[i, j]  
                '''
                # normalized pos
                if fparams['pos']['use']:
                    feat = np.hstack( (feat, i / float(rangeV), j / float(rangeH)  ) ) 
                    
                #center bottom blocks
                if fparams['bottom']['use']:
                    feat = np.hstack( (feat, bottom_block1 - blocks_features[i, j], bottom_block2 - blocks_features[i, j]) ) 
                '''
                # stack neighbors
                for area in range(1, n_area+1):
                    feat = np.hstack( (feat, blocks_features[i-area, j], blocks_features[i-area, j+area], blocks_features[i, j+area], blocks_features[i+area, j+area], blocks_features[i+area, j], blocks_features[i+area, j-area], blocks_features[i, j-area], blocks_features[i-area, j-area]))
                    
                realClass = np.argmax(classRates)
                feat = np.hstack( (feat, realClass) )
                features.append(feat)

    
    return np.array(features)
    
    
def classifyTrainingImg(img, img_gt, fparams):
    
    img_gt = fparams['problem'].image2class(img_gt)
    
    blockSize = fparams['pathConfig']['blockSize']
    rangeV = int( float(img.shape[0]) /  blockSize )
    rangeH = int( float(img.shape[1]) /  blockSize )
    
    img_r = np.empty(img_gt.shape, dtype = int)
    img_r[:,:] = -1
    
    for i in range(rangeV -1, -1, -1):
        for j in range(rangeH): 
            
            initBV = i * blockSize
            initBH = j * blockSize
            endBV = initBV + blockSize
            endBH = initBH + blockSize
            
            classRates = fparams['problem'].pathClassRate(img_gt[initBV:endBV,initBH:endBH])
            
            if np.max(classRates) >= fparams['perc']:  #if this block have all of the same class
                realClass = np.argmax(classRates)
                img_r[initBV:endBV, initBH:endBH] = realClass
  
    return img_r
    
    
def classifyImg(model, scaler, img_rgb, kmeans, fparams):
    
    blockSize = fparams['pathConfig']['blockSize']
    n_area = fparams['pathConfig']['n_area']
    
    o_shape_i = img_rgb.shape[0]
    o_shape_j = img_rgb.shape[1]
    
    i_diff = img_rgb.shape[0] % blockSize
    j_diff = img_rgb.shape[1] % blockSize

    if i_diff != 0:
        i_diff = blockSize - i_diff
    
    if j_diff != 0:
        j_diff = blockSize - j_diff
              
    img_rgb = np.pad(img_rgb, (     (   (blockSize*n_area) + i_diff   ,  (blockSize*n_area)   )    ,   (   (blockSize*n_area)  + j_diff    ,   (blockSize*n_area)  ), (0, 0)  ), mode='reflect')
    
    assert (img_rgb.shape[0]%blockSize == 0), 'The padded img size should be divisible by the block size'
    assert (img_rgb.shape[1]%blockSize == 0), 'The padded img size should be divisible by the block size'
    
    n_features, total_features = calculateFeatureSize(fparams)
    
    f_imgs = generateFeaturesImages(img_rgb, fparams)
    
    rangeV = int( float(img_rgb.shape[0]) /  blockSize )
    rangeH = int( float(img_rgb.shape[1]) /  blockSize )
    
    blocks_features = np.zeros( (rangeV, rangeH, n_features), dtype = np.float)
    
    for i in range(rangeV-1, -1, -1):
        for j in range(rangeH):      
            initBV = i * blockSize
            initBH = j * blockSize
            endBV = initBV + blockSize
            endBH = initBH + blockSize 
            
            blocks_features[i,j] = extractBlockFeatures(initBV, endBV, initBH, endBH, f_imgs, n_features, fparams)
    
    pos_i_center = blocks_features.shape[0] - 1
    pos_j_center = blocks_features.shape[1] / 2
    bottom_block1 = blocks_features[pos_i_center ,  pos_j_center + 1]
    bottom_block2 = blocks_features[pos_i_center ,  pos_j_center - 1]
    
    features = []

    for i in range(rangeV - n_area -1, n_area - 1, -1):
        for j in range(n_area, rangeH - n_area): 
                
            feat = blocks_features[i, j]  # block features
            '''    
            if fparams['pos']['use']:
                feat = np.hstack( (feat, i / float(rangeV), j / float(rangeH)  ) ) # normalized pos
                    
            if fparams['bottom']['use']:
                feat = np.hstack( (feat, bottom_block1 - blocks_features[i, j], bottom_block2 - blocks_features[i, j]) ) #bottom blocks
            '''    
            # stack neighbors
            for area in range(1, n_area+1):
                feat = np.hstack( (feat, blocks_features[i-area, j], blocks_features[i-area, j+area], blocks_features[i, j+area], blocks_features[i+area, j+area], blocks_features[i+area, j], blocks_features[i+area, j-area], blocks_features[i, j-area], blocks_features[i-area, j-area]))
                    
            features.append(feat)

    features = np.array(features).astype(None)
    features = scaler.transform(features) 
    pr = np.array(model.predict(features))
    
    img_r = np.empty((img_rgb.shape[0], img_rgb.shape[1]), dtype = np.uint8)
    
    #pos = 0
    
    #print results on imgs.
    for i in range(rangeV - n_area -1, n_area - 1, -1):
        for j in range(n_area, rangeH - n_area): 
            
            initBV = i * blockSize
            initBH = j * blockSize
            endBV = initBV + blockSize
            endBH = initBH + blockSize

            #img_r[initBV:endBV, initBH:endBH] = pr[pos] #img for benchmark
            #pos += 1
        
    img_r = img_r[(blockSize*n_area) + i_diff : (blockSize*n_area) + i_diff + o_shape_i, (blockSize*n_area)  + j_diff: (blockSize*n_area)  + j_diff + o_shape_j]
    return img_r
