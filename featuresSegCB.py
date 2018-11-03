from skimage.util import view_as_windows
import numpy as np
from sklearn import cluster
from skimage import data
from skimage.color import rgb2gray
from skimage.feature import local_binary_pattern
from skimage.feature import hog
from skimage.util import img_as_float, img_as_ubyte
from skimage.filters import gabor_kernel
from scipy import ndimage as nd
from auxfunc import *
from skimage.segmentation import slic
from skimage import color
import matplotlib.pyplot as plt
from skimage.segmentation import mark_boundaries
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import LogisticRegression

def diferenceBlockSeg(initBV, endBV, initBH, endBH, idx, idy):

    temp = np.ones((endBV - initBV, endBH - initBH), dtype = int)
    idx = idx - initBV
    idy = idy - initBH

    temp[idx, idy] = 0

    iddx, iddy = np.where(temp == 1)
    return (iddx, iddy)

def extractImageFeatures(img_rgb, img_gt, kmeans, fparams):

    n_features, total_features = calculateFeatureSize(fparams)

    f_imgs = generateFeaturesImages(img_rgb, fparams)

    img_gt = fparams['problem'].image2class(img_gt)

    marginSize = fparams['pathConfig']['marginSize']

    if marginSize != 0:

        sizeV = img_rgb.shape[0]
        sizeH = img_rgb.shape[1]

        img_rgb_red = img_rgb[marginSize:sizeV-marginSize, marginSize:sizeH-marginSize, :]

        segments_red = slic(img_rgb_red, n_segments = fparams['pathConfig']['nSegs'], compactness=10, sigma=1, enforce_connectivity=True)
        n_segments = len(np.unique(segments_red))

        segments = np.empty((img_rgb.shape[0], img_rgb.shape[1]), dtype = int)
        segments[:,:] = -1
        segments[marginSize:sizeV-marginSize, marginSize:sizeH-marginSize] = segments_red

    else:

        segments = slic(img_rgb, n_segments = fparams['pathConfig']['nSegs'], compactness=10, sigma=1)
        n_segments = len(np.unique(segments))

    bx = segments.shape[0] - 1 - marginSize
    by = int(segments.shape[1] / 2.0)
    bottom_segment = segments[bx, by]

    idx, idy = np.where(segments == bottom_segment)

    bottom_features = extractSegmentFeatures(idx, idy, f_imgs, n_features, fparams)

    features = []

    for i in range(n_segments):

        idx, idy = np.where(segments == i)
        seg = img_gt[idx,idy]

        classRates = fparams['problem'].pathClassRate(seg)

        if np.max(classRates) >= fparams['perc']:   #if this block have all of the same class

            feat_ex = np.empty(total_features)
            i_feat = 0

            feat_ex[i_feat:i_feat + n_features] = extractSegmentFeatures(idx, idy, f_imgs, n_features, fparams)
            i_feat += n_features

            if marginSize != 0:

                initBV = np.min(idx) - marginSize
                endBV = np.max(idx) + marginSize
                initBH = np.min(idy) - marginSize
                endBH = np.max(idy) + marginSize

                bidx, bidy = diferenceBlockSeg(initBV, endBV, initBH, endBH, idx, idy)

                feat_ex[i_feat:i_feat + n_features] = extractSegmentFeatures(bidx, bidy, f_imgs, n_features, fparams)
                i_feat += n_features
            '''
            # normalized pos
            if fparams['pos']['use']:
                feat_ex[i_feat] = np.mean(idx) / float(img_rgb.shape[0]); i_feat += 1
                feat_ex[i_feat] = np.mean(idy) / float(img_rgb.shape[1]); i_feat += 1

            if fparams['bottom']['use']:
                feat_ex[i_feat:i_feat+n_features] = feat_ex[0:n_features] - bottom_features; i_feat += n_features
            '''
            feat_ex[i_feat] = np.argmax(classRates)
            features.append(feat_ex)

    return np.array(features)


def classifyImg(model, scaler, img_rgb, kmeans, fparams):

    n_features, total_features = calculateFeatureSize(fparams)
    marginSize = fparams['pathConfig']['marginSize']


    if marginSize != 0:
        o_sizeV = img_rgb.shape[0]
        o_sizeH = img_rgb.shape[1]

        img_rgb = np.pad(img_rgb, (     (   marginSize   ,  marginSize   )    ,   (   marginSize    ,   marginSize  ), (0, 0)  ), mode='reflect')
        sizeV = img_rgb.shape[0]
        sizeH = img_rgb.shape[1]

        img_rgb_red = img_rgb[marginSize:sizeV-marginSize, marginSize:sizeH-marginSize, :]

        segments_red = slic(img_rgb_red, n_segments = fparams['pathConfig']['nSegs'], compactness=10, sigma=1, enforce_connectivity=True)
        n_segments = len(np.unique(segments_red))

        segments = np.empty((img_rgb.shape[0], img_rgb.shape[1]), dtype = int)
        segments[:,:] = -1
        segments[marginSize:sizeV-marginSize, marginSize:sizeH-marginSize] = segments_red

    else:

        segments = slic(img_rgb, n_segments = fparams['pathConfig']['nSegs'], compactness=10, sigma=1)
        n_segments = len(np.unique(segments))

    f_imgs = generateFeaturesImages(img_rgb, fparams)

    bx = segments.shape[0] - 1 - marginSize
    by = int(segments.shape[1] / 2.0)
    bottom_segment = segments[bx, by]

    idx, idy = np.where(segments == bottom_segment)

    bottom_features = extractSegmentFeatures(idx, idy, f_imgs, n_features, fparams)

    features = []

    for i in range(n_segments):

        idx, idy = np.where(segments == i)

        feat_ex = np.empty(total_features-1)
        i_feat = 0

        feat_ex[i_feat:i_feat + n_features] = extractSegmentFeatures(idx, idy, f_imgs, n_features, fparams)
        i_feat += n_features

        if marginSize != 0:

            initBV = np.min(idx) - marginSize
            endBV = np.max(idx) + marginSize
            initBH = np.min(idy) - marginSize
            endBH = np.max(idy) + marginSize

            bidx, bidy = diferenceBlockSeg(initBV, endBV, initBH, endBH, idx, idy)

            feat_ex[i_feat:i_feat + n_features] = extractSegmentFeatures(bidx, bidy, f_imgs, n_features, fparams)
            i_feat += n_features
        '''
        # normalized pos
        if fparams['pos']['use']:
            feat_ex[i_feat] = np.mean(idx) / float(img_rgb.shape[0]); i_feat += 1
            feat_ex[i_feat] = np.mean(idy) / float(img_rgb.shape[1]); i_feat += 1

        if fparams['bottom']['use']:
            feat_ex[i_feat:i_feat+n_features] = feat_ex[0:n_features] - bottom_features; i_feat += n_features
        '''

        features.append(feat_ex)


    features = np.array(features).astype(None)
    features = scaler.transform(features)
    pr = np.array(model.predict(features))

    img_r = np.empty((img_rgb.shape[0], img_rgb.shape[1]), dtype = np.uint8)

    if marginSize != 0:
        img_seg = mark_boundaries(img_rgb_red, segments_red)
    else:
        img_seg = mark_boundaries(img_rgb, segments)

    #print results on imgs.
    for i in range(n_segments):
        idx, idy = np.where(segments == i)
        img_r[idx,idy] = pr[i] #img for benchmark

    if marginSize != 0:
        img_r = img_r[marginSize : marginSize + o_sizeV, marginSize: marginSize + o_sizeH]


    img_boundaries = img_as_ubyte(img_seg)

    return (img_r, img_boundaries)


def classifyTrainingImg(img, img_gt, fparams):

    marginSize = fparams['pathConfig']['marginSize']

    sizeV = img.shape[0]
    sizeH = img.shape[1]

    img_red = img[marginSize:sizeV-marginSize, marginSize:sizeH-marginSize, :]

    segments_red = slic(img_red, n_segments = fparams['pathConfig']['nSegs'], compactness=10, sigma=1, enforce_connectivity=True)
    n_segments = len(np.unique(segments_red))

    segments = np.empty((img.shape[0], img.shape[1]), dtype = int)
    segments[:,:] = -1
    segments[marginSize:sizeV-marginSize, marginSize:sizeH-marginSize] = segments_red

    img_gt = fparams['problem'].image2class(img_gt)
    img_r = np.empty(img_gt.shape, dtype = int)
    img_r[:,:] = -1

    for i in range(n_segments):

        idx, idy = np.where(segments == i)
        seg = img_gt[idx,idy]

        classRates = fparams['problem'].pathClassRate(seg)

        if np.max(classRates) >= fparams['perc']:   #if this block have all of the same class
            img_r[idx, idy] = np.argmax(classRates)
    img_boundaries = mark_boundaries(img, segments)
    img_boundaries = img_as_ubyte(img_boundaries)
    return (img_r, img_boundaries)

def predictProbabilitesLR(model, scaler, img_rgb, kmeans, fparams):

    n_features, total_features = calculateFeatureSize(fparams)
    marginSize = fparams['pathConfig']['marginSize']


    if marginSize != 0:
        o_sizeV = img_rgb.shape[0]
        o_sizeH = img_rgb.shape[1]

        img_rgb = np.pad(img_rgb, (     (   marginSize   ,  marginSize   )    ,   (   marginSize    ,   marginSize  ), (0, 0)  ), mode='reflect')
        sizeV = img_rgb.shape[0]
        sizeH = img_rgb.shape[1]

        img_rgb_red = img_rgb[marginSize:sizeV-marginSize, marginSize:sizeH-marginSize, :]

        segments_red = slic(img_rgb_red, n_segments = fparams['pathConfig']['nSegs'], compactness=10, sigma=1, enforce_connectivity=True)
        n_segments = len(np.unique(segments_red))

        segments = np.empty((img_rgb.shape[0], img_rgb.shape[1]), dtype = int)
        segments[:,:] = -1
        segments[marginSize:sizeV-marginSize, marginSize:sizeH-marginSize] = segments_red

    else:

        segments = slic(img_rgb, n_segments = fparams['pathConfig']['nSegs'], compactness=10, sigma=1)
        n_segments = len(np.unique(segments))

    f_imgs = generateFeaturesImages(img_rgb, fparams)

    bx = segments.shape[0] - 1 - marginSize
    by = int(segments.shape[1] / 2.0)
    bottom_segment = segments[bx, by]

    idx, idy = np.where(segments == bottom_segment)

    bottom_features = extractSegmentFeatures(idx, idy, f_imgs, n_features, fparams)

    features = []

    for i in range(n_segments):

        idx, idy = np.where(segments == i)

        feat_ex = np.empty(total_features-1)
        i_feat = 0

        feat_ex[i_feat:i_feat + n_features] = extractSegmentFeatures(idx, idy, f_imgs, n_features, fparams)
        i_feat += n_features

        if marginSize != 0:

            initBV = np.min(idx) - marginSize
            endBV = np.max(idx) + marginSize
            initBH = np.min(idy) - marginSize
            endBH = np.max(idy) + marginSize

            bidx, bidy = diferenceBlockSeg(initBV, endBV, initBH, endBH, idx, idy)

            feat_ex[i_feat:i_feat + n_features] = extractSegmentFeatures(bidx, bidy, f_imgs, n_features, fparams)
            i_feat += n_features
        '''
        # normalized pos
        if fparams['pos']['use']:
            feat_ex[i_feat] = np.mean(idx) / float(img_rgb.shape[0]); i_feat += 1
            feat_ex[i_feat] = np.mean(idy) / float(img_rgb.shape[1]); i_feat += 1

        if fparams['bottom']['use']:
            feat_ex[i_feat:i_feat+n_features] = feat_ex[0:n_features] - bottom_features; i_feat += n_features
        '''

        features.append(feat_ex)


    features = np.array(features).astype(None)
    features = scaler.transform(features)
    pr = np.array(model.predict(features))

    img_r = np.empty((img_rgb.shape[0], img_rgb.shape[1]), dtype = np.uint8)

    if marginSize != 0:
        img_seg = mark_boundaries(img_rgb_red, segments_red)
    else:
        img_seg = mark_boundaries(img_rgb, segments)

    #print results on imgs.
    for i in range(n_segments):
        idx, idy = np.where(segments == i)
        img_r[idx,idy] = pr[i] #img for benchmark

    if marginSize != 0:
        img_r = img_r[marginSize : marginSize + o_sizeV, marginSize: marginSize + o_sizeH]

    img_proba_sum = np.zeros((img_rgb.shape[0], img_rgb.shape[1]),dtype=np.float)
    img_proba_zero = np.zeros((img_rgb.shape[0], img_rgb.shape[1]),dtype=np.float)
    img_proba_one = np.zeros((img_rgb.shape[0], img_rgb.shape[1]),dtype=np.float)


    print "Predicting probabilities for segments (LR)"
    proba = model.predict_proba(features)

    for i in range(n_segments):
        idx, idy = np.where(segments == i)
        if proba[i][0] > 0.5:
            img_proba_zero[idx,idy] = proba[i][0]
        else:
            img_proba_one[idx,idy] = proba[i][1]

    if marginSize != 0:
        img_proba_zero = img_proba_zero[marginSize : marginSize + o_sizeV, marginSize: marginSize + o_sizeH]
        img_proba_one = img_proba_one[marginSize : marginSize + o_sizeV, marginSize: marginSize + o_sizeH]

    img_proba_sum = img_proba_zero + img_proba_one

    return (img_proba_sum, img_proba_zero, img_proba_one)
