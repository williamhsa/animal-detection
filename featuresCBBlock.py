# -*- coding: utf-8 -*-
from skimage.util import view_as_windows
import numpy as np
from sklearn import cluster
from skimage import data
from skimage.feature import local_binary_pattern
from skimage.feature import hog
from skimage.util import img_as_float
from skimage.filters import gabor_kernel
from scipy import ndimage as nd
from auxfunc import *
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import LogisticRegression

def extractImageFeatures(img_rgb, img_gt, kmeans, fparams):
#def extractImageFeatures(img_rgb, img_gt, fparams): #tirei kmeans
     
    n_features, total_features = calculateFeatureSize(fparams)
    
    f_imgs = generateFeaturesImages(img_rgb, fparams)
    
    img_gt = fparams['problem'].image2class(img_gt)
    
    blockSizeIn = fparams['pathConfig']['blockSizeIn']
    blockSizeOut = fparams['pathConfig']['blockSizeOut']
    maxBlockSize = fparams['pathConfig']['maxBlockSize']
    
    
    if blockSizeOut != 0:
        beginM = int((maxBlockSize - blockSizeIn) / 2)
        #print 'valores de begin e blockSizein'
        endM = beginM + blockSizeIn - 1
        beginO = int((blockSizeOut - blockSizeIn) / 2)
        endO = beginM + blockSizeIn
        #print (beginO)
        #print(blockSizeIn)
    else:
        beginM =  endM = beginO = endO = 0
    # POR CAUSA DOS COMENTARIOS
    bottom_features = extractBlockFeatures(img_rgb.shape[0] - blockSizeIn, img_rgb.shape[0], int(int(img_rgb.shape[1]/2.0) - blockSizeIn/2.0),int(int(img_rgb.shape[1]/2.0) - blockSizeIn/2.0) + blockSizeIn, f_imgs, n_features, fparams)

    features = []
    
    for i in range(beginM, img_rgb.shape[0] - endM, blockSizeIn):
        for j in range(beginM, img_rgb.shape[1] - endM, blockSizeIn):
            
            classRates = fparams['problem'].pathClassRate(img_gt[i:i+blockSizeIn,j:j+blockSizeIn])
            
            if np.max(classRates) >= fparams['perc']:#if this block have all of the same class
                
                f_ex = np.empty(total_features) #Return a new array of given shape and type, without initializing entries.
                i_feat = 0
                
                f_ex[i_feat:i_feat+n_features] = extractBlockFeatures(i, i+blockSizeIn, j, j+blockSizeIn, f_imgs, n_features, fparams)
                i_feat += n_features
                # Bloco contextual
                if blockSizeOut != 0:
                    ref_img = np.ones((blockSizeOut, blockSizeOut), dtype = int) # np.ones -> Return a new array of given shape and type, filled with ones. Uma matriz preenchida com 1's
                    #ref_img = np.full((blockSizeOut, blockSizeOut), -1) #Retornando um matriz preenchida com -1's
                    #ref_img[beginO: beginO + blockSizeIn, beginO: beginO + blockSizeIn] = 0  # criando o bloco de classificação
                    
                    #Bloco de Classificacao
                    #Classificacao cruz
                    ref_img[5,7] = 0
                    ref_img[6,7] = 0
                    ref_img[7,5] = 0
                    ref_img[7,6] = 0
                    ref_img[7,7] = 0
                    ref_img[7,8] = 0
                    ref_img[7,9] = 0
                    ref_img[8,7] = 0
                    ref_img[9,7] = 0
                    # Ative o restante para virar alvo
                    #Classificação tipo Alvo
                    ref_img[6,6] = 0
                    ref_img[6,8] = 0
                    ref_img[8,6] = 0
                    ref_img[8,8] = 0
                    
                    #print ref_img
                    indiO, indjO = np.where(ref_img == 1)
                    indiO.flags.writeable = True; indjO.flags.writeable = True;
                    indiO += i - beginO
                    indjO += j - beginO
                    f_ex[i_feat:i_feat+n_features] = extractSegmentFeatures(indiO, indjO, f_imgs, n_features, fparams)
                    i_feat += n_features
                #POR CAUSA DO COMENTARIO    
                '''
                if fparams['pos']['use']:
                    f_ex[i_feat] = i / float(img_rgb.shape[0]); i_feat += 1
                    f_ex[i_feat] = i / float(img_rgb.shape[1]); i_feat += 1
            
                if fparams['bottom']['use']:
                    f_ex[i_feat:i_feat+n_features] = f_ex[0:n_features] - bottom_features; i_feat += n_features
                '''
                f_ex[i_feat] = np.argmax(classRates)
                
                features.append(f_ex)    
    return np.array(features)
    
def classifyTrainingImg(img_rgb, img_gt, fparams):
    
    img_gt = fparams['problem'].image2class(img_gt)
    
    blockSizeIn = fparams['pathConfig']['blockSizeIn']
    blockSizeOut = fparams['pathConfig']['blockSizeOut']
    maxBlockSize = fparams['pathConfig']['maxBlockSize']
    
    if blockSizeOut != 0:
        beginM = int((maxBlockSize - blockSizeIn) / 2)
        endM = beginM + blockSizeIn - 1
    else:
        beginM =  endM = beginO = endO = 0
    
    img_r = np.empty(img_gt.shape, dtype = int)
    img_r[:,:] = 255    #DUVIDA
    
    for i in range(beginM, img_rgb.shape[0] - endM , blockSizeIn):
        for j in range(beginM, img_rgb.shape[1] - endM, blockSizeIn):
            
            classRates = fparams['problem'].pathClassRate(img_gt[i:i+blockSizeIn,j:j+blockSizeIn])
            
            if np.max(classRates) >= fparams['perc']:   #if this block have all of the same class
                img_r[i:i+blockSizeIn,j:j+blockSizeIn] = np.argmax(classRates)
  
    return img_r
    
def classifyImg(model, scaler, img_rgb, kmeans, fparams):
     
    blockSizeIn = fparams['pathConfig']['blockSizeIn']
    blockSizeOut = fparams['pathConfig']['blockSizeOut']
    maxBlockSize = fparams['pathConfig']['maxBlockSize']
    
    if blockSizeOut != 0:
        beginM = int((maxBlockSize - blockSizeIn) / 2)
        endM = beginM + blockSizeIn - 1
        beginO = int((blockSizeOut - blockSizeIn) / 2)
        endO = beginM + blockSizeIn
        
    else:
        beginM =  endM = beginO = endO = 0
    
    o_shape_i = img_rgb.shape[0]
    o_shape_j = img_rgb.shape[1]
    
    i_diff = img_rgb.shape[0] % blockSizeIn
    j_diff = img_rgb.shape[1] % blockSizeIn

    if i_diff != 0:
        i_diff = blockSizeIn - i_diff
    
    if j_diff != 0:
        j_diff = blockSizeIn - j_diff
              
    img_rgb = np.pad(img_rgb, (     (   beginM   ,  beginM + i_diff   )    ,   (   beginM    ,   beginM + j_diff  ), (0, 0)  ), mode='reflect')
    
    n_features, total_features = calculateFeatureSize(fparams)
    
    f_imgs = generateFeaturesImages(img_rgb, fparams)
    
    bottom_features = extractBlockFeatures(img_rgb.shape[0] - blockSizeIn - beginM - i_diff, img_rgb.shape[0] - beginM - i_diff, int(int(img_rgb.shape[1]/2.0) - blockSizeIn/2.0),int(int(img_rgb.shape[1]/2.0) - blockSizeIn/2.0) + blockSizeIn, f_imgs, n_features, fparams)
    
    features = []
    
    for i in range(beginM, img_rgb.shape[0] - endM, blockSizeIn):
        for j in range(beginM, img_rgb.shape[1] - endM, blockSizeIn):
                
                f_ex = np.empty(total_features-1)
                i_feat = 0
                
                f_ex[i_feat:i_feat+n_features] = extractBlockFeatures(i, i+blockSizeIn, j, j+blockSizeIn, f_imgs, n_features, fparams)
                i_feat += n_features
                
                if blockSizeOut != 0:
                    ref_img = np.ones((blockSizeOut, blockSizeOut), dtype = int)
                    ref_img[beginO: beginO + blockSizeIn, beginO: beginO + blockSizeIn] = 0
                    
                    indiO, indjO = np.where(ref_img == 1)
                    indiO.flags.writeable = True; indjO.flags.writeable = True;
                    indiO += i - beginO
                    indjO += j - beginO
                    f_ex[i_feat:i_feat+n_features] = extractSegmentFeatures(indiO, indjO, f_imgs, n_features, fparams)
                    i_feat += n_features
                #POR CAUSA DOS COMENTARIOS
                '''
                if fparams['pos']['use']:
                    f_ex[i_feat] = i / float(img_rgb.shape[0]); i_feat += 1
                    f_ex[i_feat] = i / float(img_rgb.shape[1]); i_feat += 1
            
                if fparams['bottom']['use']:
                    f_ex[i_feat:i_feat+n_features] = f_ex[0:n_features] - bottom_features; i_feat += n_features
                '''
                features.append(f_ex)

    features = np.array(features).astype(None)
    features = scaler.transform(features) 
    pr = np.array(model.predict(features))
    
    img_r = np.empty((img_rgb.shape[0], img_rgb.shape[1]), dtype = np.uint8)

    pos = 0
    #print results on imgs.
    for i in range(beginM, img_rgb.shape[0] - endM, blockSizeIn):
        for j in range(beginM, img_rgb.shape[1] - endM, blockSizeIn):

            img_r[i:i+blockSizeIn,j:j+blockSizeIn] = pr[pos]
            pos += 1
            
    img_r = img_r[beginM : beginM + o_shape_i, beginM : beginM + o_shape_j]
            
    return img_r

def predictProbabilitesLR(model, scaler, img_rgb, kmeans, fparams):
     
    blockSizeIn = fparams['pathConfig']['blockSizeIn']
    blockSizeOut = fparams['pathConfig']['blockSizeOut']
    maxBlockSize = fparams['pathConfig']['maxBlockSize']
    
    if blockSizeOut != 0:
        beginM = int((maxBlockSize - blockSizeIn) / 2)
        endM = beginM + blockSizeIn - 1
        beginO = int((blockSizeOut - blockSizeIn) / 2)
        endO = beginM + blockSizeIn
        
    else:
        beginM =  endM = beginO = endO = 0
    
    o_shape_i = img_rgb.shape[0]
    o_shape_j = img_rgb.shape[1]
    
    i_diff = img_rgb.shape[0] % blockSizeIn
    j_diff = img_rgb.shape[1] % blockSizeIn

    if i_diff != 0:
        i_diff = blockSizeIn - i_diff
    
    if j_diff != 0:
        j_diff = blockSizeIn - j_diff
              
    img_rgb = np.pad(img_rgb, (     (   beginM   ,  beginM + i_diff   )    ,   (   beginM    ,   beginM + j_diff  ), (0, 0)  ), mode='reflect')
    
    n_features, total_features = calculateFeatureSize(fparams)
    
    f_imgs = generateFeaturesImages(img_rgb, fparams)
    
    bottom_features = extractBlockFeatures(img_rgb.shape[0] - blockSizeIn - beginM - i_diff, img_rgb.shape[0] - beginM - i_diff, int(int(img_rgb.shape[1]/2.0) - blockSizeIn/2.0),int(int(img_rgb.shape[1]/2.0) - blockSizeIn/2.0) + blockSizeIn, f_imgs, n_features, fparams)
    
    features = []
    
    for i in range(beginM, img_rgb.shape[0] - endM, blockSizeIn):
        for j in range(beginM, img_rgb.shape[1] - endM, blockSizeIn):
                
                f_ex = np.empty(total_features-1)
                i_feat = 0
                
                f_ex[i_feat:i_feat+n_features] = extractBlockFeatures(i, i+blockSizeIn, j, j+blockSizeIn, f_imgs, n_features, fparams)
                i_feat += n_features
                
                if blockSizeOut != 0:
                    ref_img = np.ones((blockSizeOut, blockSizeOut), dtype = int)
                    ref_img[beginO: beginO + blockSizeIn, beginO: beginO + blockSizeIn] = 0
                    
                    indiO, indjO = np.where(ref_img == 1)
                    indiO.flags.writeable = True; indjO.flags.writeable = True;
                    indiO += i - beginO
                    indjO += j - beginO
                    f_ex[i_feat:i_feat+n_features] = extractSegmentFeatures(indiO, indjO, f_imgs, n_features, fparams)
                    i_feat += n_features
                '''
                if fparams['pos']['use']:
                    f_ex[i_feat] = i / float(img_rgb.shape[0]); i_feat += 1
                    f_ex[i_feat] = i / float(img_rgb.shape[1]); i_feat += 1
            
                if fparams['bottom']['use']:
                    f_ex[i_feat:i_feat+n_features] = f_ex[0:n_features] - bottom_features; i_feat += n_features
                '''
                features.append(f_ex)

    features = np.array(features).astype(None)
    features = scaler.transform(features) 
    pr = np.array(model.predict(features))
    
    img_r = np.empty((img_rgb.shape[0], img_rgb.shape[1]), dtype = np.uint8)
    
    pos = 0
    
    #Logistic Regression probability    
    img_proba_sum = np.zeros((img_rgb.shape[0], img_rgb.shape[1]),dtype=np.float)
    img_proba_zero = np.zeros((img_rgb.shape[0], img_rgb.shape[1]),dtype=np.float)
    img_proba_one = np.zeros((img_rgb.shape[0], img_rgb.shape[1]),dtype=np.float)
           

    print "Predicting probabilities for blocks (LR)"
    proba = model.predict_proba(features)


    for i in range(beginM, img_rgb.shape[0] - endM, blockSizeIn):
        for j in range(beginM, img_rgb.shape[1] - endM, blockSizeIn):
            if proba[pos][0] > 0.5:
                img_proba_zero[i:i+blockSizeIn,j:j+blockSizeIn] = proba[pos][0]
                #print proba[pos][0]
            else:
                img_proba_one[i:i+blockSizeIn,j:j+blockSizeIn] = proba[pos][1]
                #print proba[pos][1]
            pos += 1
            

    img_proba_zero = img_proba_zero[beginM : beginM + o_shape_i, beginM : beginM + o_shape_j]
    img_proba_one = img_proba_one[beginM : beginM + o_shape_i, beginM : beginM + o_shape_j]
    #np.savetxt('img_proba_zero.txt', img_proba_zero,fmt='%1.3f',delimiter='\t')
    #np.savetxt('img_proba_one.txt', img_proba_one,fmt='%1.3f',delimiter='\t')
    
    img_proba_sum = img_proba_zero + img_proba_one

    img_r = np.empty((img_rgb.shape[0], img_rgb.shape[1]), dtype = np.uint8)

    pos = 0
    #print results on imgs.
    for i in range(beginM, img_rgb.shape[0] - endM, blockSizeIn):
        for j in range(beginM, img_rgb.shape[1] - endM, blockSizeIn):

            img_r[i:i+blockSizeIn,j:j+blockSizeIn] = pr[pos]
            pos += 1
            
    img_r = img_r[beginM : beginM + o_shape_i, beginM : beginM + o_shape_j]
            
    return img_proba_sum, img_proba_zero, img_proba_one
