# -*- coding: utf-8 -*-

import numpy as np
from skimage.feature import local_binary_pattern
import pickle
from scipy import ndimage as nd
from skimage import color
from skimage import data
from skimage.filters.rank import entropy
from skimage.morphology import disk
from skimage.util import img_as_ubyte
import matplotlib.pyplot as plt
from skimage import io

def calculateFeatureSize(fparams):
    
    per_segment_features = 0
    
    if fparams['RGB']['use']:
        if fparams['RGB']['mean']:
            per_segment_features += 3
        if fparams['RGB']['std']:
            per_segment_features += 3
            
    if fparams['LAB']['use']:
        if fparams['LAB']['mean']:
            per_segment_features += 3
        if fparams['LAB']['std']:
            per_segment_features += 3
            
    if fparams['HSV']['use']:
        if fparams['HSV']['mean']:
            per_segment_features += 3
        if fparams['HSV']['std']:
            per_segment_features += 3

    if fparams['YCbCr']['use']:
        if fparams['YCbCr']['mean']:
            per_segment_features += 3
        if fparams['YCbCr']['std']:
            per_segment_features += 3

    if fparams['xyz']['use']:
        if fparams['xyz']['mean']:
            per_segment_features += 3
        if fparams['xyz']['std']:
            per_segment_features += 3

    if fparams['yiq']['use']:
        if fparams['yiq']['mean']:
            per_segment_features += 3
        if fparams['yiq']['std']:
            per_segment_features += 3

    if fparams['yuv']['use']:
        if fparams['yuv']['mean']:
            per_segment_features += 3
        if fparams['yuv']['std']:
            per_segment_features += 3
            
    if fparams['entropy']['use']:
        if fparams['entropy']['mean']:
            per_segment_features += 1
        if fparams['entropy']['std']:
            per_segment_features += 1
        
    if fparams['gray']['use']:
        if fparams['gray']['mean']:
            per_segment_features += 1
        if fparams['gray']['std']:
            per_segment_features += 1
    '''
    if fparams['texton']['use']:
        if fparams['texton']['mean']:
            per_segment_features += fparams['texton']['n_kernels']
        if fparams['texton']['std']:
            per_segment_features += fparams['texton']['n_kernels']
        if fparams['texton']['hist_max']:
            per_segment_features += fparams['texton']['n_kernels']
        if fparams['texton']['hist_kmeans']:
            per_segment_features += fparams['texton']['hist_size_kmeans']
    '''
    if fparams['LBP']['use']:
        if fparams['LBP']['gray']:
            per_segment_features += 2**fparams['LBP']['n_neibor']
        if fparams['LBP']['red']:
            per_segment_features += 2**fparams['LBP']['n_neibor']
        if fparams['LBP']['green']:
            per_segment_features += 2**fparams['LBP']['n_neibor']
        if fparams['LBP']['blue']:
            per_segment_features += 2**fparams['LBP']['n_neibor']

    #Pelo fato de comentar  features da 11 ate 16
    '''
    if fparams['grayD']['use']:
        per_segment_features += 6
    '''
    '''
    if fparams['grayA']['use']:
        per_segment_features += 6
    '''
    '''
    if fparams['VegetationIndex']['use']:
        if fparams['VegetationIndex']['NDVI']:
            per_segment_features += 1
        if fparams['VegetationIndex']['NNIR']:
            per_segment_features += 1        
        if fparams['VegetationIndex']['NRED']:
            per_segment_features += 1
        if fparams['VegetationIndex']['NGREEN']:
            per_segment_features += 1
        if fparams['VegetationIndex']['PVI']:
            per_segment_features += 1 
    '''
    total_features = per_segment_features + 1 #a space for class
    
    if fparams['pathConfig'].has_key('marginSize'):
        if fparams['pathConfig']['marginSize'] != 0:
            total_features += per_segment_features
            
    if fparams['pathConfig'].has_key('blockSizeOut'):
        if fparams['pathConfig']['blockSizeOut'] != 0:
            total_features += per_segment_features
    
    if fparams['pathConfig'].has_key('n_area'): 
        total_features += per_segment_features*(8*fparams['pathConfig']['n_area'])
    #Pelo fato de comentar  features da 11 ate 16
    '''  
    if fparams['bottom']['use']:
        if fparams['pathConfig'].has_key('n_area'): 
            total_features += per_segment_features
        total_features += per_segment_features
    if fparams['pos']['use']:
        total_features += 2
    ''' 
    return (per_segment_features, total_features)

def evaluateModel(imgs, imgs_gt, model, scaler, kmeans, fparams, activeFeatures, contImgs, colorImg):

    nclass = fparams['problem'].getNClass()
    confusionm = np.zeros((nclass,nclass), dtype = np.uint64)
    
    listFeaturesActive = [] # lista de features que estão sendo utilizada na iteração atual;
    
    # print 'IMGS - Tests\n'
    # print imgs
    # print 'IMGS_GT\n'
    # print imgs_gt
    # print '\n'

    #laço para verificar quais features estão ativas no FPARAMS e colocando o nome delas numa lista
    for i in activeFeatures:
        if(activeFeatures[i]['use'] == True):
            listFeaturesActive.append(i)
    
    print listFeaturesActive
        
    for nImg in range(len(imgs)):
        img = data.imread(imgs[nImg])
        img_gt = data.imread(imgs_gt[nImg])
        
        if img_gt.ndim == 3:
            img_gt = img_gt[:,:,0]
        
        print 'Evaluating img: ' + str(imgs[nImg])
        
        img_gt = fparams['problem'].image2class(img_gt)
        
        
        if fparams['pathConfig'].has_key('marginSize'):
            pred, img = fparams['path'].classifyImg(model, scaler, img, kmeans, fparams)
        else:
            pred = fparams['path'].classifyImg(model, scaler, img, kmeans, fparams)
    
    
        for i in range(pred.shape[0]):
            for j in range(pred.shape[1]):
                if img_gt[i,j] < 127:
                    confusionm[img_gt[i,j], pred[i,j]] += 1
                
                if colorImg:
                    if img_gt[i,j] == 0 and pred[i,j] == 1: # false positive B
                        img[i,j, 2] = 255
                    elif img_gt[i,j] == 1 and pred[i,j] == 1:  # true positive G
                        img[i,j, 1] = 255
                    elif img_gt[i,j] == 1 and pred[i,j] == 0: # false negative R
                        img[i,j, 0] = 255
        # SALVANDO IMAGEM do EVALUATIONS
        if colorImg:
           io.imsave(imgs[nImg].split('/')[-1] + str(contImgs) + 'evaluation.png',img)
    
   
    
    total_pixels = np.sum(confusionm)
    acc = float(confusionm[0,0] + confusionm[1,1]) / total_pixels
    precision = float(confusionm[1,1]) / (confusionm[0,1] + confusionm[1,1])
    recall = float(confusionm[1,1]) / (confusionm[1,0] + confusionm[1,1])
    fmeasure = (2*precision*recall)/(precision+recall)
    print 'Acc: %f, precision: %f, recall: %f' % (acc, precision, recall)
    print 'F-measure:', fmeasure
    print '\nGravando F-Measure no arquivo de saida!'
    #Salvando dados no arquivo --> saidaFmeasure.txt
    file_output = open("saidaFmeasure.txt","a") #abrindo um arquivo
    out = ""
    for i in listFeaturesActive:
        out+=i+" "
    file_output.write(str(contImgs)+")"+"Features Actives-> "+out)
    file_output.write("| F-Measure->"+str(fmeasure)+"\n")

    if file_output.closed == False:
        file_output.close()    
    print '\nDados gravados no arquivo com sucesso!\n'

    return confusionm

def generateFeaturesImages(img_rgb, fparams):
    
    f_imgs = dict()
    
    f_imgs['gray'] = color.rgb2gray(img_rgb)
    
    f_imgs['RGB'] = img_rgb

    if fparams['LAB']['use']:
        f_imgs['LAB'] = color.rgb2lab(img_rgb)
    if fparams['HSV']['use']:
        f_imgs['HSV'] = color.rgb2hsv(img_rgb)
    if fparams['YCbCr']['use']:
        f_imgs['YCbCr'] = color.rgb2ycbcr(img_rgb)
    if fparams['xyz']['use']:
        f_imgs['xyz'] = color.rgb2xyz(img_rgb)
    if fparams['yiq']['use']:
        f_imgs['yiq'] = color.rgb2yiq(img_rgb)
    if fparams['yuv']['use']:
        f_imgs['yuv'] = color.rgb2yuv(img_rgb)

    if fparams['entropy']['use']:
        f_imgs['entropy'] = entropy(img_as_ubyte(f_imgs['gray']), disk(5))
    #POR CAUSA DOS COMENTARIOS     TESTE
    '''
    if fparams['texton']['use']:    # appende gabor filter responses to extract mean and std
        kernels = generateKernels()   #generate gabor filters
        texton_imgs = np.empty((f_imgs['gray'].shape[0], f_imgs['gray'].shape[1], fparams['texton']['n_kernels']))
        for k, kernel in enumerate(kernels):
            texton_imgs[:,:,k] = nd.convolve(f_imgs['gray'], kernel)
            
        if fparams['texton']['mean'] or fparams['texton']['std']:
            f_imgs['texton'] = texton_imgs
        
        if fparams['texton']['hist_max']:
            f_imgs['hist_max'] = np.argmax(texton_imgs, 2)
            
        if fparams['texton']['hist_kmeans']:
            data_t = np.empty((f_imgs['gray'].size, fparams['texton']['n_kernels']))
        
            for k in range(fparams['texton']['n_kernels']):
                data_t [:, k] = texton_imgs[:,:,k].flatten()
            
            textons = kmeans.predict(data_t)

            f_imgs['hist_kmeans'] = textons.reshape(f_imgs['gray'].shape[0], f_imgs['gray'].shape[1])
    ''' 
    '''          
    if fparams['grayD']['use'] or fparams['grayA']['use']:
        f_imgs['grayMeans'] = generateGrayMeans(f_imgs['gray'], 6, 6, 1)  #gray diff between mean gray of the block and some areas
    '''
    if fparams['LBP']['use']:
        if fparams['LBP']['gray']:
            f_imgs['LBP_gray'] = local_binary_pattern(f_imgs['gray'], fparams['LBP']['n_neibor'], fparams['LBP']['radius'])  #lbp image
        if fparams['LBP']['red']:
            f_imgs['LBP_r'] = local_binary_pattern(f_imgs['RGB'][:,:,0], fparams['LBP']['n_neibor'], fparams['LBP']['radius'])  #lbp image
        if fparams['LBP']['green']:
            f_imgs['LBP_g'] = local_binary_pattern(f_imgs['RGB'][:,:,1], fparams['LBP']['n_neibor'], fparams['LBP']['radius'])  #lbp image
        if fparams['LBP']['blue']:
            f_imgs['LBP_b'] = local_binary_pattern(f_imgs['RGB'][:,:,2], fparams['LBP']['n_neibor'], fparams['LBP']['radius'])  #lbp image            
   
    
    """ if fparams['VegetationIndex']['use']:
        img_nir = img_rgb.astype(float)[:,:,0]
        img_g = img_rgb.astype(float)[:,:,1]
        img_r = img_rgb.astype(float)[:,:,2]
        if fparams['VegetationIndex']['NDVI']:
            f_imgs['NDVI'] = ((img_nir-img_r)/(img_nir+img_r)) 
        
        if fparams['VegetationIndex']['NNIR']: 
            f_imgs['NNIR'] =  ((img_nir)/(img_nir+img_g+img_r))
            
        if fparams['VegetationIndex']['NGREEN']:
            f_imgs['NGREEN'] = ((img_g)/(img_nir+img_g+img_r))
            
        if fparams['VegetationIndex']['NRED']:
            f_imgs['NRED'] = ((img_r)/(img_nir+img_g+img_r))
            
        if fparams['VegetationIndex']['PVI']:
            b,a = np.polyfit(img_r[0,], img_nir[0,], 1) #regression to find values for b and a 
            f_imgs['PVI'] = ((b*img_nir*img_r)+a)/(np.sqrt(b+1)) """
   
    return f_imgs
    

def extractSegmentFeatures(idx, idy, f_imgs, n_features, fparams):
    
    features = np.empty(n_features)  
    i_feat = 0
    
    if fparams['gray']['use']:
        if fparams['gray']['mean']:
            features[i_feat] = np.mean( f_imgs['gray'][idx, idy] ); i_feat += 1 
        if fparams['gray']['std']:
            features[i_feat] = np.std( f_imgs['gray'][idx, idy] ); i_feat += 1 
            
    if fparams['entropy']['use']:
        if fparams['entropy']['mean']:
            features[i_feat] = np.mean( f_imgs['entropy'][idx, idy] ); i_feat += 1 
        if fparams['entropy']['std']:
            features[i_feat] = np.std( f_imgs['entropy'][idx, idy] ); i_feat += 1 
            
    if fparams['RGB']['use']:
        for i in range(3):
            if fparams['RGB']['mean']:
                features[i_feat] = np.mean( f_imgs['RGB'][idx, idy, i] ); i_feat += 1 
            if fparams['RGB']['std']:
                features[i_feat] = np.std( f_imgs['RGB'][idx, idy, i] ); i_feat += 1 
                
    if fparams['LAB']['use']:
        for i in range(3):
            if fparams['LAB']['mean']:
                features[i_feat] = np.mean( f_imgs['LAB'][idx, idy, i] ); i_feat += 1 
            if fparams['LAB']['std']:
                features[i_feat] = np.std( f_imgs['LAB'][idx, idy, i] ); i_feat += 1 
                
    if fparams['HSV']['use']:
        for i in range(3):
            if fparams['HSV']['mean']:
                features[i_feat] = np.mean( f_imgs['HSV'][idx, idy, i] ); i_feat += 1 
            if fparams['HSV']['std']:
                features[i_feat] = np.std( f_imgs['HSV'][idx, idy, i] ); i_feat += 1 

    if fparams['YCbCr']['use']:
        for i in range(3):
            if fparams['YCbCr']['mean']:
                features[i_feat] = np.mean( f_imgs['YCbCr'][idx, idy, i] ); i_feat += 1 
            if fparams['YCbCr']['std']:
                features[i_feat] = np.std( f_imgs['YCbCr'][idx, idy, i] ); i_feat += 1

    if fparams['xyz']['use']:
        for i in range(3):
            if fparams['xyz']['mean']:
                features[i_feat] = np.mean( f_imgs['xyz'][idx, idy, i] ); i_feat += 1 
            if fparams['xyz']['std']:
                features[i_feat] = np.std( f_imgs['xyz'][idx, idy, i] ); i_feat += 1

    if fparams['yiq']['use']:
        for i in range(3):
            if fparams['yiq']['mean']:
                features[i_feat] = np.mean( f_imgs['yiq'][idx, idy, i] ); i_feat += 1 
            if fparams['yiq']['std']:
                features[i_feat] = np.std( f_imgs['yiq'][idx, idy, i] ); i_feat += 1

    if fparams['yuv']['use']:
        for i in range(3):
            if fparams['yuv']['mean']:
                features[i_feat] = np.mean( f_imgs['yuv'][idx, idy, i] ); i_feat += 1 
            if fparams['yuv']['std']:
                features[i_feat] = np.std( f_imgs['yuv'][idx, idy, i] ); i_feat += 1
 
    if fparams['LBP']['use']:
        n_bins = 2**fparams['LBP']['n_neibor']
        if fparams['LBP']['gray']:
            hist, _ = np.histogram( f_imgs['LBP_gray'][idx, idy] , bins=n_bins, range = (0,n_bins-1), density = True)
            features[i_feat:i_feat+n_bins] = hist; i_feat += n_bins
        if fparams['LBP']['red']:
            hist, _ = np.histogram( f_imgs['LBP_r'][idx, idy] , bins=n_bins, range = (0,n_bins-1), density = True)
            features[i_feat:i_feat+n_bins] = hist; i_feat += n_bins
        if fparams['LBP']['green']:
            hist, _ = np.histogram( f_imgs['LBP_g'][idx, idy] , bins=n_bins, range = (0,n_bins-1), density = True)
            features[i_feat:i_feat+n_bins] = hist; i_feat += n_bins
        if fparams['LBP']['blue']:
            hist, _ = np.histogram( f_imgs['LBP_b'][idx, idy] , bins=n_bins, range = (0,n_bins-1), density = True)
            features[i_feat:i_feat+n_bins] = hist; i_feat += n_bins
    #POR CAUSA DOS COMENTARIOS
    
    '''  
    if fparams['texton']['use']:
        if fparams['texton']['mean'] or fparams['texton']['std']:
            for k in range(fparams['texton']['n_kernels']):
                if fparams['texton']['mean']:
                    features[i_feat] = np.mean( f_imgs['texton'][idx, idy] ); i_feat += 1 
                if fparams['texton']['std']:
                    features[i_feat] = np.std( f_imgs['texton'][idx, idy] ); i_feat += 1 
        
        if fparams['texton']['hist_kmeans']:
            hist, _ = np.histogram( f_imgs['hist_kmeans'][idx, idy] , bins=fparams['texton']['hist_size_kmeans'], range = (0,fparams['texton']['hist_size_kmeans']-1), density = True)
            features[i_feat:i_feat+hist.shape[0]] = hist; i_feat += hist.shape[0]
            
        if fparams['texton']['hist_max']:
            hist, _ = np.histogram( f_imgs['hist_max'][idx, idy] , bins=fparams['texton']['n_kernels'], range = (0,fparams['texton']['n_kernels']-1), density = True)
            features[i_feat:i_feat+hist.shape[0]] = hist; i_feat += hist.shape[0]
    '''
    '''   
    if fparams['grayA']['use']:
        features[i_feat:i_feat+f_imgs['grayMeans'].shape[0]] =  f_imgs['grayMeans']; i_feat += f_imgs['grayMeans'].shape[0]
    '''   
    '''
    if fparams['grayD']['use']:
        features[i_feat:i_feat+f_imgs['grayMeans'].shape[0]] = features[0] - f_imgs['grayMeans']; i_feat += f_imgs['grayMeans'].shape[0]
    
    if fparams['VegetationIndex']['use']:
        if fparams['VegetationIndex']['NDVI']:
            features[i_feat] = np.mean( f_imgs['NDVI'][idx, idy] ); i_feat += 1 
        if fparams['VegetationIndex']['NNIR']:
            features[i_feat] = np.mean( f_imgs['NNIR'][idx, idy] ); i_feat += 1 
        if fparams['VegetationIndex']['NGREEN']:
            features[i_feat] = np.mean( f_imgs['NGREEN'][idx, idy] ); i_feat += 1 
        if fparams['VegetationIndex']['NRED']:
            features[i_feat] = np.mean( f_imgs['NRED'][idx, idy] ); i_feat += 1 
        if fparams['VegetationIndex']['PVI']:
            features[i_feat] = np.mean( f_imgs['PVI'][idx, idy] ); i_feat += 1 
    '''
    return features
    
def extractBlockFeatures(initX, endX, initY, endY, f_imgs, n_features, fparams):
    
    features = np.empty(n_features)  
    i_feat = 0
    
    if fparams['gray']['use']:
        if fparams['gray']['mean']:
            features[i_feat] = np.mean( f_imgs['gray'][initX:endX, initY:endY] ); i_feat += 1 
        if fparams['gray']['std']:
            features[i_feat] = np.std( f_imgs['gray'][initX:endX, initY:endY] ); i_feat += 1 
            
    if fparams['entropy']['use']:
        if fparams['entropy']['mean']:
            features[i_feat] = np.mean( f_imgs['entropy'][initX:endX, initY:endY] ); i_feat += 1 
        if fparams['entropy']['std']:
            features[i_feat] = np.std( f_imgs['entropy'][initX:endX, initY:endY] ); i_feat += 1 
            
    if fparams['RGB']['use']:
        for i in range(3):
            if fparams['RGB']['mean']:
                features[i_feat] = np.mean( f_imgs['RGB'][initX:endX, initY:endY, i] ); i_feat += 1 
            if fparams['RGB']['std']:
                features[i_feat] = np.std( f_imgs['RGB'][initX:endX, initY:endY, i] ); i_feat += 1 
                
    if fparams['LAB']['use']:
        for i in range(3):
            if fparams['LAB']['mean']:
                features[i_feat] = np.mean( f_imgs['LAB'][initX:endX, initY:endY, i] ); i_feat += 1 
            if fparams['LAB']['std']:
                features[i_feat] = np.std( f_imgs['LAB'][initX:endX, initY:endY, i] ); i_feat += 1
                
    if fparams['HSV']['use']:
        for i in range(3):
            if fparams['HSV']['mean']:
                features[i_feat] = np.mean( f_imgs['HSV'][initX:endX, initY:endY, i] ); i_feat += 1 
            if fparams['HSV']['std']:
                features[i_feat] = np.std( f_imgs['HSV'][initX:endX, initY:endY, i] ); i_feat += 1

    if fparams['YCbCr']['use']:
        for i in range(3):
            if fparams['YCbCr']['mean']:
                features[i_feat] = np.mean( f_imgs['YCbCr'][initX:endX, initY:endY, i] ); i_feat += 1 
            if fparams['YCbCr']['std']:
                features[i_feat] = np.std( f_imgs['YCbCr'][initX:endX, initY:endY, i] ); i_feat += 1

    if fparams['xyz']['use']:
        for i in range(3):
            if fparams['xyz']['mean']:
                features[i_feat] = np.mean( f_imgs['xyz'][initX:endX, initY:endY, i] ); i_feat += 1 
            if fparams['xyz']['std']:
                features[i_feat] = np.std( f_imgs['xyz'][initX:endX, initY:endY, i] ); i_feat += 1

    if fparams['yiq']['use']:
        for i in range(3):
            if fparams['yiq']['mean']:
                features[i_feat] = np.mean( f_imgs['yiq'][initX:endX, initY:endY, i] ); i_feat += 1 
            if fparams['yiq']['std']:
                features[i_feat] = np.std( f_imgs['yiq'][initX:endX, initY:endY, i] ); i_feat += 1

    if fparams['yuv']['use']:
        for i in range(3):
            if fparams['yuv']['mean']:
                features[i_feat] = np.mean( f_imgs['yuv'][initX:endX, initY:endY, i] ); i_feat += 1 
            if fparams['yuv']['std']:
                features[i_feat] = np.std( f_imgs['yuv'][initX:endX, initY:endY, i] ); i_feat += 1
            
    if fparams['LBP']['use']:
        n_bins = 2**fparams['LBP']['n_neibor']
        if fparams['LBP']['gray']:
            hist, _ = np.histogram( f_imgs['LBP_gray'][initX:endX, initY:endY] , bins=n_bins, range = (0,n_bins-1), density = True)
            features[i_feat:i_feat+n_bins] = hist; i_feat += n_bins
        if fparams['LBP']['red']:
            hist, _ = np.histogram( f_imgs['LBP_r'][initX:endX, initY:endY] , bins=n_bins, range = (0,n_bins-1), density = True)
            features[i_feat:i_feat+n_bins] = hist; i_feat += n_bins
        if fparams['LBP']['green']:
            hist, _ = np.histogram( f_imgs['LBP_g'][initX:endX, initY:endY] , bins=n_bins, range = (0,n_bins-1), density = True)
            features[i_feat:i_feat+n_bins] = hist; i_feat += n_bins
        if fparams['LBP']['blue']:
            hist, _ = np.histogram( f_imgs['LBP_b'][initX:endX, initY:endY] , bins=n_bins, range = (0,n_bins-1), density = True)
            features[i_feat:i_feat+n_bins] = hist; i_feat += n_bins
    #POR CAUSA DOS COMENTARIOS
    '''    
    if fparams['texton']['use']:
        if fparams['texton']['mean'] or fparams['texton']['std']:
            for k in range(fparams['texton']['n_kernels']):
                if fparams['texton']['mean']:
                    features[i_feat] = np.mean( f_imgs['texton'][initX:endX, initY:endY] ); i_feat += 1 
                if fparams['texton']['std']:
                    features[i_feat] = np.std( f_imgs['texton'][initX:endX, initY:endY] ); i_feat += 1 
        
        if fparams['texton']['hist_kmeans']:
            hist, _ = np.histogram( f_imgs['hist_kmeans'][initX:endX, initY:endY] , bins=fparams['texton']['hist_size_kmeans'], range = (0,fparams['texton']['hist_size_kmeans']-1), density = True)
            features[i_feat:i_feat+hist.shape[0]] = hist; i_feat += hist.shape[0]
            
        if fparams['texton']['hist_max']:
            hist, _ = np.histogram( f_imgs['hist_max'][initX:endX, initY:endY] , bins=fparams['texton']['n_kernels'], range = (0,fparams['texton']['n_kernels']-1), density = True)
            features[i_feat:i_feat+hist.shape[0]] = hist; i_feat += hist.shape[0]
    '''  
    '''
    if fparams['grayA']['use']:
        features[i_feat:i_feat+f_imgs['grayMeans'].shape[0]] =  f_imgs['grayMeans']; i_feat += f_imgs['grayMeans'].shape[0]
    '''
    '''    
    if fparams['grayD']['use']:
        features[i_feat:i_feat+f_imgs['grayMeans'].shape[0]] = features[0] - f_imgs['grayMeans']; i_feat += f_imgs['grayMeans'].shape[0]
    
    
    if fparams['VegetationIndex']['use']:
        if fparams['VegetationIndex']['NDVI']:
            features[i_feat] = np.mean( f_imgs['NDVI'][initX:endX, initY:endY] ); i_feat += 1 
        if fparams['VegetationIndex']['NNIR']:
            features[i_feat] = np.mean( f_imgs['NNIR'][initX:endX, initY:endY] ); i_feat += 1 
        if fparams['VegetationIndex']['NGREEN']:
            features[i_feat] = np.mean( f_imgs['NGREEN'][initX:endX, initY:endY] ); i_feat += 1 
        if fparams['VegetationIndex']['NRED']:
            features[i_feat] = np.mean( f_imgs['NRED'][initX:endX, initY:endY] ); i_feat += 1 
        if fparams['VegetationIndex']['PVI']:
            features[i_feat] = np.mean( f_imgs['PVI'][initX:endX, initY:endY]); i_feat += 1 
    '''      
    return features

def writeData(traindata, testdata, arff = False):
    ff = ''
    header = '@RELATION test\n\n'
    
    for i in range(traindata.shape[1]-1):
        ff = ff + '%.5f,'
        header += '@ATTRIBUTE att%d NUMERIC\n' % i
        
    ff = ff + '%d'
    header += '@ATTRIBUTE class        {0, 1}\n'
    header += '\n\n@DATA'
        
    
    if arff == False:
        np.savetxt('train_p.csv', traindata, fmt=ff,delimiter = ',')
        np.savetxt('test_p.csv', testdata, fmt=ff,delimiter = ',')
    else:
        np.savetxt('train_p.arff', traindata, fmt=ff,delimiter = ',', header = header, comments = '')
        np.savetxt('test_p.arff', testdata, fmt=ff,delimiter = ',', header = header, comments = '')

def paintImg(img, img_gt):
    
    assert (img.shape[0] == img_gt.shape[0] and img.shape[1] == img_gt.shape[1]), 'Img and img gt should have the same size'
    
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):  
            if img_gt[i, j] != 255:
                img[i, j, img_gt[i, j]] = 255


def loadKmeansTexton(train, fparams):
    kmeans = 0
  
    #COMENTADO POR CAUSA DAS 10 FEATURES
    '''
    if fparams['texton']['use'] and fparams['texton']['hist_kmeans']:
        if fparams['texton']['create_kmeans']:
            kmeans = createKmeansModel(train, 10)
            f = open('kmeans.pkl', 'wb')
            pickle.dump(kmeans, f)
            f.close()
        else:
            f = open('kmeans.pkl', 'rb')
            kmeans = pickle.load(f)
            f.close()
    '''
            
    return kmeans
   
def generateKernels():

    #filterdir = '/root/William/PRoad/sourcepy/filterbank/' #server
    filterdir = '/home/William/pyworkspace/sourcepy/filterbank/' #Linux
    kernels = []
    #    for theta in range(4):
    #        theta = theta / 4. * np.pi
    #        for sigma in (2, 4):
    #            for frequency in (0.10, 0.25):
    #                kernel = np.real(gabor_kernel(frequency, theta=theta,
    #                                              sigma_x=sigma, sigma_y=sigma))
    #                kernels.append(kernel)
    
    for i in range(1,16):
        kernels.append(np.genfromtxt(filterdir + 'filter' + str(i) + '.csv', delimiter = ','))
                
    return kernels

def generateGrayMeans(img_gray, i_div, j_div, gray_type):
    
    if gray_type == 1:
        
        grayMeans = np.empty(6)
        
        grayMeans[0] = np.mean(img_gray)
        grayMeans[1] = np.mean(img_gray[int(img_gray.shape[0]*0.5):,:])
        grayMeans[2] = np.mean(img_gray[0:int(img_gray.shape[0]*0.5),:])
        grayMeans[3] = np.mean(img_gray[:,0:int(img_gray.shape[1] * (1./3.))])
        grayMeans[4] = np.mean(img_gray[:,int(img_gray.shape[1] * (1./3.)) : int(img_gray.shape[1] * (2./3.))])
        grayMeans[5] = np.mean(img_gray[:,int(img_gray.shape[1] * (2./3.)) : ])
        
    elif gray_type == 2:
        
        grayMeans = np.empty(i_div*j_div + 1)
        
        pos_r = 0
        grayMeans[pos_r] = np.mean(img_gray)
        pos_r += 1
        
        i_range = int(float(img_gray.shape[0]) / i_div)
        j_range = int(float(img_gray.shape[1]) / j_div)
        
        for i in range(i_div):
            for j in range(j_div):
                
                grayMeans[pos_r] = np.mean(img_gray[i*i_range : i*i_range + i_range, j*j_range : j*j_range + j_range ])
                pos_r += 1
        
    return grayMeans
    
def createKmeansModel(train, n_imgs):
    
    kernels = generateKernels()
    
    train_array = np.empty((0, len(kernels)))
    
    for i in range(n_imgs):
        img = rgb2gray(data.imread(train[i]))
        img = img_as_float(img)
        
        ddata = np.empty((img.size, len(kernels)))
        
        for k, kernel in enumerate(kernels):
            filtered = nd.convolve(img, kernel)
            ddata [:, k] = filtered.flatten()
       
        train_array = np.vstack( (train_array, ddata) )
        
    miniKmeans = cluster.MiniBatchKMeans(n_clusters = 64, init = 'k-means++', random_state = 0)
    return miniKmeans.fit(train_array)
