# -*- coding: utf-8 -*-
import sys

from skimage import data
import numpy as np
from skimage import io

from auxfunc import *

from sklearn import svm
from sklearn import preprocessing
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
import time
import matplotlib.pyplot as plt
import pickle
from sklearn.linear_model import LogisticRegression
try:
    from nolearn.dbn import DBN
except ImportError:
    #modificado
    print ('Cant import DBN!')    
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cross_validation import ShuffleSplit
from sklearn.grid_search import GridSearchCV
#import featuresNBlock
import featuresCBBlock
#import featuresSegCB
import classesAnimalsRoads

##features
fparams = dict()
fparams['RGB'] = {'use': False, 'mean': True, 'std': True}
fparams['LAB'] = {'use': False, 'mean': True, 'std': True}
fparams['HSV'] = {'use': False, 'mean': True, 'std': True}
fparams['YCbCr'] = {'use': False, 'mean': True, 'std': True}
fparams['xyz'] = {'use': False, 'mean': True, 'std': True}
fparams['yiq'] = {'use': False, 'mean': True, 'std': True}
fparams['yuv'] = {'use': False, 'mean': True, 'std': True}
fparams['gray'] = {'use': False, 'mean':True, 'std':True} 
fparams['entropy'] = {'use': False, 'mean':True, 'std':True}
fparams['LBP'] = {'use': True, 'gray': True, 'red': True, 'green': True, 'blue': True, 'n_neibor': 4, 'radius': 1} ##
# fparams['grayA'] = {'use': False}
##fparams['grayD'] = {'use': False}
##fparams['pos'] = {'use': False}
##fparams['bottom'] = {'use': False}
##fparams['VegetationIndex'] = {'use': False, 'NDVI': False, 'NNIR': False, 'NRED': False, 'NGREEN': False, 'PVI': False}
##fparams['texton'] = {'use': False, 'mean': True, 'std': True, 'n_kernels': 15, 'hist_max': True, 'hist_kmeans': False, 'create_kmeans': False, 'hist_size_kmeans': 64}
fparams['perc'] = 0.98 #escolhe a porcentagem da classe
fparams['problem'] = classesAnimalsRoads #fparams['problem'] = classesHealthyPests
fparams['save_uncertainty'] = False
fparams['grid_search'] = False

# dicionario de features  
activeFeatures = dict()
contImgs = 1

activeFeatures['RGB'] = {'use': False, 'mean': True, 'std': True} #1
activeFeatures['LAB'] = {'use': False, 'mean': True, 'std': True} #2
activeFeatures['HSV'] = {'use': False, 'mean': True, 'std': True} #3
activeFeatures['YCbCr'] = {'use': False, 'mean': True, 'std': True} #4
activeFeatures['xyz'] = {'use': False, 'mean': True, 'std': True} #5
activeFeatures['yiq'] = {'use': False, 'mean': True, 'std': True} #6
activeFeatures['yuv'] = {'use': False, 'mean': True, 'std': True} #7
activeFeatures['gray'] = {'use': False, 'mean':True, 'std':True} #8
activeFeatures['entropy'] = {'use': False, 'mean':True, 'std':True} #9
activeFeatures['LBP'] = {'use': True, 'gray': True, 'red': True, 'green': True, 'blue': True, 'n_neibor': 4, 'radius': 1} #10
# activeFeatures['grayA'] = {'use': False} #11
#activeFeatures['grayD'] = {'use': False} #12
#activeFeatures['pos'] = {'use': False} #13
#activeFeatures['bottom'] = {'use': False} #14
#activeFeatures['VegetationIndex'] = {'use': False, 'NDVI': False, 'NNIR': False, 'NRED': False, 'NGREEN': False, 'PVI': False} #15
#activeFeatures['texton'] = {'use': False, 'mean': True, 'std': True, 'n_kernels': 15, 'hist_max': True, 'hist_kmeans': False, 'create_kmeans': False, 'hist_size_kmeans': 64} #16

#fparams['path'] = featuresNBlock
#fparams['pathConfig'] = {'blockSize': 15, 'n_area': 1}

#fparams['path'] = featuresSegCB
#fparams['pathConfig'] = {'marginSize': 5, 'nSegs': 3000}

fparams['path'] = featuresCBBlock
fparams['pathConfig'] = {'blockSizeIn': 5, 'blockSizeOut': 15, 'maxBlockSize': 15} # OBS1: alteração de valores

#print (fparams)   #modificado
subsample_p = 1 #subsamples from training set

train_models = { 'lSVM': False, 'RF': True, 'MLP': False, 'DBN': False, 'KNN': False, 'SVC': False, 'LR': False, "ANN": False, 'AdaBoost': False}

write_CSV = False

show_imgs = True # show test images
save_classi = True # save classified images
show_training_imgs = False #show training images 

perblock_features, feature_size = calculateFeatureSize(fparams)

##/features

print ('Features per blocks ' + str(perblock_features) + ', features per sample ' + str(feature_size)) #modificado
sys.stdout.flush()

##load and feature extraction

train, test, train_gt, test_gt , imgs2show = fparams['problem'].loadTrainData()

## Create Kmeans model for textons
kmeans = loadKmeansTexton(train, fparams)

if np.sum(np.asarray(train_models.values())) > 0:

    start = time.clock()
    traindata = np.empty(shape=[0, feature_size])
    testdata = np.empty(shape=[0, feature_size])

    for i in range(len(train)):
        print ('Loading img for train: %s' % train[i]) #modificado
        img = data.imread(train[i])
        img_gt = data.imread(train_gt[i])

        if img_gt.ndim == 3:
            img_gt = img_gt[:,:,0]

        traindata = np.append(traindata, fparams['path'].extractImageFeatures(img, img_gt, kmeans, fparams), axis=0)

    for i in range(len(test)):
        print ('Loading img for test: %s' % test[i]) #modificado
        img = data.imread(test[i])

        img_gt = data.imread(test_gt[i])
        if img_gt.ndim == 3:
            img_gt = img_gt[:,:,0]

        testdata = np.append(testdata, fparams['path'].extractImageFeatures(img, img_gt, kmeans, fparams), axis=0)

    print('Time to load imgs ' + str(time.clock() - start) )
    sys.stdout.flush()
    ##/load and feature extraction


    np.random.seed(42)
    random_indices = np.random.choice(traindata.shape[0], int(traindata.shape[0]*subsample_p), False) #subsample training data
    traindata = traindata[random_indices, :]

    # TODO: subamostragem dos dados que não são pragas (remover alguns linhas da matriz) verificar a quantidade de amostras das duas classes.

    train_y = traindata[:,-1];
    train_x = traindata[:,:-1];

    scaler = preprocessing.StandardScaler().fit(train_x);
    train_x = scaler.transform(train_x);

    if len(test) != 0:
        test_y = testdata[:,-1];
        test_x = testdata[:,:-1];
        test_x = scaler.transform(test_x);
    else:
        test_x = np.copy(train_x)
        test_y = np.copy(train_y)


if write_CSV:
    writeData(np.hstack( (train_x, train_y.reshape((train_y.shape[0], 1))) ), np.hstack( (test_x, test_y.reshape((test_y.shape[0], 1))) ), arff = True)
    #writeData(traindata, testdata, arff = True)


models = []
if train_models['lSVM']:
    models.append(svm.LinearSVC(C=0.01,random_state=0))
if train_models['RF']:
    models.append(RandomForestClassifier(n_estimators=100, verbose=1, random_state=0, n_jobs = 3))
if train_models['KNN']:
    models.append(KNeighborsClassifier(n_neighbors=3))
if train_models['SVC']:
    models.append(svm.SVC(kernel='rbf'))
if train_models['LR']:
    models.append(LogisticRegression(C=100))
if train_models['ANN']:
    output_layer = fparams['problem'].getNClass() #2classes
    YOneHot = np.zeros([len(train_y),output_layer])
    for i in range(output_layer):
        YOneHot[:,i] = np.where(train_y==i,1,0)
    models.append(NeuralNetworkGPU.NeuralNetworkGPU(layer_shape = [train_x.shape[1],800,800,output_layer],dropout_probability = [0.2,0.5,0.5,0.0], n_epochs = 3, l2_max = 15.0))
if train_models['AdaBoost']:
    models.append(AdaBoostClassifier(n_estimators=100))


bestModel = dict()
bestModel['acc'] = 0

param_LR = {'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000] }
params_SVM_radial = {'C': [1, 10, 100, 1000], 'gamma': [0.1, 0.01, 0.001, 0.0001]}

if fparams['grid_search']:
    for model in models:
        if (type(model) is svm.SVC):
            print ("GridSearch SVM") #modificado
            models[models.index(model)] = GridSearchCV(model, params_SVM_radial, cv = ShuffleSplit(len(train_y), 1, test_size=0.3), verbose = 10, n_jobs=3)
        if (type(model) is LogisticRegression) or (type(model) is svm.LinearSVC):
            print ("GridSearch" + type(model)) #modificado
            models[models.index(model)] = GridSearchCV(model, param_LR, cv = ShuffleSplit(len(train_y), 1, test_size=0.3), verbose = 10, n_jobs=3)


for model in models:

    start = time.clock()
    print ('Training ' + str(model)) #modificado
    if train_models['ANN']:
        model.fit(train_x, YOneHot)
    else:
        model.fit(train_x, train_y)
    #print('Time to train ' + str(model) + ' ' + str(time.clock() - start))
    pr_t = model.predict(test_x)
    acc = accuracy_score(test_y, pr_t)
    print ('Model accuracy: %f' % acc) #modificado
    #Evaluate model based on class 1 [0,1]
    evaluateModel(test, test_gt, model, scaler, kmeans, fparams, activeFeatures, contImgs, colorImg = True)

    if acc > bestModel['acc']:
        bestModel['acc'] = acc
        bestModel['model'] = model


if np.sum(np.asarray(train_models.values())) > 0:
    
    print ('Best model: ' + str(bestModel['model'])) #modificado

    model = bestModel['model']
    if train_models['ANN']:
        print ("Can't save model") #modificado
    else:
        f = open('model.pkl', 'wb')
        pickle.dump(model, f)
        f.close()
    
        f = open('scaler.pkl', 'wb')
        pickle.dump(scaler, f)
        f.close()
else:

    if show_imgs:

        try:
            if train_models['ANN']:
                print ("ANN can't open model") #modificado
                IOError
            else:
                f = open('model.pkl', 'rb')
                model = pickle.load(f)
                f.close()
    
                f = open('scaler.pkl', 'rb')
                scaler = pickle.load(f)
                f.close()

        except IOError:
            print ('You need to train before testing...') #modificado

if show_imgs:

    for imgName in (test + imgs2show):
        img = data.imread(imgName)

        print ('Showing img: ' + imgName) #modificado

        if fparams['pathConfig'].has_key('marginSize'):
            img_r, img = fparams['path'].classifyImg(model, scaler, img, kmeans, fparams)
        else:
            img_r = fparams['path'].classifyImg(model, scaler, img, kmeans, fparams)

        paintImg(img, img_r)
        #plt.figure()
        #plt.imshow(img)
        #plt.show()

        if save_classi:
            
            #SALVANDO IMAGENS
            io.imsave(imgName.split('/')[-1] + 'c.png', img)
            # io.imsave(imgName.split('/')[-1] + '_binary.png', img_r)


            if fparams['save_uncertainty']:
                if (type(model) is LogisticRegression):
                    if (fparams['pathConfig'].has_key('marginSize')):
                        img = data.imread(imgName) #reload original image
                    img_proba_sum, img_proba_zero, img_proba_one = fparams['path'].predictProbabilitesLR(model, scaler, img, kmeans, fparams)
                    print("Saving uncertainty from LR") #modificado
                    #np.savetxt(imgName.split('/')[-1]+'.txt', img_proba_sum,fmt='%1.3f',delimiter='\t')
                    #np.savetxt(imgName.split('/')[-1]+'proba_zero.txt', img_proba_zero,fmt='%1.3f',delimiter='\t')
                    #np.savetxt(imgName.split('/')[-1]+'proba_one.txt', img_proba_one,fmt='%1.3f',delimiter='\t')
                plt.imsave(imgName.split('/')[-1]+'uncertainty.png',-img_proba_sum)


if show_training_imgs:

    for i in range(len(train)):
        img = data.imread(train[i])
        img_gt = data.imread(train_gt[i])
        if img_gt.ndim == 3:
            img_gt = img_gt[:,:,0]

        if fparams['pathConfig'].has_key('marginSize'):
            img_c, img = fparams['path'].classifyTrainingImg(img, img_gt, fparams)
        else:
            img_c = fparams['path'].classifyTrainingImg(img, img_gt, fparams)

        paintImg(img, img_c)
        print(train[i])
        #plt.imshow(img)
        #plt.show()
        if save_classi:
            io.imsave(train[i].split('/')[-1] + 't.png', img)
        #plt.waitforbuttonpress()
