# -*- coding: utf-8 -*-
import numpy as np
import sys
import os

data_dir = 'data/'  # Windows


def getNClass():
    return 2


def image2class(img):

    i0, j0 = np.where(img == 255)  # animals
    i1, j1 = np.where(img == 127)  # roads
    i2, j2 = np.where(np.logical_and(img != 127, img != 255))
    img[i1, j1] = 0
    img[i0, j0] = 1  # class 1 is the one that is evaluated (f-measure)
    img[i2, j2] = 255
    
    return img


def pathClassRate(img):
    # Convert the input to an array.
    return np.asarray([float(np.sum(img == 0)) / img.size, float(np.sum(img == 1)) / img.size])


def loadTrainData():
    train = []
    train_gt = []
    test = []
    test_gt = []
    imgs2show = []

    '''
    ## OBSERVAÇÃO: Após analisar as imagens geradas no treinamento, utilizando o extrator "context block" 
    selecionamos uma amostra com dez imagens (que tiveram os melhores resultados), 
    sendo que seis para a etapa de treinamento (equivalente a 2/3) e 4 para testes (equivalente a 1/3):
    '''

    train.append(data_dir + 'IMG_01.jpg')
    train_gt.append(data_dir + 'IMG_01_Labeled.jpg')

    train.append(data_dir + 'IMG_02.jpg')
    train_gt.append(data_dir + 'IMG_02_Labeled.jpg')

    train.append(data_dir + 'IMG_03.jpg')
    train_gt.append(data_dir + 'IMG_03_Labeled.jpg')

    train.append(data_dir + 'IMG_04.jpg')
    train_gt.append(data_dir + 'IMG_04_Labeled.jpg')

    train.append(data_dir + 'IMG_05.jpg')
    train_gt.append(data_dir + 'IMG_05_Labeled.jpg')

    train.append(data_dir + 'IMG_06.jpg')
    train_gt.append(data_dir + 'IMG_06_Labeled.jpg')

    train.append(data_dir + 'IMG_07.jpg')
    train_gt.append(data_dir + 'IMG_07_Labeled.jpg')

    train.append(data_dir + 'IMG_08.jpg')
    train_gt.append(data_dir + 'IMG_08_Labeled.jpg')

    train.append(data_dir + 'IMG_09.jpg')
    train_gt.append(data_dir + 'IMG_09_Labeled.jpg')

    train.append(data_dir + 'IMG_10.jpg')
    train_gt.append(data_dir + 'IMG_10_Labeled.jpg')

    train.append(data_dir + 'IMG_11.jpg')
    train_gt.append(data_dir + 'IMG_11_Labeled.jpg')

    train.append(data_dir + 'IMG_12.jpg')
    train_gt.append(data_dir + 'IMG_12_Labeled.jpg')

    train.append(data_dir + 'IMG_13.jpg')
    train_gt.append(data_dir + 'IMG_13_Labeled.jpg')

    train.append(data_dir + 'IMG_14.jpg')
    train_gt.append(data_dir + 'IMG_14_Labeled.jpg')

    train.append(data_dir + 'IMG_15.jpg')
    train_gt.append(data_dir + 'IMG_15_Labeled.jpg')

    train.append(data_dir + 'IMG_16.jpg')
    train_gt.append(data_dir + 'IMG_16_Labeled.jpg')

    train.append(data_dir + 'IMG_17.jpg')
    train_gt.append(data_dir + 'IMG_17_Labeled.jpg')

    train.append(data_dir + 'IMG_18.jpg')
    train_gt.append(data_dir + 'IMG_18_Labeled.jpg')

    train.append(data_dir + 'IMG_19.jpg')
    train_gt.append(data_dir + 'IMG_19_Labeled.jpg')

    train.append(data_dir + 'IMG_20.jpg')
    train_gt.append(data_dir + 'IMG_20_Labeled.jpg')

    train.append(data_dir + 'IMG_21.jpg')
    train_gt.append(data_dir + 'IMG_21_Labeled.jpg')

    train.append(data_dir + 'IMG_22.jpg')
    train_gt.append(data_dir + 'IMG_22_Labeled.jpg')

    train.append(data_dir + 'IMG_23.jpg')
    train_gt.append(data_dir + 'IMG_23_Labeled.jpg')

    train.append(data_dir + 'IMG_24.jpg')
    train_gt.append(data_dir + 'IMG_24_Labeled.jpg')

    train.append(data_dir + 'IMG_25.jpg')
    train_gt.append(data_dir + 'IMG_25_Labeled.jpg')

    train.append(data_dir + 'IMG_26.jpg')
    train_gt.append(data_dir + 'IMG_26_Labeled.jpg')

    train.append(data_dir + 'IMG_27.jpg')
    train_gt.append(data_dir + 'IMG_27_Labeled.jpg')

    train.append(data_dir + 'IMG_28.jpg')
    train_gt.append(data_dir + 'IMG_28_Labeled.jpg')

    train.append(data_dir + 'IMG_29.jpg')
    train_gt.append(data_dir + 'IMG_29_Labeled.jpg')

    train.append(data_dir + 'IMG_30.jpg')
    train_gt.append(data_dir + 'IMG_30_Labeled.jpg')

    train.append(data_dir + 'IMG_31.jpg')
    train_gt.append(data_dir + 'IMG_31_Labeled.jpg')

    # img teste TESTE

    test.append(data_dir + 'IMG_32.jpg')
    test_gt.append(data_dir + 'IMG_32_Labeled.jpg')

    test.append(data_dir + 'IMG_33.jpg')
    test_gt.append(data_dir + 'IMG_33_Labeled.jpg')

    test.append(data_dir + 'IMG_34.jpg')
    test_gt.append(data_dir + 'IMG_34_Labeled.jpg')

    test.append(data_dir + 'IMG_35.jpg')
    test_gt.append(data_dir + 'IMG_35_Labeled.jpg')

    test.append(data_dir + 'IMG_36.jpg')
    test_gt.append(data_dir + 'IMG_36_Labeled.jpg')

    return (train, test, train_gt, test_gt, imgs2show)
