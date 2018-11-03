# -*- coding: utf-8 -*-
import sys

from skimage import data
import numpy as np
from skimage import io
from skimage import exposure
import colorsys

from auxfunc import *

from skimage import color
from skimage import data

data_dir = 'data/' #Linux

nlist = []

for i in range(1,31):
	nlist.append(data_dir + "IMG_" + str(i) + '.jpg')

'''print "Generating imagens RGB to HSV..."
for i in range(len(nlist)):
	img = data.imread(nlist[i])
	img_hsv = color.rgb2hsv(img)
	io.imsave(data_dir + "hsv/" + "IMG_" + str(i + 1) + '.jpg', img_hsv)'''

'''print "Generating imagens RGB to GRAY..."
for i in range(len(nlist)):
	img = data.imread(nlist[i])
	img_gray = color.rgb2gray(img)
	io.imsave(data_dir + "gray/" + "IMG_" + str(i + 1) + '.jpg', img_gray)'''

print "Generating imagens RGB to LAB/YCbCr/YUV/HED/LCH/YIQ/YPbpr/XYZ/LBP..."
for i in range(len(nlist)):
	img = data.imread(nlist[i])
	#img = np.array(img, dtype=np.uint8)

	## RGB TO LAB
	#img_lab = color.rgb2lab(img)
	#img_lab = exposure.rescale_intensity(img_lab, out_range=(-1.0, 1.0))
	#io.imsave(data_dir + "lab/" + "IMG_" + str(i + 1) + '.jpg', img_lab)

	## RGB TO YCbCr
	#img_ycbcr = color.rgb2ycbcr(img)
	#img_ycbcr = exposure.rescale_intensity(img_ycbcr, out_range=(-1.0, 1.0))
	#io.imsave(data_dir + "ycbcr/" + "IMG_" + str(i + 1) + '.jpg', img_ycbcr)

	## RGB TO YUV
	#img_yuv = color.rgb2yuv(img)
	#io.imsave(data_dir + "yuv/" + "IMG_" + str(i + 1) + '.jpg', img_yuv)

	## RGB TO HED
	#img_hed = color.rgb2hed(img)
	#img_hed = exposure.rescale_intensity(img_hed, out_range=(-1.0, 1.0))
	#io.imsave(data_dir + "hed/" + "IMG_" + str(i + 1) + '.jpg', img_hed)

	## LAB TO LCH
	#img_lab = color.rgb2lab(img)
	#img_lch = color.lab2lch(img_lab)
	#img_lch = exposure.rescale_intensity(img_lch, out_range=(-1.0, 1.0))
	#io.imsave(data_dir + "lch/" + "IMG_" + str(i + 1) + '.jpg', img_lch)

	## RBG TO YIQ
	#img_yiq = color.rgb2yiq(img)
	#io.imsave(data_dir + "yiq/" + "IMG_" + str(i + 1) + '.jpg', img_yiq)

	## RBG TO YPbpr
	#img_ypbpr = color.rgb2ypbpr(img)
	#io.imsave(data_dir + "ypbpr/" + "IMG_" + str(i + 1) + '.jpg', img_ypbpr)

	## RGB TO XYZ
	#img_xyz = color.rgb2xyz(img)
	#img_xyz = exposure.rescale_intensity(img_xyz, out_range=(-1.0, 1.0))
	#io.imsave(data_dir + "xyz/" + "IMG_" + str(i + 1) + '.jpg', img_xyz)

	## Gray TO LBP
	#img_gray = color.rgb2gray(img)
	#img_lbp_gray = local_binary_pattern(img_gray, 4, 1)
	#img_lbp_gray = exposure.rescale_intensity(img_lbp_gray, out_range=(-1.0, 1.0))
	#io.imsave(data_dir + "lbp_gray/" + "IMG_" + str(i + 1) + '.jpg', img_lbp_gray)

	## RGB TO LBP
	#img_lbp = local_binary_pattern(img, 4, 1)
	#img_lbp = exposure.rescale_intensity(img_lbp, out_range=(-1.0, 1.0))
	#io.imsave(data_dir + "lbp/" + "IMG_" + str(i + 1) + '.jpg', img_lbp)

	print img