#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#  test2.py
#  
#  Copyright 2016 demo <demo@astoria.telin>
#  
#  This program is free software; you can redistribute it and/or modify
#  it under the terms of the GNU General Public License as published by
#  the Free Software Foundation; either version 2 of the License, or
#  (at your option) any later version.
#  
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
#  
#  You should have received a copy of the GNU General Public License
#  along with this program; if not, write to the Free Software
#  Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston,
#  MA 02110-1301, USA.
#  
#  

import Image
import numpy as np
import numpy.fft as fourier

# prints the image by opening a new tab
def showImage(args):
	test = args.show()
	#test.title('hello world')
	
	#plt.title('test')
	
# Converts matrix to image in order to show it
def printMatrix(matrix):
	image = Image.fromarray(matrix)
	showImage(image)

	
def im2mat(image):
	return np.asarray(image.convert('L'))
	
#based on fourier transform and only keep low frequency terms
def filter_1(mat_orig):
	a = fourier.rfft2(mat_orig,axes=(-2, -1))
	
	print(a.shape)
	
	size = len(a)

	
	
	b = fourier.irfft2(a)
	print(len(b))
	return b
	
def filter_x(mat_orig):
	#amount of low frequency terms to keep
	bound = 2;
	size = mat_orig.shape
	print(size)
	
	mat_new = np.zeros(size)
	
	for i in range(0,size[0]):
		row = mat_orig[i,:]
		a = fourier.rfft(row)
		a[0:25] = 0	#filter out low frequencies
		b = fourier.irfft(a)
		mat_new[i,:] = b 
		
		
	return mat_new

	
#fft2(a[, s, axes, norm]) 	Compute the 2-dimensional discrete Fourier Transform
#ifft2(a[, s, axes, norm]) 	Compute the 2-dimensional inverse discrete Fourier Transform.
	

#First program
def test1():
	filename = 'Lenna.png'
	image_orig = Image.open(filename)
	showImage(image_orig)
	mat_grey = im2mat(image_orig)
	mat_rgba = np.array(image_orig) #a or alpha is transparancy?
	#red, green and blue part
	mat_r = mat_rgba[:,:,0]
	mat_g = mat_rgba[:,:,1]
	mat_b = mat_rgba[:,:,2]
	
	#printMatrix(mat_grey)
	#printMatrix(mat_rgba)

	fil1 = filter_1(mat_grey)
   	printMatrix(fil1)

	fil2 =filter_x(mat_grey)
	printMatrix(fil2)
	
	
	
#Starts main when program is runned
def main(args):
	test1()
	

#first line that is runned (others are just def's
if __name__ == '__main__':
	
    import sys
    sys.exit(main(sys.argv))
    

    





