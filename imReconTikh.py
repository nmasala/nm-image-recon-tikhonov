# -*- coding: utf-8 -*-
"""
@author: Nemanja Masala
"""

# IMPORTS
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import PIL as pil
import os as os


class myTikhImRegDemo(object):
    '''
    A class for playing around with implementing Tikhonov-regularized image 
    reconstructions
    '''
    '''
    @staticmethod()   
    def displayImage():'''
        
    
    
    def __init__( self, arrIm, noiseSd=10, lambd=0.01, regType='laplacian' ):
        self.im = np.float64(arrIm)
        self.nativeSize = arrIm.shape
        self.numEl = np.prod( self.nativeSize )
        
        self.imNoisy = np.zeros( self.nativeSize )
                
        self.gaussianSD = noiseSd
        self.lambd = lambd
        self.regType = regType
        
    def computeNoisyIm(self):
        # Guassian blur and Gaussian noise
        self.imNoisy = sp.ndimage.gaussian_filter(self.im, sigma=.5)
        self.imNoisy = self.imNoisy \
            + np.random.normal(0, self.gaussianSD, self.im.shape)
        
    def vectorize(self, arrIm):
        # stack column-wise
        # this outputs a row vector, not a coumn vector
        return arrIm.flatten('F')
        
    def unvectorize(self, arrIm):
        return np.reshape( arrIm, self.nativeSize, 'F' )
        
    def computeH(self):
        self.H = np.eye( self.numEl )
        
    def computeL(self):
        if self.regType == 'l2base':
            self.L = np.eye(self.numEl)
            
        elif self.regType == 'laplacian':
            self.L = np.zeros([self.numEl, self.numEl])
            
            # define base kernel
            laplacianTempl = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]])
            delta = np.intp(laplacianTempl.shape[1] / 2)
            
            # populate each row of this matrix with vectorized Laplacian kernel
            for ii in range(0, self.numEl):
                arrtmp = np.zeros(self.nativeSize)
                
                # get the 2D index
                ir, ic = np.unravel_index(ii, self.nativeSize, 'F')
                
                # set initial estimates for array ranges, to be used to jointly
                # define which part of kernel we want and which part of arrtmp 
                # it will be written into
                rStart = -delta
                rEnd = delta
                cStart = -delta
                cEnd = delta
                
                # refine array ranges if any are out of bounds
                if (ir + rStart) < 0:
                    rStart = -ir
                if (ir + rEnd) > (arrtmp.shape[0] - 1):
                    rEnd = arrtmp.shape[0] - 1 - ir
                if (ic + cStart) < 0:
                    cStart = -ic
                if (ic + cEnd) > (arrtmp.shape[1] - 1):
                    cEnd = arrtmp.shape[1] - 1 - ic
                
                # assign to arrtmp
                # Note: the '+ 1' is because of Python indexing
                arrtmp[(ir + rStart):(ir + rEnd + 1), \
                       (ic + cStart):(ic + cEnd + 1)] \
                    = laplacianTempl[(delta + rStart):(delta + rEnd + 1), \
                                     (delta + cStart):(delta + cEnd + 1)]
                        
                self.L[ii, :] = self.vectorize(arrtmp)
                # Note that it seems to not matter whether the vector
                # is a row or column vector for this kind of assignment
                
        else:
            quit()
            
    def computeLargeMtx(self):
        
        self.largeMtx = np.dot(self.H.transpose(), self.H) + \
            self.lambd * np.dot(self.L.transpose(), self.L)
                
    def prep(self):
        '''
        Compute what needs to be computed in preparation for calculating 
        solution

        Returns
        -------
        None
        '''
        
        if np.all( self.imNoisy == 0 ):
            self.computeNoisyIm()
            
        if len(self.imNoisy.shape) > 1:
            if self.imNoisy.shape[1] > 1:
                self.imNoisy = self.vectorize(self.imNoisy)
                
        self.computeH()
        self.computeL()
        self.computeLargeMtx()
        
    def computeDirectSoln(self):
        '''
        Compute direct solution

        Returns
        -------
        regularized image
        '''
        
        self.prep()
        
        Rlambda = np.linalg.inv( self.largeMtx )
        Rlambda = np.dot(Rlambda, self.H.transpose())
        imDirect = np.dot(Rlambda, self.imNoisy)
        
        self.imNoisy = self.unvectorize(self.imNoisy)
        imDirect = self.unvectorize(imDirect)
        
        # plot
        f1 = plt.figure()
        ax1 = f1.add_subplot(1, 3, 1)
        ax1.imshow(self.imNoisy, cmap='Greys')
        ax2 = f1.add_subplot(1, 3, 2)
        ax2.imshow(imDirect, cmap='Greys')
        ax3 = f1.add_subplot(1, 3, 3)
        ax3.imshow(self.im, cmap='Greys')
        plt.show()
        
        return imDirect
            
    def demoIterSoln(self, nIter=10):
        '''
        Run iterative solution with animation
        
        NOT FINISHED
        '''
        
        self.prep()
        
        sOld = self.vectorize(self.imNoisy)
        
        # set-up plot
        f1 = plt.figure()
        ax1 = f1.add_subplot(1, 3, 1)
        ax1.imshow(self.imNoisy, cmap='Greys')
        ax2 = f1.add_subplot(1, 3, 2)
        ax3 = f1.add_subplot(1, 3, 3)
        ax3.imshow(self.im, cmap='Greys')
        plt.show()
        
        for ii in range(0, nIter):
            imtmp = self.runOneStepIter(sOld)
            
            # update plot
            ax2.imshow(imtmp)
            plt.draw()
            
        return imtmp
        
    def runOneStepIter(self, s, gamma=0.001):
        ''

# veriables
mainDir = r''
imgFilename = 'walkbridge.tiff'

imPath = r'{}\{}'.format(mainDir, imgFilename)

# check that file exists
os.path.isfile( imPath )

# read in image
imageIm = pil.Image.open( imPath )

# get some info and display
print('Image dimensions are: {:d} x {:d}' \
      .format(imageIm.size[0], imageIm.size[1]) )
#imageIm.show() # pops up image in Windows dialog box
f0 = plt.figure()
ax = f0.add_subplot(1, 2, 1)
ax.imshow(np.array( imageIm ), cmap='Greys')

# crop to have reasonable array sizes for Tikhonov regularizer
cropScalar = 0.15
imageIm = imageIm.crop([100, 100, 100+int(cropScalar * imageIm.size[0]), \
                        100+int(cropScalar * imageIm.size[1])])

# get some info and display
print('Image dimensions are: {:d} x {:d}' \
      .format(imageIm.size[0], imageIm.size[1]) )
#imageIm.show()
ax = f0.add_subplot(1, 2, 2)
ax.imshow(np.array( imageIm ), cmap='Greys')

# convert to array
arrIm = np.array( imageIm )

# initialize
tikhObj = myTikhImRegDemo(arrIm, 10)

tikhObj.computeDirectSoln()

tikhObj.lambd = .1
tikhObj.computeDirectSoln()