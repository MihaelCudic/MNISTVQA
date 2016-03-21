"""
Create online database for question/answer on a canvas of MNIST images

@author: Mihael Cudic 
"""
import numpy as np
from scipy.misc import imresize
from PIL import Image

import random
random.seed()

from keras.datasets import mnist

class Data(object):
    
    def __init__(self,
                 X, Y, batchSize,
                 xCanDim = 64,
                 yCanDim = 64,                 
                 canDim = [],
                 border = [],
                 xBorder = 0,
                 yBorder = 0,
                 nImgs = [],
                 minNImgs = 1,
                 maxNImgs = 2,
                 randNImgs = False,
                 scaling = [],
                 minScale = 1,
                 maxScale = 1,
                 minImgSep = None,
                 ):
        """
        X - input data
        Y - classification data assigned to input data
        batchSize - size of batches
        
        xCanDim - X dimension of canvas (Default: 64)
        yCanDim - y dimension of canvas (Default: 64)        
        xBorder - inside horizontal border (Default: 0)
        yBorder - inside vertical border (Default: 0)
        minNImgs - minimum number of images pasted onto a canvass (Default: 1)
        maxNImgs - maximum number of images pasted onto a canvass (Default: 2)
        randNIms - Default False: paste maxNImgs images on canvas; 
                   True: randomly select number of images to paste on canvas
        minScale - minimum scaling of image (Default: 1)
        maxScale - maximum scaling of image (Default: 1)
        minImgSep - minimum half distance between centers of images pasted onto a 
                    canvas (Default: max(imageDimensions)/2)        
        
        USE ARRAYS To Simplify initializations
        canDim - [xCanDim, yCanDim]
        border - [xBorder, yBorder]
        nImgs - [minNImgs, maxNImgs, randNImgs]
        scaling - [minScale, maxScale]
        """
        
        # Store attributes of input data
        self.X = X
        self.Y = Y
        self.batchSize = batchSize
        
        # Set dimensions of input images
        self.imgDim = [len(X[1][0][0]), len(X[1][0])]

        # Set dimensions of canvas       
        self.canDim = [xCanDim, yCanDim] if canDim == [] else canDim        
        
        # Set borders inside canvas       
        self.xBorder, self.yBorder = [xBorder, yBorder] if border == [] else border 
        
        # Set the attribute regarding number of images pasted to canvas
        self.minNImgs, self.maxNImgs, self.randNImgs = [minNImgs, maxNImgs, randNImgs] if nImgs == [] else nImgs       
        
        # Set the the scaling of the pasted images
        self.minScale, self.maxScale = [minScale, maxScale] if scaling == [] else scaling
        
        # Set seperation of pasted images
        self.minImgSep = np.max(self.imgDim)/2 if minImgSep is None else minImgSep   
    
    def get_batch(self):
        """
        Create batch of input data
        INPUT: None
        OUTPUT:
            canvas - 3d matrix of images (Note: [image number, :, :] == input canvas)
            question - question corresponding to canvas (Specific to MNIST)
            answer - answer to question
        """
        # Preallocate space for output variables
        canvas = np.zeros([self.batchSize, self.canDim[0], self.canDim[1]])
        question = self.batchSize*[None]
        answer = self.batchSize*[None]        
        
        # List of possible functions called
        qFunct = [self.containsA, self.containsOdd, self.totalSum, self.totalProduct, 
                  self.xMost, self.xNumerical, self.xSize, self.nSubImages]    
        
        # Generate input data for batch
        for i in range(self.batchSize):         
            canvas[i,:,:], subImgs, subClasses, subScales, subStrts, subCenters = self.createCanvas()
            question[i], answer[i] = random.choice(qFunct)(subStrts, subCenters, subScales, subClasses)
            
        return canvas, question, answer
    
    def createCanvas(self):
        """
        Create new canvas give specs assigned previously
        INPUT: None
        OUTPUT:
            canvas - 2d matrix of pixel values
            subImgs - 2d matrix of input images with non-scaled dimensions
            subClasses - classes of input images in canvas
            subScales - scaling of every input image on canvas
            subStrts - starting [x, y] indeces of input images on canvas
            subCenters - centered [x, y] indeces of input images on canvas
        """
        # Create canvas
        canvas = np.zeros(self.canDim)                             
        
        # Decide how many images will be pasted
        nImgs = self.maxNImgs if not self.randNImgs else random.randint(self.minNImgs, self.maxNImgs)
        
        # Initialize specs of canvas
        subImgs = np.zeros([nImgs, self.imgDim[0], self.imgDim[1]]) # 2d matrix of input images with non-scaled dimensions
        subClasses = np.zeros([nImgs, 1]) # classes of input images in canvas
        subScales = np.zeros([nImgs, 1]) # scaling of every input image on canvas
        subStrts = np.zeros([nImgs, 2]) # starting [x, y] indeces of input images on canvas
        subCenters = np.zeros([nImgs, 2]) # centered [x, y] indeces of input images on canvas
        
        # Iterate and paste image
        for i in range(nImgs):
            # Choose random image from input
            indX = random.randint(0,(len(self.X)-1)) 
            subImgs[i,:,:] = np.squeeze(self.X[indX][None][:][:])
            subClasses[i] = self.Y[indX]
            
            # Add image to canvas specs
            subScales[i], subStrts[i,:], subCenters[i,:] = self.createSubImage(i,
                                                                               subScales, 
                                                                               subStrts, 
                                                                               subCenters)
            
            # Calculate new dimensions
            newImgDim = np.round(np.multiply(subScales[i],self.imgDim)).astype(int)
            
            # Resize image to new dimensions
            img = imresize(subImgs[i,:,:], newImgDim)
            img = img.astype('float32')
            img /= 255
            
            # Find start and end points of image projected onto canvas
            xStrt, yStrt = subStrts[i,:].astype(int)
            xEnd, yEnd = np.add([xStrt, yStrt],newImgDim).astype(int)
            
            # Restrict strt and end points to not index out of bounds
            # NOTE: I think this can be simplified
            xImgStrt, yImgStrt = [0,0]
            xImgEnd, yImgEnd = newImgDim
            if xStrt < 0:
                xImgStrt = -xStrt
                xStrt = 0
            if yStrt < 0:
                yImgStrt = -yStrt
                yStrt = 0
            if xEnd > self.canDim[0]:
                xImgEnd = newImgDim[0]-(xEnd-self.canDim[0])
                xEnd = self.canDim[0]
            if yEnd > self.canDim[1]:
                yImgEnd = newImgDim[1]-(yEnd-self.canDim[1])
                yEnd = self.canDim[1]
            
            # Paste image to canvas
            canvas[yStrt:yEnd,xStrt:xEnd] = np.add(canvas[yStrt:yEnd,xStrt:xEnd],img[yImgStrt:yImgEnd,xImgStrt:xImgEnd])
        
        # Cap canvas at 1
        # NOTEL I think this can be done faster
        canvas[canvas>1] = 1 
        
        return canvas, subImgs, subClasses, subScales, subStrts, subCenters
                
    def createSubImage(self, nSubs, subScales, subStrts, subCenters):  
        """
        Create subimage to paste into canvas
        INPUT: 
            nSubs - current number of subImages on the canvas
            subScales - scaling of the subImages on the canvas
            subStrts - starting indeces of the subImages on the canvas
            subCenters - center points of the subImages on the canvas
        OUTPUT:
            scale - scaling of the new subImage
            strt - starting indeces of the new subImage
            center - center points of new subImage
        """
        
        # Initialize possible starting locations for the subImage (Allow adjustment due to border restrictions)
        # 1 means that image can fit given its conditions
        dim = [x-2*y for (x,y) in zip(self.canDim,[self.xBorder, self.yBorder]) ]
        fitCanvas = np.ones(dim)    
        
        # Create new scaling of subImage
        scale = random.uniform(self.minScale, self.maxScale)
        newImgDim = np.round(np.multiply(scale,self.imgDim))
        halfImgDim = [(x-1)/2 for x in newImgDim ] # used for determining center points of image
        
        # Adjust possible starting locations for new subImage
        # Apply size resitrictions (image isn't placed where it will index out of position)
        fitCanvas[-newImgDim[0]:,:] = 0 
        fitCanvas[:,-newImgDim[1]:] = 0 
        
        # Pick starting indeces noting overlap conditions
        strt = self.addMinSep(nSubs, subScales, subCenters, fitCanvas, scale, halfImgDim)

        # Calculate center points
        center = np.add(strt, halfImgDim)       
        
        return scale, strt, center


    def addMinSep(self, nSubs, subScales, subCenters, fitCanvas, scale, halfImgDim):
        """
        Find starting indeces noting minimum seperation restrictions between previously pasted images
        Create subimage to paste into canvas
        INPUT: 
            nSubs - current number of subImages on the canvas
            subScales - scaling of the subImages on the canvas
            subCenters - center points of the subImages on the canvas
            fitCanvas - 2d matrix of allowable starting conditions
            scale - scale of current subImage pasted into canvas
            halfImgDim - the half dimensions of new scaled subImage 
                         (used to calculate subImage center)
        OUTPUT:
            strt - starting [x, y] indeces to paste subImage to canvas
        """
        # Initialize numebrs
        maxSep = 0
        maxSepInd = 0
        
        # Find all starting locations that fit incanvas 
        posFit = np.nonzero(fitCanvas)
        randPerm = random.sample(range(len(posFit[0])),len(posFit[0])) # Random permutation
        
        # Minimum seperation distance for the subImage to be deemed resolved
        minSep = (scale+subScales[0:nSubs])*self.minImgSep
        
        for i in randPerm:
            # Generate starting indeces and starting points 
            newXStrt = posFit[0][i]+self.xBorder
            newYStrt = posFit[1][i]+self.yBorder
            newStrt = [newXStrt, newYStrt]
            newCenter = np.add(halfImgDim,newStrt)
            
            # Calculate distances between previously pasted images
            sep = np.sqrt(np.sum((subCenters[0:nSubs]-newCenter)**2,axis=1))
            
            # Append no fit list if newSubImage intrudes on already pasted subImage
            booleans = [ x>y for (x,y) in zip(sep[0:nSubs], minSep[0:nSubs])]
            if all(booleans):
                return newStrt # return if possible pasting condition is found
            elif(np.sum(sep) > maxSep): # Keep track of maximum seperation                 
                    maxSepInd = i
                    maxSep = np.sum(sep)
        
        # Paste image in position that allows for maximum seperation
        maxXStrt = posFit[0][maxSepInd]+self.xBorder
        maxYStrt = posFit[1][maxSepInd]+self.yBorder
        return [maxXStrt, maxYStrt]
        
    
    # Create Questions    
    def containsA(self, subStrts, subCenters, subScales, subClasses):
        num = random.randint(0,9)
        question = 'Does the image contain a '+str(num)+'?'
        answer = 'No'        
        
        booleans = [ x==num for x in subClasses ]        
        if(any(booleans)):
            answer = 'Yes'
        
        return [question, answer]
    def containsOdd(self, subStrts, subCenters, subScales, subClasses):
        num = random.randint(0,1)        
        question = ['Does the image contain an odd number?', 'Does the image contain an even number?']
        answer = 'No'        
        
        booleans = [ x%2==1 for x in subClasses ]
        booleansEven = [ x%2==0 for x in subClasses ]
        if(any(booleans) and num==0):
            answer = 'Yes'
        elif(any(booleansEven) and num==1):
            answer = 'Yes'
        
        return [question[num], answer]
    def totalSum(self, subStrts, subCenters, subScales, subClasses):
        question = 'What is the total sum of all the numbers in the image?'
        ans = '0'
        if len(subClasses>0):
            ans = str(np.sum(subClasses).astype(int))
        return [question, ans]
    def totalProduct(self, subStrts, subCenters, subScales, subClasses):
        question = 'What is the total product of all the numbers in the image?'
        ans = '0'
        if len(subClasses>0):
            ans = str(np.product(subClasses).astype(int))
        return [question, ans]
    def xMost(self, subStrts, subCenters, subScales, subClasses):
        options = ['top', 'right', 'bottom', 'left']
        choice = random.randint(0,len(options)-1)
        
        question = 'What is the '+options[choice]+' most number in the image?'
        ans = 'None'
        
        if len(subClasses>0):
            if choice == 0:
                ind = subCenters[:,1].index(max(subCenters[:,1]))
                ans = str(subClasses[ind])
            elif choice == 1:
                ind = subCenters[:,0].index(max(subCenters[:,0]))
                ans = str(subClasses[ind])
            elif choice == 2:
                ind = subCenters[:,1].index(min(subCenters[:,1]))
                ans = str(subClasses[ind])
            elif choice == 2:
                ind = subCenters[:,0].index(min(subCenters[:,0]))
                ans = str(subClasses[ind])
        return [question, ans]
    def xNumerical(self, subStrts, subCenters, subScales, subClasses):
        options = ['smallest', 'largest']
        choice = random.randint(0,len(options)-1)
        
        question = 'What is the '+options[choice]+' numerical value in the image?'
        ans = 'None'
        
        if len(subClasses>0):
            if choice == 0:
                ind = subClasses.index(max(subClasses))
                ans = str(subClasses[ind])
            elif choice == 1:
                ind = subClasses.index(min(subClasses))
                ans = str(subClasses[ind])
        return [question, ans]
    def xSize(self, subStrts, subCenters, subScales, subClasses):
        options = ['smallest', 'largest']
        choice = random.randint(0,len(options)-1)
        
        question = 'What is the '+options(choice)+' sized number in the image?'
        ans = 'None'
        
        if len(subClasses>0):
            if choice == 0:
                ind = subScales.index(max(subScales))
                ans = str(subClasses[ind])
            elif choice == 1:
                ind = subScales.index(min(subScales))
                ans = str(subClasses[ind])
        return [question, ans]
    def nSubImages(self, subStrts, subCenters, subScales, subClasses):
        question = 'How many numbers are in the image?'
        return [question, str(len(subClasses))]

## input image dimensions
#img_rows, img_cols = 28, 28
## number of convolutional filters to use
#nb_filters = 32
## size of pooling area for max pooling
#nb_pool = 2
## convolution kernel size
#nb_conv = 3
#
#(X_train, y_train), (X_test, y_test) = mnist.load_data()
#
## the data, shuffled and split between train and test sets
#(X_train, y_train), (X_test, y_test) = mnist.load_data()
#
#X_train = X_train.reshape(X_train.shape[0], 1, img_rows, img_cols)
#X_test = X_test.reshape(X_test.shape[0], 1, img_rows, img_cols)
#X_train = X_train.astype('float32')
#X_test = X_test.astype('float32')
#X_train /= 255
#X_test /= 255
#print('X_train shape:', X_train.shape)
#print(X_train.shape[0], 'train samples')
#print(X_test.shape[0], 'test samples')
#
## the data, shuffled and split between train and test sets
#(X_train, y_train), (X_test, y_test) = mnist.load_data()
#
#X_train = X_train.reshape(X_train.shape[0], 1, img_rows, img_cols)
#X_test = X_test.reshape(X_test.shape[0], 1, img_rows, img_cols)
#X_train = X_train.astype('float32')
#X_test = X_test.astype('float32')
#X_train /= 255
#X_test /= 255

data = Data(X_test[0:100], y_test[0:100], 1, canDim=[64,64], maxNImgs=2, border=[-5,-5], minScale=.5)
canvas, question, answer = data.get_batch()

canvas[0,0,:] = 1
canvas[0,:,0] = 1
        
im = Image.fromarray(canvas[0,:,:]*255)
im.show()