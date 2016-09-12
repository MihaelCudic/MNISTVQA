
# coding: utf-8

# In[1]:

"""
Create online database for question/answer on a canvas of MNIST images

@author: Mihael Cudic 
"""
import re
import numpy as np
from scipy.misc import imresize

import random
random.seed()


# In[2]:

class VQAData(object):
    
    def __init__(self,
                 X, Y, batchSize,
                 xCanDim = 64,
                 yCanDim = 64,                 
                 canDim = None,
                 border = None,
                 xBorder = 0,
                 yBorder = 0,
                 nImgs = None,
                 minNImgs = 1,
                 maxNImgs = 2,
                 randNImgs = False,
                 scaling = None,
                 minScale = 1,
                 maxScale = 1,
                 minImgSep = None,
                 clutterDim = None,
                 xClutter = None,
                 yClutter = None,
                 nClutter = None,
                 clutter = False,
                 minClutter = 1,
                 maxClutter = 3,
                 randClutter = False,
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
        clutter - enable cluttering in canvasses (Default: False)
        minClutter - minimum distortions added to canvas (Default: 1)
        maxClutter - maximum distortions added to canvas (Default: 3)
        randClutter - Default False: paste maxClutter distortions on canvas; 
                      True: randomly select number of distortions to paste on canvas
        xClutter - X dimension of distortion (Default: 0)
        yClutter - Y dimension of distortion (Default: 0)
        
        
        USE ARRAYS To Simplify initializations
        canDim - [xCanDim, yCanDim]
        border - [xBorder, yBorder]
        nImgs - [minNImgs, maxNImgs, randNImgs]
        scaling - [minScale, maxScale]
        clutterDim - [xClutter, yClutter]
        nClutter - [clutter, minClutter, maxClutter, randClutter]
        """
        
        # Store attributes of input data
        self.X = X
        self.Y = Y
        self.batchSize = batchSize
        self.inputSize = len(X)
        
        # Set dimensions of input images
        self.imgDim = [len(X[1][0][0]), len(X[1][0])]

        # Set dimensions of canvas       
        self.canDim = [xCanDim, yCanDim] if canDim == None else canDim        
        
        # Set borders inside canvas       
        self.xBorder, self.yBorder = [xBorder, yBorder] if border == None else border 
        
        # Set the attribute regarding number of images pasted to canvas
        self.minNImgs, self.maxNImgs, self.randNImgs = [minNImgs, maxNImgs, randNImgs] if nImgs == None else nImgs       
        
        # Set the the scaling of the pasted images
        self.minScale, self.maxScale = [minScale, maxScale] if scaling == None else scaling
        
        # Set seperation of pasted images
        self.minImgSep = np.max(self.imgDim)/2 if minImgSep is None else minImgSep     
        
        # Set the attribute regarding number of images pasted to canvas
        self.clutter, self.minClutter, self.maxClutter, self.randClutter = [clutter, minClutter, maxClutter, randClutter] if nClutter == None else nClutter  
        
        # Set borders inside canvas       
        self.clutterDim = [xClutter, yClutter] if clutterDim == None else clutterDim
        
        # List of possible functions called
        self.askQ = [self.containsA, self.containsOdd, self.totalSum, self.totalProduct, 
                     self.xMost, self.xNumerical, self.xSize, self.nSubImages, self.xRange]
        
    def getBatch(self):
        """
        Create batch of input data
        INPUT: None
        OUTPUT:
            canvas - 3d matrix of images (Note: [image number, :, :] == input canvas)
            question - question corresponding to canvas (Specific to MNIST)
            answer - answer to question
            qType - type of question asked
        """ 
        # Preallocate space for output variables
        canvas = np.zeros([self.batchSize, self.canDim[0], self.canDim[1]])
        question = [None]*self.batchSize
        answer = [None]*self.batchSize
        qType = [None]*self.batchSize
        classes = []
        
        # Generate input data for batch
        for i in range(self.batchSize):         
            canvas[i,:,:], subClasses, subScales, subStrts, subCenters = self.createCanvas()
            classes.append(subClasses)
            question[i], answer[i], qType[i] = random.choice(self.askQ)(subStrts, subCenters, subScales, subClasses)
        
        canvas = canvas[:, None, :, :]
            
        return canvas, question, answer, qType, classes
    
    def randomDataset(self):
        """
        Randomize the given dataset. However, instead of shuffling the data, we will just create a random
        permutation and index from there
        """
        self.rIndPerm = np.random.permutation(self.inputSize)
        self.rInd = 0
    
    def createCanvas(self):
        """
        Create new canvas given specs assigned previously
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
        
        # Create specs for new subImgs
        subScales = []
        newImgDim = []
        subStrts = []
        subCenters = []
        if nImgs > 0:
            subScales, newImgDim, subStrts, subCenters = self.addSubImgs(nImgs) 
        
        # Preallocate space for subImg classification
        subClasses = [0]*nImgs # classes of input images in canvas

        self.randomDataset() # Shuffle dataset 
        # Iterate and paste subImgs on canvas
        for i in range(nImgs):
            # Choose random image from input permutation
            indX = self.rIndPerm[self.rInd]
            self.rInd += 1
            if self.rInd == self.inputSize:
                self.randomDataset() # Reshuffle input data
            
            # Store specs
            subImg = np.squeeze(self.X[indX][None][:][:])
            subClasses[i] = self.Y[indX]
            
            # Resize image to new dimensions
            img = imresize(subImg, newImgDim[i,:])
            img = img.astype('float32')
            img /= 255
            
            # Find start and end points of image projected onto canvas
            xStrt, yStrt = subStrts[i,:]
            xEnd, yEnd = np.add([xStrt, yStrt],newImgDim[i,:])
            
            # Restrict strt and end points to not index out of bounds
            # NOTE: I think this can be simplified
            xImgStrt, yImgStrt = [0,0]
            xImgEnd, yImgEnd = newImgDim[i,:]
            
            if xStrt < 0:
                xImgStrt = -xStrt
                xStrt = 0
            if yStrt < 0:
                yImgStrt = -yStrt
                yStrt = 0
            if xEnd > self.canDim[0]:
                xImgEnd = newImgDim[i,0]-(xEnd-self.canDim[0])
                xEnd = self.canDim[0]
            if yEnd > self.canDim[1]:
                yImgEnd = newImgDim[i,1]-(yEnd-self.canDim[1])
                yEnd = self.canDim[1]
            
            # Paste image to canvas
            canvas[yStrt:yEnd,xStrt:xEnd] = np.add(canvas[yStrt:yEnd,xStrt:xEnd],img[yImgStrt:yImgEnd,xImgStrt:xImgEnd])
        
        # Add clutter
        if self.clutter:
            # Decide how many images will be pasted
            nClutter = self.maxClutter if not self.randClutter else random.randint(self.minClutter, self.maxClutter)
            
            # Add distortion
            for i in xrange(nClutter):
                # Choose random image from input permutation
                indX = self.rIndPerm[self.rInd]
                self.rInd += 1
                if self.rInd == self.inputSize:
                    self.randomDataset() # Reshuffle input data
                    
                # Distortion Img
                dist = np.squeeze(self.X[indX][None][:][:])
                
                # Find coordinates
                imgX = random.randint(1,(self.imgDim[0]-self.clutterDim[0]))-1
                imgY = random.randint(1,(self.imgDim[1]-self.clutterDim[1]))-1
                canX = random.randint(1,(self.canDim[0]-self.clutterDim[0]))-1
                canY = random.randint(1,(self.canDim[1]-self.clutterDim[1]))-1
                
                newPatch = np.add(canvas[canY:canY+self.clutterDim[1], canX:canX+self.clutterDim[0]],subImg[imgY:imgY+self.clutterDim[1], imgX:imgX+self.clutterDim[0]])
                canvas[canY:canY+self.clutterDim[1], canX:canX+self.clutterDim[0]] = newPatch
        
        # Cap canvas at 1
        # NOTE: I think this can be done faster
        canvas[canvas>1] = 1
        
        return canvas, subClasses, subScales, subStrts, subCenters

    def addSubImgs(self, nImgs):
        """
        Find specs of each pasted subImg
        INPUT: 
            nImgs - how subImgs will be pasted
        OUTPUT:
            scale - scaling for each subImgs
            newImgDim - new dimensions once scaled
            strt - starting indeces [x, y] of new subImg
            center - center points [x, y] of new subImg 
        """
        # Dimensions of canvas accounting for border restrictions
        dim = [x-2*y for (x,y) in zip(self.canDim,[self.xBorder, self.yBorder]) ]
        
        # Preallocate space
        strt = np.zeros([nImgs, 2]).astype(int) # starting [x, y] indeces of input images on canvas
        center = np.zeros([nImgs, 2]) # centered [x, y] indeces of input images on canvas
        
        # Create new scaling of subImages
        scale = np.random.uniform(self.minScale, self.maxScale, nImgs)
        newImgDim = np.round(self.imgDim * scale[:, np.newaxis]).astype(int)
        halfImgDim = map(lambda x: (x-1.0)/2.0, newImgDim) # used for determining center points of image
        halfImgDim = np.reshape(halfImgDim, (nImgs, 2))
        
        # Generate specs of first subImgs
        newXStrt = random.randint(0,dim[0]-newImgDim[0,0]-1)+self.xBorder
        newYStrt = random.randint(0,dim[1]-newImgDim[0,1]-1)+self.yBorder
        strt[0,:] = [newXStrt, newYStrt]
        center[0,:] = np.add(halfImgDim[0,:],strt[0,:])
        nSubs = 1;
        
        # Generate specs of other subImgs
        for i in range(1, nImgs):
            found = False
            
            # Initialize numbers
            maxSep = 0
            maxSepInd = 0        
            
            # Initialize possible starting locations for the subImage
            # 1 means that image can fit given its conditions
            fitCanvas = np.ones(dim)
            
            # Adjust possible starting locations for new subImage
            # Apply size resitrictions (image isn't placed where it will index out of position)
            fitCanvas[-newImgDim[i,0]:,:] = 0 
            fitCanvas[:,-newImgDim[i,1]:] = 0 
            
            # Find all starting locations that fit incanvas 
            posFit = np.nonzero(fitCanvas)
            randPerm = np.random.permutation(len(posFit[0])) # Random permutation
            
            # Minimum seperation distance for the subImage to be deemed resolved
            minSep = (scale[0:i]+scale[i])*self.minImgSep

            for ind in randPerm:
                # Generate starting indeces and starting points 
                newXStrt = posFit[0][ind]+self.xBorder
                newYStrt = posFit[1][ind]+self.yBorder
                newStrt = [newXStrt, newYStrt]
                newCenter = np.add(halfImgDim[i,:],newStrt)

                # Calculate distances between previously pasted images
                sep = np.sqrt(np.sum((center[0:i]-newCenter)**2,axis=1))

                # Append no fit list if newSubImage intrudes on already pasted subImage
                booleans = [ x>y for (x,y) in zip(sep[0:i], minSep[0:i])]
                if all(booleans):
                    strt[i,:] = newStrt
                    center[i,:] = newCenter
                    found = True
                    break
                elif(np.sum(sep) > maxSep): # Keep track of maximum seperation                 
                        maxSepInd = ind
                        maxSep = np.sum(sep)
            if not found:
                # Paste image in position that allows for maximum seperation
                maxXStrt = posFit[0][maxSepInd]+self.xBorder
                maxYStrt = posFit[1][maxSepInd]+self.yBorder
                strt[i,:] = [maxXStrt, maxYStrt]
                center[i,:] = np.add(halfImgDim[i,:],strt[i,:])
        return scale, newImgDim, strt, center 

    
    
    def getVocab(self):
        """
            Generate all possible words used for question and answers
            INPUTS: None
            OUTPUTS:
                vocab - unique vocab used in questions and answers
        """
        qWords = '' 
        for q in self.askQ:
            qWords += " "+q(vocab=True)

        # List of words used in questions ((\W+)? are seperted as well)
        vocabList = [x.strip() for x in re.split('(\W+)?', qWords) if x.strip()]

        # Predefined list of words used in questions
        # NOTE: these might have to be adjusted. I did this to save time
        vocabList += ['None', 'Yes', 'No', 'NA'] # Known outputs
        vocabList += map(str, xrange((9**self.maxNImgs)+1)) # Account for products

        return list(set(vocabList))
    
    
    
    """
        Create Questions about canvas
            INPUTS:
                subStrts -  array of starting indeces of subImgs ([x, y])
                subCenters - array of center indeces of subImgs ([x, y])
                subScales - array of subImg scaling
                subClasses - array of classifications for subImgs
                vocab - boolean stating whether you want the possible questions outputed
            OUPUTS:
                [question, answer, type] - question and answer of canvas; type of question asked
    """   
    def containsA(self,
                  subStrts = None,
                  subCenters = None,
                  subScales = None,
                  subClasses = None,
                  vocab = False):
        num = random.randint(0,9)
        aNum = 'an '+str(num) if num == 8 else 'a '+str(num)
        question = ['Does the image contain '+aNum+'?',
                    'Is '+str(num)+' in the image?',
                    'Is there '+aNum+'?']
        answer = 'No'       
        
        if(vocab):
            return " ".join(question+['a', 'an'])
        
        booleans = [ x==num for x in subClasses ]        
        if(any(booleans)):
            answer = 'Yes'
        
        return [random.choice(question), answer, 0]
    def containsOdd(self,
                    subStrts = None,
                    subCenters = None,
                    subScales = None,
                    subClasses = None,
                    vocab = False):
        num = random.randint(0,1)
        p = ['odd', 'even']
        question = ['Does the image contain an '+p[num]+' number?',
                    'Is an '+p[num]+' number in the image?',
                    'Is there an '+p[num]+' number?']
        answer = 'No'        
        
        if(vocab):
            return " ".join(question+p)
        
        booleans = [ x%2==1 for x in subClasses ]
        booleansEven = [ x%2==0 for x in subClasses ]
        if(any(booleans) and num==0):
            answer = 'Yes'
        elif(any(booleansEven) and num==1):
            answer = 'Yes'
        
        return [random.choice(question), answer, 1]
    def totalSum(self,
                 subStrts = None,
                 subCenters = None,
                 subScales = None,
                 subClasses = None,
                 vocab = False):
        question = ['What is the total sum?',
                    'What is the sum of all the numbers?',
                    'The total sum is?']
        ans = 'NA'
        
        if(vocab):
            return " ".join(question)
        
        if len(subClasses)>0:
            ans = str(int(np.sum(subClasses)))
        return [random.choice(question), ans, 2]
    def totalProduct(self,
                     subStrts = None,
                     subCenters = None,
                     subScales = None,
                     subClasses = None,
                     vocab = False):
        question = ['What is the total product?',
                    'What is the product of all the numbers?',
                    'The total product is?']
        ans = 'NA'
        
        if(vocab):
            return " ".join(question)
        
        if len(subClasses)>0:
            ans = str(int(np.product(subClasses)))
        return [random.choice(question), ans, 3]
    def xMost(self,
              subStrts = None,
              subCenters = None,
              subScales = None,
              subClasses = None,
              vocab = False):
        options = ['top', 'right', 'bottom', 'left']
        choice = random.randint(0,len(options)-1)
        
        question = ['What is the '+options[choice]+' most number?',
                    'The '+options[choice]+' most number is?',
                    'What number is closest to the '+options[choice]+' side?']
        ans = 'None'
        
        if(vocab):
            return " ".join(question+options)
        
        if len(subClasses)>0:
            if choice == 0:
                ind = subCenters[:,1].tolist().index(min(subCenters[:,1]))
                ans = subClasses[ind]
            elif choice == 1:
                ind = subCenters[:,0].tolist().index(max(subCenters[:,0]))
                ans = subClasses[ind]
            elif choice == 2:
                ind = subCenters[:,1].tolist().index(max(subCenters[:,1]))
                ans = subClasses[ind]
            elif choice == 3:
                ind = subCenters[:,0].tolist().index(min(subCenters[:,0]))
                ans = subClasses[ind]
        return [random.choice(question), str(ans), 4]
    def xNumerical(self,
                   subStrts = None,
                   subCenters = None,
                   subScales = None,
                   subClasses = None,
                   vocab = False):
        options = ['smallest', 'largest']
        choice = random.randint(0,len(options)-1)
        
        question = ['What is the '+options[choice]+' numerical value?',
                    'The '+options[choice]+' numerical value is?',
                    'What number is numerically the '+options[choice]+'?']
        ans = 'None'
        
        if(vocab):
            return " ".join(question+options)
        
        if len(subClasses)>0:
            if choice == 0:
                ind = subClasses.index(min(subClasses))
                ans = subClasses[ind]
            elif choice == 1:
                ind = subClasses.index(max(subClasses))
                ans = subClasses[ind]
        return [random.choice(question), str(ans), 5]
    def xSize(self,
              subStrts = None,
              subCenters = None,
              subScales = None,
              subClasses = None,
              vocab = False):
        options = ['smallest', 'largest']
        choice = random.randint(0,len(options)-1)
        
        question = ['What is the '+options[choice]+' sized number?',
                    'The '+options[choice]+' sized number is?',
                    'What number has the '+options[choice]+' size?']
        ans = 'None'
        
        if(vocab):
            return " ".join(question+options)

        if len(subClasses)>0:
            if choice == 0:
                ind = subScales.tolist().index(min(subScales))
                ans = subClasses[ind]
            elif choice == 1:
                ind = subScales.tolist().index(max(subScales))
                ans = subClasses[ind]
        return [random.choice(question), str(ans), 6]
    def nSubImages(self,
                   subStrts = None,
                   subCenters = None,
                   subScales = None,
                   subClasses = None,
                   vocab = False):
        question = ['How many numbers are in the image?',
                    'How many numbers are there?',
                    'The image contains how many numbers?']
        
        if(vocab):
            return " ".join(question)

        return [random.choice(question), str(len(subClasses)), 7]
    
    def xRange(self,
                   subStrts = None,
                   subCenters = None,
                   subScales = None,
                   subClasses = None,
                   vocab = False):
        question = ['What is the range?',
                    'What is the range of all the numbers?',
                    'The total range is?']
        ans = 'NA'
        
        if(vocab):
            return " ".join(question)
        
        if len(subClasses)>1:
            num = int(max(subClasses)-min(subClasses))
            ans = str(num)
        
        return [random.choice(question), ans, 8]


# In[3]:

def formatClasses(classes):
    length = len(classes)
    Y = np.array(np.zeros([length, 10]))
    for ind in xrange(length):
        Y[ind, np.array(classes[ind][:]).astype(int)] = 1.

    return np.array(Y)


# In[4]:

"""
    Generate MNIST Images
"""
from keras.datasets import mnist
from keras.utils import np_utils

nb_classes = 10
# input image dimensions
img_rows, img_cols = 28, 28
# number of convolutional filters to use
nb_filters = 32
# size of pooling area for max pooling
nb_pool = 2
# convolution kernel size
nb_conv = 3

# the data, shuffled and split between train and test sets
(X_train, Y_train), (X_test, Y_test) = mnist.load_data()

X_train = X_train.reshape(X_train.shape[0], 1, img_rows, img_cols)
X_test = X_test.reshape(X_test.shape[0], 1, img_rows, img_cols)
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255


# In[5]:

"""
    Generate VQA datasets
"""
print("Creating Objects")
# Initialize Parameters
nTrain = 100000 # Size of training dataset
nValid = 25000 # Size of validation dataset
pValid = float(nValid)/(nTrain+nValid)
nTest = 25000 # Size of test dataset
canDim = [64, 64]
border=[-3,-3]
nImgs=[1,3, True]
scaling=[.6,1.2]
clutter = True
clutterDim=[8,8]
maxClutter = 6

# Seperate MNIST
trMNIST = int(np.round((1-pValid)*len(X_train)))

X_valid = X_train[trMNIST:,:,:,:]
Y_valid = Y_train[trMNIST:]
X_train = X_train[0:trMNIST,:,:,:]
Y_train = Y_train[0:trMNIST]

# Create instance of data
train = VQAData(X_train, 
            Y_train, 
            nTrain, 
            canDim=canDim, 
            border=border, 
            nImgs=nImgs, 
            scaling=scaling,
            clutter=clutter,
            clutterDim = clutterDim,
            maxClutter = maxClutter)

valid = VQAData(X_valid, 
            Y_valid, 
            nValid, 
            canDim=canDim, 
            border=border, 
            nImgs=nImgs, 
            scaling=scaling,
            clutter=clutter,
            clutterDim = clutterDim,
            maxClutter = maxClutter)
test = VQAData(X_test, 
            Y_test, 
            nTest, 
            canDim=canDim, 
            border=border, 
            nImgs=nImgs, 
            scaling=scaling,
            clutter=clutter,
            clutterDim = clutterDim,
            maxClutter = maxClutter)


# In[6]:


"""
    Store Data
"""
"""
print("Storing...")
import h5py

title = "VQADataCluttered.hdf5"
fout = h5py.File(title, "w")


C_train, Q_train, A_train, T_train, Class_train = train.getBatch()
fout.create_dataset("C_train", data=C_train)
fout.create_dataset("Q_train", data=Q_train)
fout.create_dataset("A_train", data=A_train)
fout.create_dataset("T_train", data=T_train)
del C_train, Q_train, A_train, T_train
print("Training finished...")

C_valid, Q_valid, A_valid, T_valid, Class_valid = valid.getBatch()
fout.create_dataset("C_valid", data=C_valid)
fout.create_dataset("Q_valid", data=Q_valid)
fout.create_dataset("A_valid", data=A_valid)
fout.create_dataset("T_valid", data=T_valid)
del C_valid, Q_valid, A_valid, T_valid
print("Validation finished...")

C_test, Q_test, A_test, T_test, Class_test = test.getBatch()
fout.create_dataset("C_test", data=C_test)
fout.create_dataset("Q_test", data=Q_test)
fout.create_dataset("A_test", data=A_test)
fout.create_dataset("T_test", data=T_test)
del C_test, Q_test, A_test, T_test
print("Test finished...")

vocab = train.getVocab()
fout.create_dataset("vocab", data=vocab)
print("Vocab finished...")

fout.close()
"""

print("Storing...")
import h5py

title = "imgRecognitionCluttered.hdf5"
fout = h5py.File(title, "w")

C_train, Q_train, A_train, T_train, Class_train = train.getBatch()
fout.create_dataset("C_train", data=C_train)
fout.create_dataset("Class_train", data=formatClasses(Class_train))
del C_train, Q_train, A_train, T_train, Class_train

C_valid, Q_valid, A_valid, T_valid, Class_valid = valid.getBatch()
fout.create_dataset("C_valid", data=C_valid)
fout.create_dataset("Class_valid", data=formatClasses(Class_valid))
del C_valid, Q_valid, A_valid, T_valid, Class_valid
print("Validation finished...")

C_test, Q_test, A_test, T_test, Class_test = test.getBatch()
fout.create_dataset("C_test", data=C_test)
fout.create_dataset("Class_test", data=formatClasses(Class_test))
del C_test, Q_test, A_test, T_test, Class_test
print("Test finished...")

print("Training finished...")
fout.close()


# In[7]:

"""
%matplotlib inline
import matplotlib.pyplot as plt 
from PIL import Image

# Initialize Parameters
nTrain = 9 # Size of training dataset
nValid = 50000 # Size of validation dataset
pValid = float(nValid)/(nTrain+nValid)
nTest = 50000 # Size of test dataset
canDim = [64, 64]
border=[-3,-3]
nImgs=[1,3, True]
minImgSep = 0,
scaling=[.6,1.2]
clutter = True
clutterDim=[8,8]
maxClutter = 6

# Create instance of data
train = VQAData(X_train, 
            Y_train, 
            nTrain, 
            canDim=canDim, 
            border=border, 
            nImgs=nImgs, 
            scaling=scaling,
            clutter=clutter,
            clutterDim = clutterDim,
            maxClutter = maxClutter)

C_train, Q_train, A_train, T_train = train.getBatch()

for i in range(9):
    plt.figure(figsize=(10, 10))
    plt.subplot(3, 3, i+1)
    plt.imshow(C_train[i,0,:,:], cmap='gray')
    plt.gca().get_xaxis().set_ticks([])
    plt.gca().get_yaxis().set_ticks([])
    plt.title("%s" % Q_train[i], fontsize=8)
    plt.ylabel("answer = %s" % A_train[i], fontsize=8)
    
#print(question[0])
#print(answer[0])

plt.axis("off")
"""


# In[8]:

"""
I = C_train[7,0,:,:]

I8 = (((I - I.min()) / (I.max() - I.min())) * 255.9).astype(np.uint8)

img = Image.fromarray(I8)
img.save("clutter1.png")
"""

