import numpy as np
import math
import cv2
import PIL
import matplotlib.pyplot as plt
from numpy.lib.function_base import gradient
import scipy.signal 
from skimage import data, filters

from numba import njit, jit
from numba.experimental import jitclass


class cannyEdgeDetection():
    """
    Function Prototype:

    Class for implementation of Canny Edge Detection
    Algorithm overview:
    A multi-stage algorithm for edge detection

    1. Noise Reduction - First filter and smooth image using Gaussian Filter
    2. Find Intensity Gradient of Image - First derivative in horizontal 
    direction (Gx) and vertical direction (Gy). From these images two images we can find edge gradient and direction for each pixel

    Edge Gradient (G) = Sqrt((Gx)^2 + (Gy)^2)
    Angle(theta) = arctan(Gy/Gx)

    3. Non-Maximum Suppression - After getting gradient magnitude and direction, a full scan of image is done to remove any unwanted 
    pixels which many not be an edge. For every pixel, we check if it is a local maximum in its neighborhood in the driection of the gradient

    4. Hysteresis Thresholding  - This stage decides which edges are really edges and not an edge. For this we need two thresholds vals,
    minVal and Maxval. 

    """
    def __init__(self,imgPath,kSize,std):
        self.imgPath = imgPath
        self.kSize = kSize
        self.std = std

    def GaussianKernel(self, OneDim = True, plot = False):
        """
        Input sigma value, and length of desired kernel
        returns a 1D or 2D Gaussian Kernel depending on OneDim True (1d) or False (2d)
        plot arg for showing kernel
        """
        #Center the array around zero
        ax = np.linspace(-(self.kSize-1)/2,(self.kSize-1)/2,self.kSize)
        #norm = 1/(2 * np.pi * self.std**2)
        Gaussian =np.exp(-(np.square(ax)/(2*np.square(self.std))))
        
        if OneDim == True :
            #One Dimensional Output
            Kernel = (1/np.sqrt(2*np.pi*self.std)) * Gaussian
            if plot == True:
                axis = plt.axes()
                axis.plot(ax, Kernel)
                plt.show()
            else:
                pass
            return Kernel
        else:
            #Two Dimensional Output
            Kernel = np.outer(Gaussian,Gaussian)
            if plot == True:
                plt.imshow(Kernel,interpolation='none')
                plt.show()
            else:
                pass
            return np.asmatrix(Kernel/ np.sum(Kernel))


    def loadImg(self, plot=False):
        """
        Return Mat object of input image, set plot = True to plot this function
        """
        
        I = cv2.imread(self.imgPath,0)
        if plot == True:
            print("Image has X {} pixels and Y {} pixels".format(I.shape[0],I.shape[1]))
            
            #Show original image
            #cv2.imshow('Original Image',I)
            #cv2.waitKey(0)
            plt.imshow(I, cmap='gray')
            plt.show()

        else:
            pass
        return I

    def oneDimMasks(self):
        """
        Return Gx and Gy Gaussian Masks
        
        By returning Gaussian Kernel Fct
        size of Gx is 1xn and Gy is calculated 
        by transposing from 1xn to nx1.
        """
        Gx = self.GaussianKernel()
        Gy = np.transpose(self.GaussianKernel())
        return Gx,Gy      
   
    def convolutionOperation(self, plot= False):
        """
        Convolution operation to apply
        Kernel Filter for the Gaussian Blur Filter

        Convolution for 1D filter
        (a*v)[n] = Sum (a[m]*v[n-m]) from m = -inf to m = inf
        """

        # Load image and Gaussian kernels for x and y gradients
        I = self.loadImg()
        Gx,Gy = self.oneDimMasks()
        #reverse kernel for Convolution operation
        Gy = np.flip(Gx)
        Gx = np.flip(Gy)
        
        
        #padwidth = int(self.kSize/2 - 1 )
        #I = np.pad(I,padwidth,mode='constant', constant_values = 0)
        #plt.imshow(I)
        #plt.show()

        XgaussFilter = np.zeros((I.shape[0], I.shape[1]))
        YgaussFilter = np.zeros((I.shape[0], I.shape[1]))
        
        
        #Perform Convolution on each direction X and Y
        #This algoritm is x30 faster 
        #Use this function to compare
        '''
        for i in range(I.shape[0]):
            XgaussFilter[i,:] = np.convolve(I[i,:], Gx, 'same')
        for j in range(I.shape[1]):
            YgaussFilter[:,j] = np.convolve(I[:,j],Gy, 'same')
        '''
        # Custom 1D Convolution operation
        
        M,N = I.shape

        for i in range(M) :
            for j in range(N-np.int32(kSize)):
                try:
                    # Perform 1D Convolution on both arrays to get blur
                    # in Y direction
                    YgaussFilter[i,j] = (Gx * I[i : i + Gx.size,j]).sum()
                except:
                    pass
        #1D Convolution operation
        for i in range(M-np.int32(kSize)):
            for j in range(N):
                try:
                    # Perform 1D Convolution on both arrays to get blur
                    # in X 
                    XgaussFilter[i,j] = (Gy* I[i, j : j + Gy.size]).sum()
                except:
                    pass

        #
        XgaussFilter = XgaussFilter[:-kSize,:-kSize]
        YgaussFilter = YgaussFilter[:-kSize,:-kSize]
        if plot == True:
            #print("X Blur")
            plt.imshow(XgaussFilter, cmap='gray')
            plt.show()
            #print("Y Blur")
            plt.imshow(YgaussFilter, cmap='gray')
            plt.show()
            
        return XgaussFilter, YgaussFilter

    def derivativeCalculation(self, plotXYBlur = False, plotGradient = False):
        """
        Using a sobel filter to convolve with the Gaussian
        Blurred image calculate the gradient and slope of gradient (theta)
        
        G = Sqrt(X^2 + Y^2)
        theta = arctan2(X/Y)

        """
        Yblur, Xblur = self.convolutionOperation()
        #Create a sobel filter for this operation of the same size of kernel.
        #Kernel for both X and Y gradients
        Xgradient = np.array([[-1,0,1],[-2,0,2],[-1,0,1]])
        Ygradient = np.transpose(Xgradient)
        
        #Declare empty array for 2d Convolutions
        Xresult = np.zeros_like(Xblur)
        Yresult = np.zeros_like(Yblur)

        # Scipy Convolution for comparision
        #Yresult = scipy.signal.convolve2d(Yblur,Xgradient,'same')
        #Xresult = scipy.signal.convolve2d(Xblur,Ygradient,'same')
        
        #Custom 2D Convolution Operation
        
        padwidth = int(Xgradient.shape[0]/2 - 1)
        Xblur = np.pad(Xblur,padwidth,mode='constant', constant_values = 0)
        Yblur = np.pad(Yblur,padwidth,mode='constant', constant_values = 0)
        M,N = Xblur.shape
        
        for i in range(M):
            for j in range(N):
                try:
                    # Perform 1D Convolution on both arrays to get blur
                    # in Y direction
                    Xresult[i,j] = (Xgradient * Xblur[i:i+Xgradient.shape[0], j: j+Xgradient.shape[1]]).sum()
                    Yresult[i,j] = (Ygradient * Yblur[i:i+Ygradient.shape[0], j: j+Ygradient.shape[1]]).sum()
                except:
                    pass
        
        
        if plotXYBlur == True:
            plt.imshow(Xresult, cmap='gray')
            plt.show()
            plt.imshow(Yresult, cmap ='gray')
            plt.show()
        
        #Calculate the Magnitude of the two Gradients
        Gradient = np.hypot(Xresult,Yresult)
        #Gradient normalization
        Gradient = (Gradient/Gradient.max()) * 255
        theta = np.arctan2(Xresult,Yresult)
        if plotGradient == True:
            plt.imshow(Gradient, cmap ='gray')
            plt.show()
        #Calculate slope of gradient
        
        #plt.imshow(theta, cmap ='gray')
        #plt.show()
        return Gradient, theta , Xresult, Yresult
    
    def nonmaxSuppression(self, ploten = False):
        """
        Non-Max-Suppression:
        1. Create a matrix of zeros as the same size as the gradient output
        2. Identify the edge direction based on the angle value from the angle matrix
        3. Check if the pixel in the same direction has a higher intensity than the current pix
        4. Return the image 
        """     
        G, theta, Xresult, Yresult = self.derivativeCalculation()
        #convert from rads to degrees and offset to +/- pi

        G = G[:-kSize,:-kSize]
        theta = theta * 180. / np.pi
        #theta[theta < 0] += 180
        
        #Create empty array for looping throguh image
        M, N = G.shape
        Z = np.zeros((M,N), dtype=np.int16)
        print("Row: {}, Column {}". format(M,N))

        #Loop through all points in the image
        for i in range(1,M-1):
            for j in range(1,N-1):
                #Find the neighboring pixels in the direction along theta vector
                #XY prime
                
                xp1 = math.cos(theta[i,j])
                yp1 = math.sin(theta[i,j])
                
                #Perform an interpolation alogirthm to find
                #the closest pixel to the coordinates found by
                # cos(theta) and sin(theta)
                # North South East West
                # We need to only calculate 
                # N and NW, W, and SW 
                # due to Symmetry
                if (xp1 < -0.5) and (yp1 < -0.5):
                    #South West
                    col_xp1 = -1
                    row_yp1 = -1
                
                elif (-0.5 <= xp1 <= 0.5) and (yp1 < -0.5):
                    #West
                    col_xp1 = -1
                    row_yp1 = 0

                elif (xp1 > 0.5) and (yp1 < -0.5):
                    #North West
                    col_xp1 = -1
                    row_yp1 = 1
                
                elif (xp1 < -0.5) and (-0.5 <= yp1 <= 0.5):
                    #South
                    col_xp1 = 0
                    row_yp1 = -1

                #Calculate the pix intensity for both points (odd function)
                try:
                    intensityP1 = G[i + col_xp1,j + row_yp1]
                    intensityP2 = G[i - col_xp1, j - row_yp1]
                    intensityOrigin = G[i,j]
                    
                    if (intensityOrigin >= intensityP1) and (intensityOrigin >= intensityP2):
                        Z[i,j] = G[i,j]
                    else:
                        Z[i,j] = 0
                except:
                    pass

        if(ploten == True):            
            plt.imshow(Z, cmap='gray')
            plt.show()
        return Z

    def hysteresisThreshold(self, low = 0.05, high = 0.09, ploten=True):
        """
        Hystersis Threshold:
        Identify pixel intenisties: strong, weak, and non-edge
        1. strong are edges since they fall above the high threshold
        2. weak are pixels below high threshold but cannot be ruled out as an edge
        3. Non-edge are pixels below do not meet the criteria for an edge
        4. Assign strong pixels to 255
        5. Assign Non-Edge to 0
        6. Perform Connectivity algorithm on weak pixels to find which pixels are truly edges
        """
        nmsImg = self.nonmaxSuppression()
        highThres = high*nmsImg.max()
        lowThres = low*highThres

        M,N = nmsImg.shape
        Z = np.zeros((M,N), dtype=np.int32)

        #Declare weak and strong pixel intensities to be assigned
        weak = 50
        strong = 200

        #Return all pixels above highthres criteria
        strong_x, strong_y = np.where(nmsImg>= highThres)
        #Return all pixels below lowthres criteria
        weak_x, weak_y = np.where((nmsImg <= highThres) & (nmsImg >= lowThres))
        #Pixels to be set to zero
        zero_x, zero_y = np.where(nmsImg < lowThres)

        #Set pixels to strong and weak for connectivity algorithm
        Z[strong_x, strong_y] = strong
        Z[weak_x, weak_y] = weak
        Z[zero_x, zero_y] = 0

        for i in range(1,M-1):
            for j in range(1,N-1):
                if (Z[i,j] == weak):
                    try:
                        if ((Z[i+1, j-1] == strong) or (Z[i+1, j] == strong) or (Z[i+1, j+1] == strong)
                            or (Z[i, j-1] == strong) or (Z[i, j+1] == strong)
                            or (Z[i-1, j-1] == strong) or (Z[i-1, j] == strong) or (Z[i-1, j+1] == strong)):
                            Z[i, j] = strong
                        else:
                            Z[i, j] = 0
                    except:
                        pass

        if(ploten == True):            
            plt.imshow(Z, cmap='gray')
            plt.show()
        return Z


        

imgPath = r'C:/MastersCourses/gitWorkspace/CAP5415/CAP5415/Project1/images/img7.jpg'

minVal = 0
maxVal = 0
kSize = 7
std= 3

original_img = cannyEdgeDetection(imgPath,kSize,std).loadImg()
Xgauss, Ygauss = cannyEdgeDetection(imgPath,kSize,std).convolutionOperation()
grad, theta, xblur, yblur  = cannyEdgeDetection(imgPath,kSize,std).derivativeCalculation()
thinned = cannyEdgeDetection(imgPath,kSize,std).nonmaxSuppression()
cannyOutput = cannyEdgeDetection(imgPath,kSize,std).hysteresisThreshold()

'''
#Plot 3 standard deviation outputs
fig = plt.figure(figsize=(40,20))
rows = 1
columns = 3
std = 0.5
for i in range(3):
    
    cannyOutput = cannyEdgeDetection(imgPath,kSize,std).hysteresisThreshold(ploten=False)
    fig.add_subplot(1, columns, i+1)
    plt.imshow(cannyOutput, cmap='gray')
    plt.axis('off')
    plt.title('Canny Edge Output, std = %f ' % std) 
    std = std + 2

plt.show()
'''


fig = plt.figure(figsize=(40,20))
rows = 2
columns = 4


#fig.add_subplot(rows,columns,1)
fig.add_subplot(rows, columns, 1)
plt.imshow(original_img, cmap='gray')
plt.axis('off')
plt.title('Original Image')

# Adds a subplot at the 2nd position
fig.add_subplot(rows, columns, 2)
plt.imshow(Xgauss, cmap='gray')
plt.axis('off')
plt.title('Gaussian Blur X Direction')

# Adds a subplot at the 3rd position
fig.add_subplot(rows, columns, 3)
plt.imshow(Ygauss, cmap='gray')
plt.axis('off')
plt.title('Gaussian Blur Y Direction')


# Adds a subplot at the 3rd position
fig.add_subplot(rows, columns, 4)
plt.imshow(xblur, cmap='gray')
plt.axis('off')
plt.title('Gradient Derivative X')


# Adds a subplot at the 3rd position
fig.add_subplot(rows, columns, 5)
plt.imshow(yblur, cmap='gray')
plt.axis('off')
plt.title('Gradient Derivative Y')


# Adds a subplot at the 3rd position
fig.add_subplot(rows, columns, 6)
plt.imshow(grad, cmap='gray')
plt.axis('off')
plt.title('Gradient')

# Adds a subplot at the 3rd position
fig.add_subplot(rows, columns, 7)
plt.imshow(thinned, cmap='gray')
plt.axis('off')
plt.title('NMS')

# Adds a subplot at the 3rd position
fig.add_subplot(rows, columns, 8)
plt.imshow(cannyOutput, cmap='gray')
plt.axis('off')
plt.title('Canny Edge Output')
plt.show()
