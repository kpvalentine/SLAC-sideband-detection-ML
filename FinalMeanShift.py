# -*- coding: utf-8 -*-
"""
Created on Wed Aug  8 11:03:51 2018
Physics smooth, one image experiment
@author: Kelley Valentine
"""

import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import truncnorm
import pandas as pd
from sklearn.cluster import MeanShift, estimate_bandwidth
from scipy.optimize import curve_fit
from scipy import asarray as ar, exp
from sklearn.neighbors import LocalOutlierFactor
sigmaWeight = 3
alpha = 1 #This is how close the sigmas need to be until convergence is achieved. Too small and will never be satisfied (will jump)

def gaussian(x, a, x0, sigma):
    return a*exp((-(x-x0)**2)/(2*sigma**2))

def zeroMainFreqTmp(tmp):
    xmax = np.argmax(tmp)
    sigma = findSigma(tmp)
    tmp, beg, end = zeroSigmaTmp(tmp, sigma, xmax)
    return tmp, beg, end

def zeroSigmaTmp(tmp, sigma, xmax):
    weightedSigma = sigma*sigmaWeight
    bottom = xmax - weightedSigma
    top = xmax + weightedSigma
    for x in range(int (bottom), int (top+1), 1):
        tmp[x] = 0
        
    return tmp, bottom, top

def findSigma(tmp):
    #print("Find Sigma")
    tmp = graphShift(tmp)
    #Find the index value corresponding to max y value
    xmax = np.argmax(tmp)
    #Find max y value
    ymax = np.max(tmp)
    #Find the t max
    #40,000 is the lower frequency average
    halfPeak = ymax/2
    #Now, split the graph into two, right at where ymax occurs
    firstHalfGraph = tmp[0:(xmax+1):1]
    secondHalfGraph = tmp[xmax:tmp.shape[0]:1]
    #Find where, in the first half, the graph hits halfPeak
    firstHalfIndex = (np.abs(firstHalfGraph - halfPeak)).argmin()
    #Find where, in the second half, the graph hits halfPeak
    secondHalfIndex = (np.abs(secondHalfGraph - halfPeak)).argmin()
    #Find the true halfPeak point of the second half by shifting over
    secondHalfTrue = secondHalfIndex + xmax
    #fWHM = Full Width Half Max
    fWHM = secondHalfTrue - firstHalfIndex
    
    #Determine sigma from fWHM
    newSigma = fWHM / 2.355
#    print("FWHM Sigma", newSigma)
#    sigmaList.append(newSigma)   
    return newSigma

#This will shift the graph down by the minimum y value found
def graphShift(tmp):
    ymin = np.min(tmp)
    #tmp[:] = [y-ymin for y in tmp]
    tmp = tmp - ymin
    return tmp

#Changes all values with in +-3 (on X axis) sigma of max Y's index to 0
def zeroedSigmaTmpLR(tmp, sigma, xmax):
    weightedSigma = sigma*sigmaWeight
    bottom = xmax - weightedSigma
    top = xmax + weightedSigma
    for x in range(int (bottom), int (top+1), 1):
        tmp[x] = 0
        
    return tmp   
    
def cutZeros(tmp):
    Y = []
    X = []
    for x in range(len(tmp)):
        if(tmp[x]>0):
            Y.append(tmp[x])
            X.append(x)
    Y = np.array(Y)
    X = np.array(X) 
    return Y, X
    

#Returns two arrays:
    #dataFirst: The first portion of the data that isn't part of the zeroed middle
    #dataSecond: The second portion of the data that isn't part of the zeroed middle
def split(data, beg, end):
    dataFirst = data[0:int(beg)]
    dataSecond = data[int(end)+1:data.shape[0]]
    return dataFirst, dataSecond
    
#Removes outliers
def outlierDetection(data, beg, end):
    dataFirst, dataSecond = split(data, beg, end)
    firstIndexArray = physicsSmooth(dataFirst)
    secondIndexArray = physicsSmooth(dataSecond)
    for x in range(len(secondIndexArray)):
        secondIndexArray[x] += int(end+1)
    

    
    mergedIndex = firstIndexArray + secondIndexArray
    
    return mergedIndex
    

#Smooths a portion once. Will return whether or not it has converged.
def physicsSmooth(data):
    indexArray = []
    #Transform outliers to zero
    dataX = np.arange(len(data))
    #First, fit a polynomial curve to part 1
    fullPack = np.polyfit(dataX, data, 3, full = True)
    z = fullPack[0]
    f = np.poly1d(z)
    ###To see our fit against the data###
#    plt.plot(dataX, data)
#    plt.plot(dataX, f(dataX))
#    plt.show()
    yHat = f(dataX)
    #Second, determine difference between data and yHat
    resArray = data - yHat
    #Third, make a histogram of these differences.
    hist, bins = np.histogram(resArray)
    #Fourth, get the std (sigma) from the histogram
    mids = 0.5*(bins[1:] + bins[:-1])
    mean = np.average(mids, weights=hist)
    var = np.average((mids - mean)**2, weights=hist)
    sigma = np.sqrt(var)
    #Fifth, take the absolute value of resArray so we can compare to 3sigma.
    #If a value is greater than 3sigma, we get that index and turn it to zero.
    absResArray = np.absolute(resArray)
    for i in range(len(absResArray)):
        if(absResArray[i] > (sigmaWeight*sigma)):
            indexArray.append(i)
             
    return indexArray 

def findNewVal(img, y, x):
    if (x==0):
        return img[y][x+1]
    elif (x >= (img.shape[1] -1)):
        return img[y][x-1]
    else:
        firstVal = int(img[y][x-1])
        scndVal = int(img[y][x+1])
        value = firstVal + scndVal
        return (value/2)

def getCurve(dataIndex, dataValues):
    fullPack = np.polyfit(dataIndex, dataValues, 3, full = True)
    z = fullPack[0]
    f = np.poly1d(z)
    #yHat = f(dataIndex)
    return f

def zeroIMG(img, beg, end):
        for x in range(img.shape[1]):
            for y in range(img.shape[0]):
                if x>=beg:
                    if x<=end:
                        img[y][x] = 0      
        return img
    
def IMGOutlierRemoval(img, tmp, beg, end):
    #Index holds outlier indices
    outlierIndex = outlierDetection(tmp, beg, end)
    img = averageNeighbors(img, outlierIndex)
    return img

def averageNeighbors(img, targetIndex):
    for x in targetIndex:
        for y in range(img.shape[0]):
            #Average the neighbors
            img[y][x] = findNewVal(img, y, x)   
    return img
    
def removeWeakPoints(img):
    df = pd.DataFrame(img)
    df_med = df
    df_med = df - df.median().median()
    X = filter(df_med, 40) 
    return X

def filter(img, cutoff):
    data_sep = [] #empty array
    for x in range(img.shape[0]): #so, this iterates through the x axis
        for y in range(img.shape[1]): #iterates through y axis
            if img[y][x] > cutoff:
                for s in range(int(img[y][x])):
                    #add as many points according to strength
                    data_sep.append([y, x])
    return np.array(data_sep)

def myMeanShift(Y):
    #Y.reshape(-1,1)
    bandwidth = estimate_bandwidth(Y, quantile=.2) 
    ms = MeanShift(bandwidth=bandwidth, bin_seeding=True)
    ms.fit(Y)
    labels=ms.labels_
    #cluster_centers = ms.cluster_centers_
    labels_unique = np.unique(labels)
    n_clusters_ = len(labels_unique)
    clustersList.append(n_clusters_)

#Change according to where your file is
data = sio.loadmat('C:\\Users\\Kelley\\Desktop\\raw_datas\img1791.mat')
clustersList = []

for n in range(107):
    #Use these for each IMG that doesn't work
    if n == 37:
        continue
    elif n == 61:
        continue
    elif n == 63:
        continue
    elif n == 87:
        continue
    print ("n:", n)
    img = data['data'][:,:,n]

    tmp = np.sum(img, axis=0)
    tmpX = np.arange(len(tmp))
    tmp = graphShift(tmp)
    #beg, end BOTH ZERO INCLUSIVE. These are indexes. so, all non zeroes will be [0:beg] and [end+1: tmp.shape[0]]
    tmp, beg, end = zeroMainFreqTmp(tmp)
    tmp, beg, end = zeroMainFreqTmp(tmp)

    img = zeroIMG(img, beg, end)

    img = IMGOutlierRemoval(img, tmp, beg, end)

    #Now just do the mean shift part

    Y = removeWeakPoints(img)

    myMeanShift(Y)

np.asarray(clustersList)


g = plt.figure(2)
plt.title("Number of Clusters")
plt.hist(clustersList)
g.show()
