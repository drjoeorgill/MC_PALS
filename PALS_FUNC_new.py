# -*- coding: utf-8 -*-
"""
Created on Thu Jun 29 13:30:24 2023

@author: php20jo
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.interpolate import splev, splrep
import os

def single(x, A, l1):
    return A * np.exp(-(l1 * x))

def conv(x,height,position,std,A,l1,B,C,l3):
    l2=(1/0.125)
    g=height*np.exp(-(x-position)**2/(2*std**2))
    e=(A * (np.exp(-(l1 * x))) + B * (np.exp(-(l2 * x))) + C * (np.exp(-(l3 * x))))# + D)
    return (np.convolve(g,e,mode='full') / sum(e) )[:667]

def flat(x,D):
    y=D
    return y

# Print the current working directory
print("Current working directory: {0}".format(os.getcwd()))

#usefull colorblind frieldly colors for the plots
CB = ['#377eb8', '#ff7f00', '#4daf4a','#f781bf', '#a65628', '#984ea3','#999999', '#e41a1c', '#dede00']

iD = 25


def plot_pals(sample):
   
    files = np.loadtxt('charlies_samples.txt',dtype=str)
    filename = files[sample] + '_T1'
    
    label = filename
    
    print(label)
    file = np.loadtxt(filename + '.dat', comments = 'S')
    a = len(file[3:])
    time = np.linspace(-10,50,a)
    N = np.zeros(a)
    for i in range(len(N)):
        N[i] = file[i+3]
    Nerr=(np.sqrt(N))
    
    #find back
    zero = 333   #fit background before rise (before zero)
    xb = time[:zero]  #only use x & y & error before zero
    yb = N[:zero] 
    Nerrb = Nerr[:zero]
    back, backpcov = curve_fit(flat,xb,yb,p0=(iD),sigma=Nerrb)  #fit flat backgound
    Nb = N-back #remove background from all y data

    print(back)
    
    #useful data
    start = 333
    stop = 1000
    y = Nb[start:stop]  #useful section of background removed y data
    x = time[start:stop] 
    #error needs to be derived from the original value for N
    ##Nerr defined at top
    Nberr=Nerr[start:stop]
    
    plt.figure('PALS histogram fit')
    plt.errorbar(x,y,yerr=Nberr,label=label,color=CB[1])
    plt.xlabel('Time [ns]')
    plt.ylabel('Counts')
    plt.legend()
    #plt.show()
    #return(float(FV), FVe, FFV3, FFV3e,label)
    return()

plot_pals(0)