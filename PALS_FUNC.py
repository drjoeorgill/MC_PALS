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

# Change the current working directory
os.chdir('C:/Users/pczjo/OneDrive - The University of Nottingham/Desktop/Pals')
#os.chdir('C:/Users/php20jo/Desktop/experiments/PALSvsComputer')

# Print the current working directory
print("Current working directory: {0}".format(os.getcwd()))

#usefull colorblind frieldly colors for the plots
CB = ['#377eb8', '#ff7f00', '#4daf4a','#f781bf', '#a65628', '#984ea3','#999999', '#e41a1c', '#dede00']

#%%import and plot TaoEldrup function

TaoEldrup=np.load('TaoEldrup.npy') #import TaoEldrup bubble model with 10,000 points (calculated from equation)
TaoEldrup_x=TaoEldrup[0,:] 
TaoEldrup_y=TaoEldrup[1,:]

r = np.zeros(len(TaoEldrup_y))
for i in range(len(TaoEldrup_y)):
    r[i] = ((3*TaoEldrup_y[i])/(4*np.pi))**(1/3)


fig, ax1 = plt.subplots(figsize=(8, 8))
ax2 = ax1.twinx()

ax1.plot(TaoEldrup_x,TaoEldrup_y,color=CB[0])
ax2.plot(TaoEldrup_x,r,color=CB[1])

ax1.set_xlabel("oPs Lifetime [ns]")
ax1.set_ylabel("Free Volume [$\AA^3$]", color=CB[0], fontsize=14)
ax1.tick_params(axis="y", labelcolor=CB[0])

ax2.set_ylabel("Void Radius [$\AA$]", color=CB[1], fontsize=14)
ax2.tick_params(axis="y", labelcolor=CB[1])

#%%samples

sample_list = ['test','silicon','unknownepoxy']
#%%
#exp initial guess
iA = 1000000
il1=1/0.3
iB = 1000000
iC = 1700
il3 = 1/1.6
iD = 25

#gauss initial guess
iheight= 36651
iposition=1.2
istd=0.21


def load(sample,plot,print_values):
    '''
    Inputs: sample = sample number in txt doc / list above
    plot = 'yes' or 'no' if you want graph
    print_values = 'yes' if you want values to be printed
    
    returns: graph of fit if plot = 'yes'
    FV = free volume
    FVe = error in free volume fit
    FFV3 = fractional free volume of the oPs positrons
    FFVe = error in FFV fit
    Plots a graph of fit if plot = 'yes'
    '''
    #files = np.loadtxt('samples.txt',dtype=str)
    #filename = files[sample]# + '_T1'
    filename = sample_list[sample]
    label = filename
    if print_values == 'yes':
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
    if print_values == 'yes':
        print(back)
    
    #useful data
    start = 333
    stop = 1000
    y = Nb[start:stop]  #useful section of background removed y data
    x = time[start:stop] 
    #error needs to be derived from the original value for N
    ##Nerr defined at top
    Nberr=Nerr[start:stop]
    
    #fit convolution func
    popt, pcov = curve_fit(conv,x,y,p0=(iheight,iposition,istd,iA,il1,iB,iC,il3),bounds=(0,np.inf),sigma=np.mean(Nberr))
    height,position,std,A,l1,B,C,l3 = popt #define optimised variables
    l2 = 1/0.125 #define L2 because it is fixed in the function
    t3 = 1/l3 #calculate the oPs lifetime
    
    #find FV
    TaoEldrup=np.load('TaoEldrup.npy') #import TaoEldrup bubble model with 10,000 points (calculated from equation)
    TaoEldrup_x=TaoEldrup[0,:] 
    TaoEldrup_y=TaoEldrup[1,:]
    spl = splrep(TaoEldrup_x, TaoEldrup_y)
    FV = splev(t3,spl) # calculate the FV for the measured oPs lifetime
    
    #print results
    #print('height= ',height,'position= ',position,'std= ',std,'\nA= ',A,'\nB= ',B,'\nC= ',C,'\nD= ',back,'\nt1= ',1/l1,'\nt2= ',1/l2,'\nt3= ',1/l3)
    FWHM = 2*np.sqrt(2*np.log(2))*std
    FWHM = FWHM * 1000
    perr = np.sqrt(np.diag(pcov))
    errsd = perr[2]
    errFWHM = (errsd/std)*FWHM
    if print_values == 'yes':
        print('FWHM = ', FWHM)
        print('tau = ', t3)
        print('Free volume = ', FV, ' A^3')
    FVe = (perr[-1]/l3)*FV
    if print_values == 'yes':
        print('free volume error = ', FVe)
    A1 = A/l1     #Area under exponential decay
    A2 = B/l2
    A3 = C/l3
    At = A1+A2+A3
    I1 = A1/At    #Intensity is proportional to area under curve: Klym, H.; Annihilation Lifetime Spectroscopy Insight on Free Volume Conversion of Nanostructured MgAl2O4 Ceramics. Nanomaterials 2021, 11, 3373
    I2 = A2/At
    I3 = A3/At
    It = I1+I2+I3
    I3e = (perr[-1]/l3)*I3
    #print('I1 = %f I2 = %f I3 = %f It = %f' %(I1,I2,I3,It))
    FFV3 = I3*FV
    FFV3e = (perr[-1]/l3)*FFV3
    if print_values == 'yes':
        print('FFV = ',FFV3)
        print('FFV error = ',FFV3e)
    
    if plot == 'yes':
        #plot graph
        #plt.figure()
        #plt.plot(N)
        plt.figure('PALS histogram fit')
        plt.errorbar(x,y,yerr=Nberr,label=label)
        plt.plot(x,conv(x,height,position,std,A,l1,B,C,l3),label='fit')
        #plt.title(label)
        plt.xlabel('Time [ns]')
        plt.ylabel('Counts')
        plt.legend()
        plt.show()
    return(float(FV), FVe, FFV3, FFV3e,label)

#%%run pals func


FV, FVerr, FFV3,FFVerr, label = load(2,'yes','yes')


#%% the rest is just looping through samples left in as an example
"""
this = np.arange(177,191)
data = np.zeros((4,len(this)))
names = []
for i in range(len(this)):
    data[0,i] = load(this[i],'','')[0]
    data[1,i] = load(this[i],'','')[1]
    data[2,i] = load(this[i],'','')[2]
    data[3,i] = load(this[i],'','')[3]
    name = (load(this[i],'',''))[4]
    print(name)
    print('FV=',data[0,i],'+/-',data[1,i],' FFV=',data[2,i],'+/-',data[3,i])
    
    
#%%
this = np.arange(177,191)
data = np.zeros((4,len(this)))

for i in range(len(this)):
    data[:,i] = load(this[i],'','yes')
#%%
for i in range(len(this)):
    print(data[4,i])

    
#load(103,'','yes')
    
#%%
"""

