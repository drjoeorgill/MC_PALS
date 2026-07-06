from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

import pytensor
# Force the compiler to stop trying to be 'too smart' (speeds up start time)
pytensor.config.mode = 'FAST_COMPILE'

import pymc as pm
import pytensor.tensor as pt
import numpy as np
import arviz as az

def flat(x,D):
    y=D
    return y
iD = 25

def conv(x,height,position,std,A,B,C,l1,l3):
    l2=(1/0.125)
    g=height*np.exp(-(x-position)**2/(2*std**2))
    e=(A * (np.exp(-(l1 * x))) + B * (np.exp(-(l2 * x))) + C * (np.exp(-(l3 * x))))# + D)
    return (np.convolve(g,e,mode='full') / sum(e) )[:667]

def plot_pals(sample):
   
    files = np.loadtxt('charlies_samples.txt',dtype=str)
    filename = 'data/' + files[sample] + '_T1'
    
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
    
    #plt.figure('PALS histogram fit')
    #plt.errorbar(x,y,yerr=Nberr,label=label,color=CB[1])
    #plt.xlabel('Time [ns]')
    #plt.ylabel('Counts')
    #plt.legend()
    #plt.show()
    #return(float(FV), FVe, FFV3, FFV3e,label)
    return(x,y)

x,y = plot_pals(0)

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

#fit convolution func
popt, pcov = curve_fit(conv,x,y,p0=(iheight,iposition,istd,iA,il1,iB,iC,il3),bounds=(0,np.inf))

print('curve fit results', popt)
 
print("curve fit complete")

print("data loaded")



import pytensor
# 1. Force faster compilation by preventing over-optimization
pytensor.config.mode = 'FAST_COMPILE'

import pymc as pm
import pytensor.tensor as pt
import numpy as np
import arviz as az
import matplotlib.pyplot as plt

# --- DATA SETUP ---
# Ensure these variables (x and y) are defined in your environment before running
x_val = x.astype("float64")
y_val = y.astype("float64")
N = 667

# --- MATH FUNCTION ---
def matrix_conv(g, e):
    """Vectorized convolution using a Toeplitz-style matrix."""
    indices = pt.arange(N)
    diff = indices[:, None] - indices[None, :]
    mask = pt.ge(diff, 0)
    conv_matrix = pt.switch(mask, e[diff], 0)
    return pt.dot(conv_matrix, g)

if __name__ == '__main__':
    print("we're In")

# --- THE MODEL ---
    with pm.Model() as model:
        # --- Gaussian Priors (Direct Scale) ---
        h = pm.Normal("height", mu=popt[0], sigma=popt[0]*0.1, initval=popt[0])
        pos = pm.Normal("position", mu=popt[1], sigma=0.05, initval=popt[1])
        s = pm.TruncatedNormal("std", mu=popt[2], sigma=0.02, lower=0.01, initval=popt[2])
    
        # Scale A, B, C similarly
        #A = pm.Normal("A", mu=popt[3], sigma=popt[3]*0.2, initval=popt[3])
        #B = pm.Normal("B", mu=popt[4], sigma=popt[4]*0.2, initval=popt[4])
        #C = pm.Normal("C", mu=popt[5], sigma=popt[5]*0.2, initval=popt[5])
        # Dirichlet enforces that these 3 relative intensities sum to 1.
        # We initialize them using the relative proportions of your popt values.
        popt_abc = np.array([popt[3], popt[4], popt[5]])
        init_fractions = popt_abc / np.sum(popt_abc)

        intensities = pm.Dirichlet("intensities", a=np.ones(3), initval=init_fractions)
        ###    
        l1 = pm.Normal("l1", mu=popt[6], sigma=0.2, initval=popt[6])

        # Keep l1 exactly as it is
        #l1 = pm.Normal("l1", mu=popt[6], sigma=0.2, initval=popt[6])

        # Instead of making l3 an independent variable, make it an offset from l1.
        # This mathematically guarantees that l3 is ALWAYS greater than l1.
        #l3_offset = pm.HalfNormal("l3_offset", sigma=0.5, initval=max(0.1, popt[7] - popt[6]))
        #l3 = pm.Deterministic("l3", l1 + l3_offset)

        l3 = pm.Normal("l3", mu=popt[7], sigma=0.2, initval=popt[7])
    
        noise = pm.HalfNormal("sigma_noise", sigma=100)
        l2_const = 1 / 0.125
        iD_const = 25 


        # --- THE GRAPH ---
        g_sig = h * pt.exp(-(x_val - pos)**2 / (2 * s**2))
        
        # Exponential signal using the rescaled variables
        #e_sig = (A * pt.exp(-l1 * x_val)) + \
         #       (B * pt.exp(-l2_const * x_val)) + \
          #      (C * pt.exp(-l3 * x_val))
        
        #e_norm = e_sig / pt.sum(e_sig)
        # Use the intensities vector elements instead of A, B, C.
        # Since they already sum to 1, e_sig is naturally normalized!
        e_norm = (intensities[0] * pt.exp(-l1 * x_val)) + \
              (intensities[1] * pt.exp(-l2_const * x_val)) + \
              (intensities[2] * pt.exp(-l3 * x_val))
        # Convolution math
        mu_conv = matrix_conv(g_sig, e_norm)
        mu_final = mu_conv + iD_const

        # Likelihood
        y_lik = pm.Normal("y_lik", mu=mu_final, sigma=noise, observed=y_val)

        # --- EXECUTION ---
        print("Compiling and Sampling (Optimized Scale)...")
        # chains=2 allows for convergence checks; draws=200 is a good balance
        trace = pm.sample(
            draws=400, 
            tune=400, 
            chains=4, 
            target_accept=0.85, 
            init='jitter+adapt_diag'
        )

    # --- POST-PROCESSING ---
    print("\n--- RESULTS SUMMARY ---")
    # Focus on the physical parameters
    print(az.summary(trace, var_names=["height", "position", "std", "intensities", "l1", "l3"]))

    # --- PLOTTING ---
    print("\nGenerating Posterior Predictive Plot...")
    with model:
        ppc = pm.sample_posterior_predictive(trace)

    # Extract the mean of the simulated curves
    y_fit = ppc.posterior_predictive['y_lik'].mean(axis=(0, 1))

    plt.figure(figsize=(10, 6))
    plt.scatter(x_val, y_val, s=2, color='black', alpha=0.3, label='Data')
    plt.plot(x_val, y_fit, color='red', lw=2, label='Bayesian Mean Fit')
    plt.title("Convolution Fit (Scaled NUTS)")
    plt.legend()
    plt.savefig('convolution_fit6.png')
    print("Plot saved as 'convolution_fit6.png'")

    az.plot_posterior(trace, var_names=["l1", "l3"])
    plt.savefig('posterior_dist6.png')

    az.plot_trace(trace, var_names=["l1", "l3"])
    plt.savefig('trace_diagnostic6.png')