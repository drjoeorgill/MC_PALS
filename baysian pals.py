#baysian pals

import pymc as pm
import numpy as np
import matplotlib.pyplot as plt
import pytensor.tensor as pt


samples = np.loadtxt("charlies_samples.txt", dtype = str)

print(samples[0])

y = np.loadtxt(samples[0]+"_T1.dat", skiprows=4)
plt.figure()
plt.plot(y)
plt.show()


