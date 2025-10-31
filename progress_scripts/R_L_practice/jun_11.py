#example 1 from paper
import numpy as np 
import matplotlib.pyplot as plt
import math
from scipy.ndimage import gaussian_filter1d
#rng generator from numpy
rng = np.random.default_rng()
#create fake data, in the same way they did in paper
normal1 = rng.normal(0.3, 0.10, 25000)
normal2 = rng.normal(0.75, 0.08, 15000)
uniform = rng.uniform(0, 1, 10000)
#combine data
true_data = np.hstack((normal1, normal2, uniform))
#smear the data
recon_data = gaussian_filter1d(true_data, (1 / (2 * math.pi * 0.07))) 





#this is useful, but maybe if I write a full histogram class this will go away
def get_middles(edges):
    middles = []
    for i in range(1, len(edges)):
        middles.append((edges[i] + edges[i - 1]) / 2)
    return middles

#smear the data
hist, edges = np.histogram(true_data, bins=40, range=(0, 1))
middles_40 = get_middles(edges)
#formula for sigma from here: https://en.wikipedia.org/wiki/Gaussian_filter
smeared = gaussian_filter1d(hist, (1 / (2 * math.pi * 0.07))) 
plt.plot(middles_40, smeared, 'rh')

ITERATIONS = 15

middles_20 = np.linspace(start=0, stop=1, num=20)
unfolded = np.array([2500/2] * 20) #starting with uniform
A = rng.normal(0.07, 0.07, (len(smeared), len(unfolded))) #random until I figure out how to define it
#probabiltyu true value is some value and reconstucted is some value
plt.plot(middles_20, unfolded, 'bo')
for _ in range(ITERATIONS):
    d_k = np.matvec(A, unfolded.reshape(-1, 20))
    for j in range(len(unfolded)):
        new_bin_j = 0
        alpha_j = 0
        for i in range(len(smeared)):
            alpha_j += A[i][j]
        for i in range(len(smeared)):
            new_bin_j += A[i][j] * unfolded[j] * (smeared[i] / (d_k[0][i] * alpha_j))
        unfolded[j] = new_bin_j
plt.plot(middles_20, unfolded, 'mo')

#put it on a histogram
plt.hist(true_data, bins=40, range=(0, 1), edgecolor = 'black')
plt.show()