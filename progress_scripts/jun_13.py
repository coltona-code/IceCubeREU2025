#Create an A matrix based on a model where the true values for every event is constant, and there is a 10% resolution error
import numpy as np
import matplotlib.pyplot as plt

#number of data points 
N = 10000
#resolution
sigma = 0.1
#Create N data points, with a true (mean) value of 1 TeV, and a resolution error of sigma
#rng generator from numpy
rng = np.random.default_rng()
recon_data = rng.normal(1, sigma, N)

#number of reconstructed bins, a constant for now
bin_count = 10
#edges of the bins of the histogram, for use linking each element to a bin, letting numpy automaticlly find the edges
edges = np.delete(np.histogram_bin_edges(recon_data, bin_count), 0)

#create the empty A matrix, the 1 is because the second dim is only 1 right now
A = np.zeros((bin_count, 1))
#this adds 1 to each bin in A based on the index
np.add.at(A, np.digitize(recon_data, edges, right=True), 1)
#divide every value by total (which here is N) to get a probablility
np.divide(A, N, out=A)

#plt.hist(recon_data, bins=bin_count, edgecolor="black")
#plt.plot(edges, A.reshape(bin_count), 'ks')
#plt.yscale('log')
plt.imshow(A, cmap='hot', interpolation='nearest')
plt.show()