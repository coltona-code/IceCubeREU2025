import numpy as np
import h5py as h5

def writeAMatrix(AMatrix, alpha, path):
    with h5.File(path, "w") as f:
        f.create_dataset("AMatrix", data=AMatrix)
        f.create_dataset('alpha', data = alpha)

def readAMatrix(path):
    with h5.File(path, "r") as f:
        return f['AMatrix'][:], f['alpha'][:]

def RLUnfolding1D(ITERATIONS, AMatrix, alpha, recon, unfolded_bins, unfolded_size):
    #Implementation of the R-L algorithm
    unfolded = np.array([unfolded_size] * (unfolded_bins))
    for _ in range(ITERATIONS):
        d_k = np.matmul(AMatrix, unfolded)
        ratio = recon / (d_k + 1e-12)  # avoid division by zero
        unfolded = unfolded * (AMatrix.T @ ratio) / (alpha + 1e-12)
    return unfolded

