import numpy as np
import h5py as h5
import matplotlib.pyplot as plt



def writeAMatrix(AMatrix, alpha, path):
    with h5.File(path, "w") as f:
        f.create_dataset("AMatrix", data=AMatrix)
        f.create_dataset('alpha', data = alpha)

def readAMatrix(path):
    with h5.File(path, "r") as f:
        return f['AMatrix'][:], f['alpha'][:]

def cal_llh(hist_1, hist_2):
    return (-2 * (np.sum(hist_1 * np.log(hist_2 + 1e-12) - hist_2) - np.sum(hist_1 * np.log(hist_1 + 1e-12) - hist_1)))

def RLUnfolding1D(AMatrix, alpha, recon, unfolded_start_value, unfolded_bin_count):
    #Implementation of the R-L algorithm
    unfoleded = np.array([unfolded_start_value] * unfolded_bin_count)
    sampled_recon = [cal_llh(rng.poisson(recon), recon) for _ in range(100000)]
    iter_count = 0
    N_iter = np.matmul(AMatrix, unfolded)
    llh = cal_llh(recon, N_iter)

    while llh > np.median(sampled_recon):
        d_k = np.matmul(AMatrix, unfolded)
        ratio = recon / (d_k + 1e-12)  # avoid division by zero
        unfolded = unfolded * (AMatrix.T @ ratio) / (alpha + 1e-12)
        llh = cal_llh(recon, d_k)
        iter_count += 1

    for _ in range(iter_count):
        d_k = np.matmul(AMatrix, unfolded)
        ratio = recon / (d_k + 1e-12)  # avoid division by zero
        unfolded = unfolded * (AMatrix.T @ ratio) / (alpha + 1e-12)
    return unfolded

def RLUnfolding2D(AMatrix, alpha, recon, unfolded_start_value, true_x_bin_count, true_y_bin_count, recon_x_bin_count, recon_y_bin_count):
    #2D impletmentation
    flat_recon_bins = recon.flatten()
    flat_A = A.transpose(0, 2, 1, 3).reshape(recon_x_bin_count * recon_y_bin_count, true_x_bin_count * true_y_bin_count)
    flat_alpha = alpha.flatten()
    unfolding = RLUnfolding1D(flat_A, flat_alpha, flat_recon_bins, unfolded_start_value, true_x_bin_count * true_y_bin_count)
    return unfolded.reshape(true_bin_energy_count, true_bin_zenith_count)

def create_final_plot(recon_binned, true_binned, final_unfolded, xedges_recon, yedges_recon, xedges_true, yedges_true):
    X_recon, Y_recon = np.meshgrid(xedges_recon, yedges_recon)
    X_true, Y_true = np.meshgrid(xedges_true, yedges_true)
    tot_min = np.min([np.min(i) for i in (recon_binned, true_binned, final_unfolded)])
    tot_max = np.max([np.max(i) for i in (recon_binned, true_binned, final_unfolded)])
    fig, ax = plt.subplots(1, 3, layout = 'constrained')
    fig.set_figheight(4)
    fig.set_figwidth(13)

    ax[0].pcolormesh(X_recon, Y_recon, recon_binned.T, cmap = 'magma', norm=LogNorm(vmin=.01, vmax=tot_max))
    ax[0].set_title('Reconstructed')
    ax[0].set_xlabel('Log(Energy / GeV)')
    ax[0].set_ylabel('Cos(Zenith Angle)')

    im = ax[1].pcolormesh(X_true, Y_true, true_binned.T, cmap = 'magma', norm=LogNorm(vmin=.01, vmax=tot_max))
    ax[1].set_title('True')
    ax[1].set_xlabel('Log(Energy / GeV)')
    ax[1].set_ylabel('Cos(Zenith Angle)')
    ax[1].set_xticks([2.6, 3, 4, 5, 6, 7])

    ax[2].pcolormesh(X_true, Y_true, final_unfolded.T, cmap = 'magma', norm=LogNorm(vmin=.01, vmax=tot_max))
    ax[2].set_title('Unfolded')
    ax[2].set_xlabel('Log(Energy / GeV)')
    ax[2].set_ylabel('Cos(Zenith Angle)')
    ax[2].set_xticks([2.6, 3, 4, 5, 6, 7])

    cbar = fig.colorbar(im, ax=ax, shrink = 1)
    cbar.set_label('Weighted # of events', rotation=270)
    plt.show()