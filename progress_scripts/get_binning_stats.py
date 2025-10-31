import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from utility import read_neutrino_simulationsNUE
from bins import *

true_table, recon_table, weights = read_neutrino_simulationsNUE()
weights = np.array(weights) * 10


true_zenith = np.cos(true_table['zenith'].to_numpy())
recon_zenith = np.cos(recon_table['zenith'].to_numpy())
true_energy = np.log10(true_table['energy'].to_numpy())
recon_energy = np.log10(recon_table['energy'].to_numpy())

starting_width = np.std((true_energy - recon_energy) / true_energy)

#mid - 3 to 5

trimming_cond = (true_energy > 3) & (true_energy < 5)
weights = weights[trimming_cond]
true_energy = true_energy[trimming_cond]
recon_energy = recon_energy[trimming_cond]
true_zenith = true_zenith[trimming_cond]
recon_zenith = recon_zenith[trimming_cond]

sorted_indexes = np.argsort(true_energy)
sort_data = np.take_along_axis(true_energy, sorted_indexes, axis = -1)
sort_weights = np.take_along_axis(weights, sorted_indexes, axis = -1)

trimming_cond2 = (np.take_along_axis(recon_energy, sorted_indexes, axis = -1) > 3) & (np.take_along_axis(recon_energy, sorted_indexes, axis = -1) < 5)
recon2 = np.take_along_axis(recon_energy, sorted_indexes, axis = -1)[trimming_cond2]
sorted_indexes2 = np.argsort(recon2)
sort_data2 = np.take_along_axis(recon2, sorted_indexes2, axis = -1)
sort_weights2 = np.take_along_axis(sort_weights[trimming_cond2], sorted_indexes2, axis = -1)



futuredf = []
bin_size = np.linspace(starting_width, starting_width * 30, 100)

for width in bin_size:
    print(width)
    hist_true = Histogram(uniform = True)
    hist_recon = Histogram(uniform = True)
    hist_true.fill_histogram(width, sort_data, sort_weights, sorted_indexes)
    hist_recon.fill_histogram(width, sort_data2, sort_weights2, sorted_indexes[trimming_cond2])

    hist_tot = 0
    for i in range(len(hist_recon.bins)):
        bin_tot = 0
        for index in hist_true.bins[i].indices:
            if index in hist_recon.bins[i].indices:
                bin_tot += 1
        bin_tot /= len(hist_true.bins[i].indices)
        hist_tot += bin_tot
    
    hist_tot /= len(hist_recon.bins)
    point_data = hist_true.get_point_data()
    bin_data = hist_true.get_bin_data()

    futuredf.append({
        'bin_width' : width,
        'num_bins'  : len(hist_true.bins),
        'min_stats' : point_data[0],
        'max_stats' : point_data[1],
        'mean_stats': point_data[2],
        'min_value' : bin_data[0],
        'max_value' : bin_data[1],
        'mean_value': bin_data[2], 
        'resol_score' : hist_tot
    })

df = pd.DataFrame(futuredf)
df.to_csv('/home/colton/school/REU_2025/code/bin_resol_data3_5.csv')

