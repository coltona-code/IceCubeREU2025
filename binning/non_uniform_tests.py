import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from utility import read_neutrino_simulationsNUE
from bins import *

#import data
true_table, recon_table, weights = read_neutrino_simulationsNUE()
weights = np.array(weights) * 11


true_zenith = np.cos(true_table['zenith'].to_numpy())
recon_zenith = np.cos(recon_table['zenith'].to_numpy())
true_energy = np.log10(true_table['energy'].to_numpy())
recon_energy = np.log10(recon_table['energy'].to_numpy())

starting_width = np.std((true_energy - recon_energy) / true_energy)

RANGE_START = 2.6
RANGE_END = 7.1

a = np.arange(2.6, 4, 0.05)
b = np.arange(4, 5.5, 0.15)
c = np.arange(5.5, 7.6, 0.4)
edges = np.hstack((a, b, c))


trimming_cond = (true_energy > RANGE_START) & (true_energy < RANGE_END)
weights = weights[trimming_cond]
true_energy = true_energy[trimming_cond]
recon_energy = recon_energy[trimming_cond]
true_zenith = true_zenith[trimming_cond]
recon_zenith = recon_zenith[trimming_cond]

sorted_indexes = np.argsort(true_energy)
sort_data = np.take_along_axis(true_energy, sorted_indexes, axis = -1)
sort_weights = np.take_along_axis(weights, sorted_indexes, axis = -1)

hist_true = Histogram(False)
hist_true.fill_histogram(edges, sort_data, sort_weights, sorted_indexes)

#Bin Migration Score 
#---------------------------------------------------------
# trimming_cond2 = (np.take_along_axis(recon_energy, sorted_indexes, axis = -1) > RANGE_START) & (np.take_along_axis(recon_energy, sorted_indexes, axis = -1) < RANGE_END)
# recon2 = np.take_along_axis(recon_energy, sorted_indexes, axis = -1)[trimming_cond2]
# sorted_indexes2 = np.argsort(recon2)
# sort_data2 = np.take_along_axis(recon2, sorted_indexes2, axis = -1)
# sort_weights2 = np.take_along_axis(sort_weights[trimming_cond2], sorted_indexes2, axis = -1)

# hist_recon = Histogram(False)
# hist_recon.fill_histogram(edges, sort_data2, sort_weights2, sorted_indexes[trimming_cond2])

# hist_tot = 0
# for i in range(len(hist_recon.bins)):
#     bin_tot = 0
#     for index in hist_true.bins[i].indices:
#         if index in hist_recon.bins[i].indices:
#             bin_tot += 1
#     bin_tot /= len(hist_true.bins[i].indices) + 1e-12
#     hist_tot += bin_tot

# hist_tot /= len(hist_recon.bins)

# print(f'Bin Migration Score: {hist_tot}')

print(hist_true.statistics())
#hist_true.graph_it(color = 'yellow', label=f'Log(E)')
print(hist_true.edges)