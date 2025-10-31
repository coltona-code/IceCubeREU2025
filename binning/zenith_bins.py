import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from utility import read_neutrino_simulationsNUE, create_graphs2D
from bins import *

true_table, recon_table, weights = read_neutrino_simulationsNUE()
weights = np.array(weights) * 11


true_zenith = np.cos(true_table['zenith'].to_numpy())
recon_zenith = np.cos(recon_table['zenith'].to_numpy())
true_energy = np.log10(true_table['energy'].to_numpy())
recon_energy = np.log10(recon_table['energy'].to_numpy())

#print(starting_width)

RANGE_START = 2.6
RANGE_END = 7
#WIDTH = 0.24
WIDTH = (RANGE_END - RANGE_START) / 22
print(WIDTH)

trimming_cond = (true_energy > RANGE_START) & (true_energy < RANGE_END)
weights = weights[trimming_cond]
true_energy = true_energy[trimming_cond]
recon_energy = recon_energy[trimming_cond]
true_zenith = true_zenith[trimming_cond]
recon_zenith = recon_zenith[trimming_cond]

sorted_indexes = np.argsort(true_energy)
sort_data = np.take_along_axis(true_energy, sorted_indexes, axis = -1)
sort_weights = np.take_along_axis(weights, sorted_indexes, axis = -1)

hist_true = Histogram(True)
hist_true.fill_histogram(WIDTH, sort_data, sort_weights, sorted_indexes)


for b in hist_true.bins:
    b.create_sub_histogram(False, [-1, .2, .6, 1], true_zenith[b.indices], weights[b.indices], sorted_indexes[b.indices])

histo2d = []

for b in hist_true.bins:
    print(b.sub_histogram.statistics())
    histo2d.append(b.sub_histogram.get_counts())

# plt.imshow(np.array(histo2d).T, cmap='gist_ncar', interpolation='nearest', origin='lower')
# plt.show()


# print(hist_true.statistics())
print(hist_true.edges)
# hist_true.bins[10].sub_histogram.graph_it(color = 'green', label='Log(Z)')