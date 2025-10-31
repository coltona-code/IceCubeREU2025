import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from utility import read_neutrino_simulations

#get full reconstructed and true data
recon_table, true_table = read_neutrino_simulations(flavor='nu_e', directory='./colton/hdf/hdf_finallevel/')

#Energy Resolution Data
mode = 'energy'
sigma_array = np.log10((true_table[mode] / recon_table[mode])) / np.log10(true_table[mode])
log_true = np.log10(true_table[mode])
plt.xlabel('log(true)')

#Zenith resolution data
# mode = 'zenith'
# sigma_array = (np.cos(true_table[mode]) - np.cos(recon_table[mode]))
# log_true = np.cos(true_table[mode])
# plt.xlabel('cos(true)')

#Plot all points
#plt.plot(log_true, sigma_array, 'b.')

#sort by log_true (x-axis)
sorted_indexs = np.argsort(log_true)
sigma_array = np.take_along_axis(sigma_array.to_numpy(), sorted_indexs, axis=-1)
log_true = np.take_along_axis(log_true.to_numpy(), sorted_indexs, axis=-1)

diff = 0.0302
sigma_mean = []
sigma_std = []
log_true_final = []

current = 0

point_count = []

for i in range(0, len(log_true)):
    #group them only when difference between last is diff
    if log_true[i] - log_true[current] >= diff:
        point_count.append(i - current)
        #add means to corresponding arrays
        log_true_final.append(np.mean(log_true[current:i]))
        mean = np.mean(sigma_array[current:i])
        sigma_mean.append(mean)
        sigma_std.append(np.std(sigma_array[current:i], mean=mean))
        #new value of current
        current = i
#getting any values that may have been missed at the end
log_true_final.append(np.mean(log_true[current:]))
mean = np.mean(sigma_array[current:])
sigma_mean.append(mean)
sigma_std.append(np.std(sigma_array[current:], mean=mean))

print(np.mean(point_count))
print(len(point_count))

#plots
plt.plot(log_true_final, sigma_mean, 'k.', label='Mean')
# plt.plot(log_true_final, np.add(sigma_mean,sigma_std), 'k--', label='1 std')
# plt.plot(log_true_final, np.subtract(sigma_mean, sigma_std), 'k--')
plt.ylabel('Resolution Error')
plt.title(mode)
plt.legend()
plt.show()