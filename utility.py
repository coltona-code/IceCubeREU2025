import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def get_nfiles():
    nfiles = {
        21813:9998,
        21814:9998,
        21867:100,
        21868:100,
        21870:100,
        21871:100,
        21938:99,
        21939:100,
        21940:96,
    }
    return nfiles

def get_fit_params():
    astro_norm = 1.58
    astro_index = -2.53
    conv_norm = 1.02
    pr_norm = 0
    return astro_norm,astro_index,conv_norm,pr_norm

def get_pass2_ltime():
    ltime_pass2_burn = 3600*24*365*1.1 # 1 year in seconds
    return ltime_pass2_burn

def get_astro_w(PrimE,OneWeight,nevents,nfiles):
    ltime = get_pass2_ltime()
    #ltime = 1
    astro_norm,astro_index,conv_norm,pr_norm = get_fit_params()
    flux = astro_norm*10**(-18)*np.power(PrimE/100000,astro_index)*ltime
    return flux*OneWeight/(nevents*nfiles)

def get_conv_w(OneWeight,conv_weight,pass_rate,nevents,nfiles):
    ltime = get_pass2_ltime()
    #ltime = 1
    astro_norm,astro_index,conv_norm,pr_norm = get_fit_params()
    return OneWeight*2/(nevents*nfiles)*conv_norm*conv_weight*pass_rate*ltime

def get_pr_w(OneWeight,pr_weight,pass_rate,nevents,nfiles):
    ltime = get_pass2_ltime()
    #ltime = 1
    astro_norm,astro_index,conv_norm,pr_norm = get_fit_params()
    return OneWeight*2/(nevents*nfiles)*pr_norm*pr_weight*pass_rate*ltime

def read_neutrino_simulations1(flavor='nu_e', selection='cascade', data1='cscdSBU_MonopodFit4', data2='cscdSBU_MCTruth', directory='/home/colton/school/REU_2025/code/colton/hdf/hdf_finallevel/'):
    flavor_map = {
        'nu_e': (21813, 21814, 21938),
        'nu_u': (21867, 21868, 21939),
        'nu_t': (21870, 21871, 21940) 
    }
    data1_events = []
    data2_events = []
    weight_astro = []
    weight_conv = []
    weight_pr = []
    for simid in flavor_map[flavor]:
        filename = f"{directory}{simid}_{selection}.h5"
        data1_events.append(pd.read_hdf(filename,data1))
        data2_events.append(pd.read_hdf(filename,data2))

        I3MCWeightDict = pd.read_hdf(filename,"I3MCWeightDict")
        nfiles = get_nfiles()

        my_conv_jun = pd.read_hdf(filename,"mceq_HillasGaisser2012_H4a_CORSIKA_SouthPole_June_SIBYLL2.3c_conv")
        my_conv_dec = pd.read_hdf(filename,"mceq_HillasGaisser2012_H4a_CORSIKA_SouthPole_December_SIBYLL2.3c_conv")
        my_conv = (my_conv_jun+my_conv_dec)/2

        my_pr_jun = pd.read_hdf(filename,"mceq_HillasGaisser2012_H4a_CORSIKA_SouthPole_June_SIBYLL2.3c_pr")
        my_pr_dec = pd.read_hdf(filename,"mceq_HillasGaisser2012_H4a_CORSIKA_SouthPole_December_SIBYLL2.3c_pr")
        my_pr = (my_pr_jun+my_pr_dec)/2

        conv_passrate = pd.read_hdf(filename,"cscdSBU_AtmWeight_Conv_PassRate")
        pr_passrate = pd.read_hdf(filename,"cscdSBU_AtmWeight_Prompt_PassRate")

        weight_astro.append(get_astro_w(I3MCWeightDict.PrimaryNeutrinoEnergy,I3MCWeightDict.OneWeight,I3MCWeightDict.NEvents,nfiles[simid]))
        weight_conv.append(get_conv_w(I3MCWeightDict.OneWeight,my_conv.value,conv_passrate.value,I3MCWeightDict.NEvents,nfiles[simid]))
        weight_pr.append(get_pr_w(I3MCWeightDict.OneWeight,my_pr.value,pr_passrate.value,I3MCWeightDict.NEvents,nfiles[simid]))
    weight_final = pd.concat(weight_astro) + pd.concat(weight_conv) + pd.concat(weight_pr)
    return pd.concat(data1_events), pd.concat(data2_events), weight_final.to_numpy()

def read_neutrino_simulations2(flavor='nu_e', directory='/home/colton/school/REU_2025/code/Full_Finallevel_nugen_22684-22692.h5'):
    flavor_map = {
        'nu_e': (12),
        'nu_u': (14),
        'nu_t': (16) 
    }
    df = pd.read_hdf(directory)
    true_table = df.loc[abs(df['I3MCWeightDict_PrimaryNeutrinoType']) == flavor_map[flavor], ['cscdSBU_MCTruth_zenith', 'cscdSBU_MCTruth_energy']]
    recon_table = df.loc[abs(df['I3MCWeightDict_PrimaryNeutrinoType']) == flavor_map[flavor], ['cscdSBU_MonopodFit4_noDC_zenith', 'cscdSBU_MonopodFit4_noDC_energy']]
    weights = df.loc[abs(df['I3MCWeightDict_PrimaryNeutrinoType']) == flavor_map[flavor], 'weight']

    true_named = true_table.rename({"cscdSBU_MCTruth_zenith": "zenith", "cscdSBU_MCTruth_energy": "energy"}, axis="columns")
    recon_named = recon_table.rename({"cscdSBU_MonopodFit4_noDC_zenith": "zenith", "cscdSBU_MonopodFit4_noDC_energy": "energy"}, axis="columns")

    return true_named, recon_named, weights.to_numpy()

def read_neutrino_simulationsNUE(flavor = 'nu_e', directory = '/home/colton/school/REU_2025/code/newMCdataNU_E.h5'):
    df = pd.read_hdf(directory)

    true_table = df[['cscdSBU_MCTruth_zenith', 'cscdSBU_MCTruth_energy']]
    recon_table = df[['cscdSBU_MonopodFit4_noDC_zenith', 'cscdSBU_MonopodFit4_noDC_energy']]

    true_named = true_table.rename({"cscdSBU_MCTruth_zenith": "zenith", "cscdSBU_MCTruth_energy": "energy"}, axis="columns")
    recon_named = recon_table.rename({"cscdSBU_MonopodFit4_noDC_zenith": "zenith", "cscdSBU_MonopodFit4_noDC_energy": "energy"}, axis="columns")

    return true_named, recon_named, df['weight'].to_numpy()


def create_graphs2D(histograms, names, ticks, cmap, labels = None, colorbar=True):
    fig, ax = plt.subplots(1,len(histograms), layout = 'constrained')
    tot_min = np.min([np.min(i) for i in histograms])
    tot_max = np.max([np.max(i) for i in histograms])
    im = None
    if (len(histograms) == 1):
            im = ax.imshow(histograms[0], cmap=cmap, vmin=tot_min, vmax=tot_max, interpolation='nearest', origin='lower', extent=(ticks[0][0], ticks[0][1], ticks[1][0], ticks[1][1]))
            ax.set_title(names)

            x_len = len(histograms[0])
            y_len = len(histograms[0][0])
            # ax.set_xticks([0, x_len * .25, x_len / 2, x_len * .75, x_len], labels = [round(ticks[0][0], 2), round(((ticks[0][1] - ticks[0][0]) * .25) + ticks[0][0], 2), round(((ticks[0][1] - ticks[0][0]) / 2) + ticks[0][0], 2), round(((ticks[0][1] - ticks[0][0]) * 0.75) + ticks[0][0], 2), round(ticks[0][1], 2)])
            # ax.set_yticks([0, y_len * .25, y_len / 2, y_len * .75, y_len], labels = [round(ticks[1][0], 2), round(((ticks[1][1] - ticks[1][0]) * .25) + ticks[1][0], 2), round(((ticks[1][1] - ticks[1][0]) / 2) + ticks[1][0], 2), round(((ticks[1][1] - ticks[1][0]) * 0.75) + ticks[1][0], 2), round(ticks[1][1], 2)])
            if labels:
                ax.set_xlabel(labels[0])
                ax.set_ylabel(labels[1])
    else:
        for i in range(len(histograms)):
            im = ax[i].imshow(histograms[i], cmap=cmap, vmin=tot_min, vmax=tot_max, interpolation='nearest', origin='lower', extent=(ticks[i][0][0], ticks[i][0][1], ticks[i][1][0], ticks[i][1][1]))
            ax[i].set_title(names[i])

            x_len = len(histograms[i])
            y_len = len(histograms[i][0])
            # ax[i].set_xticks([0, x_len * .25, x_len / 2, x_len * .75, x_len], labels = [round(ticks[i][0][0], 2), round(((ticks[i][0][1] - ticks[i][0][0]) * .25) + ticks[i][0][0], 2), round(((ticks[i][0][1] - ticks[i][0][0]) / 2) + ticks[i][0][0], 2), round(((ticks[i][0][1] - ticks[i][0][0]) * 0.75) + ticks[i][0][0], 2), round(ticks[i][0][1], 2)])
            # ax[i].set_yticks([0, y_len * .25, y_len / 2, y_len * .75, y_len], labels = [round(ticks[i][1][0], 2), round(((ticks[i][1][1] - ticks[i][1][0]) * .25) + ticks[i][1][0], 2), round(((ticks[i][1][1] - ticks[i][1][0]) / 2) + ticks[i][1][0], 2), round(((ticks[i][1][1] - ticks[i][1][0]) * 0.75) + ticks[i][1][0], 2), round(ticks[i][1][1], 2)])
            if labels:
                ax[i].set_xlabel(labels[i][0])
                ax[i].set_ylabel(labels[i][1])
    if colorbar:
        fig.colorbar(im, ax=ax, shrink = 0.5)
    plt.show()
 
def create_graphs1D(histograms, names, ticks, color, sharey = False):
    fig, ax = plt.subplots(1,len(histograms), sharey=sharey)
    fig.tight_layout()
    for i in range(len(histograms)):
        x_len = len(histograms[i])
        ax[i].bar(range(len(histograms[i])), height = histograms[i], width=1, align='edge', edgecolor = 'black', color=color, alpha=0.5)
        ax[i].set_xticks([0, x_len / 4, x_len / 2, x_len * .75, x_len], labels = [ticks[i][0], ticks[i][1] / 4, ticks[i][1] / 2, ticks[i][1] * .75, ticks[i][1]])
        ax[i].set_title(names[i])
    plt.show()
