import matplotlib.pyplot as plt
import numpy as np
import sys, h5py

inputFile = sys.argv[1]
odir = sys.argv[2]

f = h5py.File(inputFile, "r")

feats = f['feats'][:]
print(feats.shape)
feat_names = ["tot_mass", "Hbb_mass", "Hgg_mass", "Hbb_pt", "Hgg_pt", "dR_b1_phot1", "dR_b1_phot2", "dR_b2_phot1", "dR_b2_phot2", "dR_bb", "dR_gg"]

for i in range(len(feat_names)):
    feat = feats[:, i]

    plt.figure()
    plt.hist(feat, bins = 50)
    plt.xlabel(feat_names[i], fontsize=24)
    plt.savefig(odir + feat_names[i] + ".png", bbox_inches='tight')

