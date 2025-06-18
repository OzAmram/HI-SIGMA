from utils import *

# create datasets

working_dir = ""


background = h5py.File("sim_data/bkg_bb_aa.h5", "r")
signal = h5py.File("sim_data/sm_dihiggs.h5", "r")
data_dir = os.path.join(working_dir, "sim_data_4feats_smear0/")
# data_dir = os.path.join(working_dir,'datasets/4feats_smear15_with_enlargment/')
os.makedirs(data_dir, exist_ok=True)

selected_feats = ["Hgg_mass", "dR_gg", "dR_bb", "Hgg_pt", "Hbb_pt"]
# selected_feats = ["Hgg_mass","dR_gg", "dR_bb", "Hgg_pt", "Hbb_pt", "Hbb_mass", "dR_b1_phot2", "dR_b2_phot1", "dR_b2_phot2", "dR_b1_phot1", "tot_mass"]  # all features

# amount of smearing, in units of standard deviations
smear_scale = 0.2

# enlargment to avoid numerical issues in jacobian computations
enlargement_factor = 0.0


background.keys()


feat_names = [
    "tot_mass",
    "Hbb_mass",
    "Hgg_mass",
    "Hbb_pt",
    "Hgg_pt",
    "dR_b1_phot1",
    "dR_b1_phot2",
    "dR_b2_phot1",
    "dR_b2_phot2",
    "dR_bb",
    "dR_gg",
]
bkg_features = background["feats"][:]
sig_features = signal["feats"][:]

print(bkg_features.shape[0], sig_features.shape[0])


feat_names_dict = {}
for i in range(len(feat_names)):
    feat_names_dict[feat_names[i]] = int(i)
feat_names_dict


a, b, c = plt.hist(
    bkg_features[:, feat_names_dict["Hgg_mass"]],
    color="blue",
    bins=np.linspace(90, 200, 10),
    histtype="step",
)
a, b, c = plt.hist(
    sig_features[:, feat_names_dict["Hgg_mass"]],
    color="red",
    bins=np.linspace(90, 200, 10),
    histtype="step",
)


mgg_low = 90
mgg_high = 180
bkg_mask_gg = (bkg_features[:, feat_names_dict["Hgg_mass"]] >= mgg_low) & (
    bkg_features[:, feat_names_dict["Hgg_mass"]] < mgg_high
)
sig_mask_gg = (sig_features[:, feat_names_dict["Hgg_mass"]] >= mgg_low) & (
    sig_features[:, feat_names_dict["Hgg_mass"]] < mgg_high
)
print("nbkg, nsig in mass range %i-%i" % (mgg_low, mgg_high))
print(np.sum(bkg_mask_gg), np.sum(sig_mask_gg))


bkg_features = bkg_features[bkg_mask_gg]
sig_features = sig_features[sig_mask_gg]


# rescale Hgg pt to have reduced mass dependence
bkg_features[:, feat_names_dict["Hgg_pt"]] /= bkg_features[
    :, feat_names_dict["Hgg_mass"]
]
sig_features[:, feat_names_dict["Hgg_pt"]] /= sig_features[
    :, feat_names_dict["Hgg_mass"]
]


indexes = [feat_names_dict[i] for i in selected_feats]
print(indexes)
Nfeatures = len(indexes)

bkg_features_clean = bkg_features[:, indexes]
sig_features_clean = sig_features[:, indexes]

bkg_features_clean[:, 0] = convert_mgg(bkg_features_clean[:, 0])
sig_features_clean[:, 0] = convert_mgg(sig_features_clean[:, 0])

sb_low = convert_mgg(115.0)
sb_high = convert_mgg(135.0)


# apply smearing to non-mass features
if smear_scale > 0:
    smearer = SmearScaler(smear_scale)
    # need to apply same smearing to both sig and bkg
    bkg_features_clean[:, 1:] = smearer.fit_transform(bkg_features_clean[:, 1:])
    sig_features_clean[:, 1:] = smearer.transform(sig_features_clean[:, 1:])


np.random.seed(42)
Nevents = bkg_features_clean.shape[0]
holdout_frac = 0.3
true_S_fraction = 0.01
labels_all = st.bernoulli(p=true_S_fraction).rvs(Nevents)
trueB = int(np.sum(labels_all == 0))
trueS = int(np.sum(labels_all))
sim_S = int(len(sig_features - trueS))
print(trueB, trueS, trueS / np.sqrt(trueB))


print(data_dir)


data_all = np.zeros((Nevents, Nfeatures))
data_all[labels_all == 0] = bkg_features_clean[:trueB]
data_all[labels_all == 1] = sig_features_clean[:trueS]
sim_all = sig_features_clean[trueS:]


from sklearn.model_selection import train_test_split

data, data_holdout, labels, labels_holdout = train_test_split(
    data_all, labels_all, test_size=holdout_frac, random_state=42
)
sim, sim_holdout = train_test_split(sim_all, test_size=holdout_frac, random_state=42)

print("Train size %i, holdout size %i" % (data.shape[0], data_holdout.shape[0]))
print("Sim size %i, holdout size %i" % (sim.shape[0], sim_holdout.shape[0]))

np.save(data_dir + "data.npy", data)
np.save(data_dir + "data_holdout.npy", data_holdout)
np.save(data_dir + "labels.npy", labels)
np.save(data_dir + "labels_holdout.npy", labels_holdout)

np.save(data_dir + "sim.npy", sim)
np.save(data_dir + "sim_holdout.npy", sim_holdout)


# Setup preprocessors, don't transform
# Separate transforms for sig and bkg models
eps = 1e-8
bkg_scaler = make_pipeline(LogitScaler(eps), StandardScaler())
sig_scaler = make_pipeline(LogitScaler(eps), StandardScaler())

bkg_scaler_quantile = make_pipeline(
    LogitScaler(eps),
    QuantileTransformer(copy=True, n_quantiles=5000, output_distribution="normal"),
)
sig_scaler_quantile = make_pipeline(
    LogitScaler(eps),
    QuantileTransformer(copy=True, n_quantiles=1000, output_distribution="normal"),
)

### enlarge the range of the dataset to avoid numerical issues due to boundary effects

for nfeat in range(1, Nfeatures):
    data_min = np.min(data[:, nfeat])
    first_min = np.where(data[:, nfeat] == data_min)[0][0]
    if data_min > 0:
        data[first_min, nfeat] *= 1.0 - enlargement_factor
    else:
        data[first_min, nfeat] *= 1.0 + enlargement_factor
    data_max = np.max(data[:, nfeat])
    first_max = np.where(data[:, nfeat] == data_max)[0][0]
    if data_max > 0:
        data[first_max, nfeat] *= 1.0 + enlargement_factor
    else:
        data[first_max, nfeat] *= 1.0 - enlargement_factor
    sim_min = np.min(sim[:, nfeat])
    first_min = np.where(sim[:, nfeat] == sim_min)[0][0]
    if sim_min > 0:
        sim[first_min, nfeat] *= 1.0 - enlargement_factor
    else:
        sim[first_min, nfeat] *= 1.0 + enlargement_factor
    sim_max = np.max(sim[:, nfeat])
    first_max = np.where(sim[:, nfeat] == sim_max)[0][0]
    if sim_max > 0:
        sim[first_max, nfeat] *= 1.0 + enlargement_factor
    else:
        sim[first_max, nfeat] *= 1.0 - enlargement_factor
    print(
        "feat %i, min %f, transformed min %f, max %f,  transformed max %f"
        % (nfeat, data_min, np.min(data[:, nfeat]), data_max, np.max(data[:, nfeat]))
    )
    print(
        "feat %i, min %f, transformed min %f, max %f,  transformed max %f"
        % (nfeat, sim_min, np.min(sim[:, nfeat]), sim_max, np.max(sim[:, nfeat]))
    )

# bkg scaler from data sideband
SB_mask = (data[:, 0] < sb_low) | (data[:, 0] > sb_high)

bkg_scaler.fit(data[SB_mask, 1:])
sig_scaler.fit(sim[:, 1:])


bkg_scaler_quantile.fit(data[SB_mask, 1:])
sig_scaler_quantile.fit(sim[:, 1:])

bkg_trans = bkg_scaler_quantile.transform(data[SB_mask, 1:])
bkg_rev = bkg_scaler_quantile.inverse_transform(bkg_trans)

joblib.dump(bkg_scaler, data_dir + "bkg_scaler.pkl")
joblib.dump(sig_scaler, data_dir + "sig_scaler.pkl")

joblib.dump(bkg_scaler_quantile, data_dir + "bkg_scaler_quantile.pkl")
joblib.dump(sig_scaler_quantile, data_dir + "sig_scaler_quantile.pkl")
