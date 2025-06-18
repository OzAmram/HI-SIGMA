from utils import *

# Draw histograms of features


working_dir = ""
plot_dir = "plots/feat_check/"
os.makedirs(plot_dir, exist_ok=True)

background = h5py.File("sim_data/bkg_bb_aa.h5", "r")
signal = h5py.File("sim_data/sm_dihiggs.h5", "r")

smear_scale = 0.2

do_shape_syst = True
feats_percentual_uncertainty = 1e-1 * np.ones(4)
variations = np.array([[0, 0, 0, 1], [0, 0, 0, -1]])
variations_names = ["plus", "minus"]

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
feat_labels = [
    "Total Mass",
    r"H$_{bb}$ Mass",
    r"H$_{\gamma\gamma}$ Mass",
    r"H$_{bb}$ $p_T$",
    r"H$_{\gamma\gamma}$ $p_T / M $",
    r"$\Delta R_{\gamma^1 b^1}$",
    r"$\Delta R_{\gamma^1 b^2}$",
    r"$\Delta R_{\gamma^2 b^1}$",
    r"$\Delta R_{\gamma^2 b^2}$",
    r"$\Delta R_{bb}$",
    r"$\Delta R_{\gamma\gamma}$",
]
bkg_features = background["feats"][:]
sig_features = signal["feats"][:]


feat_names_dict = {}
for i in range(len(feat_names)):
    feat_names_dict[feat_names[i]] = int(i)


mgg_low = 90
mgg_high = 180
bkg_mass = bkg_features[:, feat_names_dict["Hgg_mass"]]
bkg_mask_gg = (bkg_mass >= mgg_low) | (bkg_mass < mgg_high)
sig_mask_gg = [
    a and b
    for a, b in zip(
        sig_features[:, feat_names_dict["Hgg_mass"]] >= mgg_low,
        sig_features[:, feat_names_dict["Hgg_mass"]] < mgg_high,
    )
]


mass_low_sr = 115
mass_high_sr = 135
SR_mask = (bkg_mass > mass_low_sr) & (bkg_mass < mass_high_sr)
SB_mask = (bkg_mass <= mass_low_sr) | (bkg_mass >= mass_high_sr)

print(bkg_features[:, feat_names_dict["Hgg_mass"]][:10])

plt.figure()
plt.hist(
    bkg_features[:, feat_names_dict["Hgg_mass"]],
    color="blue",
    alpha=0.6,
    histtype="stepfilled",
    bins=np.linspace(mgg_low, mgg_high, 90),
    label="Background",
    density=True,
)
plt.hist(
    sig_features[:, feat_names_dict["Hgg_mass"]],
    color="red",
    alpha=0.6,
    histtype="stepfilled",
    bins=np.linspace(mgg_low, mgg_high, 90),
    label="Signal",
    density=True,
)
plt.xlabel(feat_labels[feat_names_dict["Hgg_mass"]], fontsize=20)
plt.vlines(mass_low_sr, 0, 0.1, color="red", linestyle="dashed")
plt.vlines(mass_high_sr, 0, 0.1, color="red", linestyle="dashed")
plt.legend()
plt.savefig(plot_dir + "Hgg_mass_window.pdf", bbox_inches="tight")

bkg_features = bkg_features[bkg_mask_gg]
sig_features = sig_features[sig_mask_gg]


bkg_features[:, feat_names_dict["Hgg_pt"]] /= bkg_features[
    :, feat_names_dict["Hgg_mass"]
]
sig_features[:, feat_names_dict["Hgg_pt"]] /= sig_features[
    :, feat_names_dict["Hgg_mass"]
]

# bkg_features[:, feat_names_dict["tot_mass"]] /= bkg_features[:, feat_names_dict["Hgg_mass"]]
# sig_features[:, feat_names_dict["tot_mass"]] /= sig_features[:, feat_names_dict["Hgg_mass"]]

print(np.mean(bkg_features[:, feat_names_dict["Hgg_pt"]]))


bkg_features_SB = bkg_features[bkg_mask_gg & SB_mask]
bkg_features_SR = bkg_features[SR_mask]

for i in range(sig_features.shape[1]):
    sig_thresh = np.percentile(sig_features[:, i], 99.7, axis=0)
    bkg_thresh = np.percentile(bkg_features[:, i], 99.7, axis=0)
    thresh = max(sig_thresh, bkg_thresh)
    sig_features[:, i] = np.clip(sig_features[:, i], None, thresh)
    bkg_features_SR[:, i] = np.clip(bkg_features_SR[:, i], None, thresh)
    bkg_features_SB[:, i] = np.clip(bkg_features_SB[:, i], None, thresh)


# make smeared features
smearer = SmearScaler(smear_scale)
bkg_features_smeared = smearer.fit_transform(bkg_features)
sig_features_smeared = smearer.transform(sig_features)


# plot all features for sig vs bkgs
for i in range(len(feat_names)):
    # smeared version
    plt.figure()
    feat_name = feat_names[i]
    feat_label = feat_labels[i]
    n, sig_bins, _ = plt.hist(
        sig_features_smeared[:, feat_names_dict[feat_name]],
        alpha=0.6,
        color="red",
        histtype="stepfilled",
        bins=50,
        label="Smeared Signal",
        density=True,
    )
    n, bkg_bins, _ = plt.hist(
        bkg_features_smeared[bkg_mask_gg & SB_mask, feat_names_dict[feat_name]],
        alpha=0.6,
        color="blue",
        histtype="stepfilled",
        bins=50,
        label="Smeared Background (SB)",
        density=True,
    )
    plt.hist(
        bkg_features_smeared[SR_mask, feat_names_dict[feat_name]],
        alpha=0.6,
        color="green",
        histtype="stepfilled",
        bins=bkg_bins,
        label="Smeared Background (SR)",
        density=True,
    )
    plt.xlabel(feat_label, fontsize=18)
    plt.legend()
    plt.savefig(plot_dir + "smeared_" + feat_name + ".pdf", bbox_inches="tight")

    # regular version
    plt.figure()
    feat_name = feat_names[i]
    feat_label = feat_labels[i]
    n, bins, _ = plt.hist(
        sig_features[:, feat_names_dict[feat_name]],
        alpha=0.6,
        color="red",
        histtype="stepfilled",
        bins=sig_bins,
        label="Signal",
        density=True,
    )
    n, bins, _ = plt.hist(
        bkg_features_SB[:, feat_names_dict[feat_name]],
        alpha=0.6,
        color="blue",
        histtype="stepfilled",
        bins=bkg_bins,
        label="Background (SB)",
        density=True,
    )
    plt.hist(
        bkg_features_SR[:, feat_names_dict[feat_name]],
        alpha=0.6,
        color="green",
        histtype="stepfilled",
        bins=bkg_bins,
        label="Background (SR)",
        density=True,
    )
    plt.xlabel(feat_label, fontsize=18)
    plt.legend()
    plt.savefig(plot_dir + feat_name + ".pdf", bbox_inches="tight")


for feat_name in feat_names:
    plt.figure()
    print(feat_name, np.mean(bkg_features_SR[:, feat_names_dict[feat_name]]))
    nBins = 30
    SR_feat = bkg_features_SR[:, feat_names_dict[feat_name]]
    SB_feat = bkg_features_SB[:, feat_names_dict[feat_name]]
    make_postfit_plot(
        SR_feat,
        template_samples=[SB_feat],
        template_norms=[SR_feat.shape[0]],
        labels=["SB Bkg"],
        colors=["blue"],
        bins=nBins,
        axis_label=feat_name,
        ratio_range=(0.8, 1.2),
        fname=plot_dir + "SR_%s_modeling.pdf" % feat_name,
    )


# separate transform for sig and bkg models

scaler = SmearScaler(0.1)
sig_scaler = SmearScaler(0.1)
# scaler = make_pipeline(LogitScaler(), StandardScaler())
# sig_scaler = make_pipeline(LogitScaler(), StandardScaler())

bkg_features_trans_SB = scaler.fit_transform(bkg_features_SB)
bkg_features_trans_SR = scaler.transform(bkg_features_SR)

sig_features_trans = sig_scaler.fit_transform(sig_features)

for i in range(len(feat_names)):
    plt.figure()
    feat_name = feat_names[i]
    feat_label = feat_labels[i]
    n, bins, _ = plt.hist(
        sig_features_trans[:, feat_names_dict[feat_name]],
        color="red",
        histtype="step",
        bins=50,
        label="Signal",
        density=True,
    )
    plt.hist(
        bkg_features_trans_SB[:, feat_names_dict[feat_name]],
        color="blue",
        histtype="step",
        bins=bins,
        label="Background (SB)",
        density=True,
    )
    plt.hist(
        bkg_features_trans_SR[:, feat_names_dict[feat_name]],
        color="green",
        histtype="step",
        bins=bins,
        label="Background (SR)",
        density=True,
    )
    plt.xlabel(feat_label, fontsize=18)
    plt.legend()
    plt.savefig(plot_dir + "transformed_" + feat_name + ".pdf")


if do_shape_syst:
    # plot all features for sig vs bkgs
    for nvariation, (variation, variation_name) in enumerate(
        zip(variations, variations_names)
    ):
        sig_features_syst = sig_features.copy()
        bkg_features_syst_SB = bkg_features_SB.copy()
        bkg_features_syst_SR = bkg_features_SR.copy()
        print("Applying variation: ", variation_name)
        for ni, i in enumerate([10, 9, 4, 3]):  # dR_gg, dR_bb, Hgg_pt, Hbb_pt
            print(
                "Applying variation to feature: ",
                feat_names[i],
                feat_names_dict[feat_names[i]],
            )
            print("Variation: ", variation[ni])
            print("Percentual uncertainty: ", feats_percentual_uncertainty[ni])
            sig_features_syst[:, feat_names_dict[feat_names[i]]] *= (
                1 + variation[ni] * feats_percentual_uncertainty[ni]
            )
            bkg_features_syst_SB[:, feat_names_dict[feat_names[i]]] *= (
                1 + variation[ni] * feats_percentual_uncertainty[ni]
            )
            bkg_features_syst_SR[:, feat_names_dict[feat_names[i]]] *= (
                1 + variation[ni] * feats_percentual_uncertainty[ni]
            )

        for i in range(len(feat_names)):

            plt.figure()
            feat_name = feat_names[i]
            feat_label = feat_labels[i]
            n, sig_bins, _ = plt.hist(
                sig_features_smeared[:, feat_names_dict[feat_name]],
                alpha=0.6,
                color="red",
                histtype="stepfilled",
                bins=50,
                label="Smeared Signal",
                density=True,
            )
            n, bkg_bins, _ = plt.hist(
                bkg_features_smeared[bkg_mask_gg & SB_mask, feat_names_dict[feat_name]],
                alpha=0.6,
                color="blue",
                histtype="stepfilled",
                bins=50,
                label="Smeared Background (SB)",
                density=True,
            )

            plt.clf()

            plt.figure()
            feat_name = feat_names[i]
            feat_label = feat_labels[i]
            n, bins, _ = plt.hist(
                sig_features_syst[:, feat_names_dict[feat_name]],
                alpha=0.6,
                color="red",
                histtype="stepfilled",
                bins=sig_bins,
                label="Signal",
                density=True,
            )
            n, bins, _ = plt.hist(
                bkg_features_syst_SB[:, feat_names_dict[feat_name]],
                alpha=0.6,
                color="blue",
                histtype="stepfilled",
                bins=bkg_bins,
                label="Background (SB)",
                density=True,
            )
            plt.hist(
                bkg_features_syst_SR[:, feat_names_dict[feat_name]],
                alpha=0.6,
                color="green",
                histtype="stepfilled",
                bins=bkg_bins,
                label="Background (SR)",
                density=True,
            )
            plt.xlabel(feat_label, fontsize=18)
            plt.legend()
            plt.savefig(
                plot_dir + feat_name + "_" + variation_name + ".pdf",
                bbox_inches="tight",
            )
### now group all variations together
for ni, i in enumerate([10, 9, 4, 3]):  # dR_gg, dR_bb, Hgg_pt, Hbb_pt
    plt.figure()
    feat_name = feat_names[i]
    feat_label = feat_labels[i]
    n, sig_bins, _ = plt.hist(
        sig_features[:, feat_names_dict[feat_name]],
        alpha=0.6,
        color="red",
        histtype="stepfilled",
        bins=50,
        label=r"Signal, $\nu = 0$",
        density=True,
    )
    n, bkg_bins, _ = plt.hist(
        bkg_features_SB[:, feat_names_dict[feat_name]],
        alpha=0.6,
        color="blue",
        histtype="stepfilled",
        bins=50,
        label=r"Background (SB), $\nu = 0$",
        density=True,
    )
    plt.hist(
        bkg_features_SR[:, feat_names_dict[feat_name]],
        alpha=0.6,
        color="green",
        histtype="stepfilled",
        bins=50,
        label=r"Background (SR), $\nu = 0$",
        density=True,
    )

    for nvariation, (variation, variation_name, linestyleval, nuval) in enumerate(
        zip(variations, variations_names, ["dashed", "dotted"], ["1", "-1"])
    ):
        sig_features_syst = sig_features.copy()
        bkg_features_syst_SB = bkg_features_SB.copy()
        bkg_features_syst_SR = bkg_features_SR.copy()
        print("Applying variation: ", variation_name)
        print(
            "Applying variation to feature: ",
            feat_names[i],
            feat_names_dict[feat_names[i]],
        )
        print("Variation: ", variation[ni])
        print("Percentual uncertainty: ", feats_percentual_uncertainty[ni])
        sig_features_syst[:, feat_names_dict[feat_name]] *= (
            1 + variation[ni] * feats_percentual_uncertainty[ni]
        )
        bkg_features_syst_SB[:, feat_names_dict[feat_name]] *= (
            1 + variation[ni] * feats_percentual_uncertainty[ni]
        )
        bkg_features_syst_SR[:, feat_names_dict[feat_name]] *= (
            1 + variation[ni] * feats_percentual_uncertainty[ni]
        )

        n, bins, _ = plt.hist(
            sig_features_syst[:, feat_names_dict[feat_name]],
            alpha=0.6,
            color="red",
            histtype="step",
            bins=sig_bins,
            label=r"Signal, $\nu = $" + nuval,
            density=True,
            linestyle=linestyleval,
        )
        n, bins, _ = plt.hist(
            bkg_features_syst_SB[:, feat_names_dict[feat_name]],
            alpha=0.6,
            color="blue",
            histtype="step",
            bins=bkg_bins,
            label=r"Background (SB), $\nu = $" + nuval,
            density=True,
            linestyle=linestyleval,
        )
        plt.hist(
            bkg_features_syst_SR[:, feat_names_dict[feat_name]],
            alpha=0.6,
            color="green",
            histtype="step",
            bins=bkg_bins,
            label=r"Background (SR), $\nu = $" + nuval,
            density=True,
            linestyle=linestyleval,
        )

    plt.xlabel(feat_label, fontsize=18)
    plt.legend()
    plt.savefig(plot_dir + feat_name + "all_together.pdf", bbox_inches="tight")
