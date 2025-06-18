from utils import *
from train_classifier import train_classifier

# train a classifier between truth bkg and generated bkg in SR
data_dir = os.path.join("datasets/4feats_smear15/")
data = np.load(data_dir + "data.npy")
labels = np.load(data_dir + "labels.npy")


background_dir = "bkg_models/4feats_smear15/"
plot_dir_all = "plots/bkg_4feats_bkg_check_may20_smear15/boot%i/"
device = "cuda"


nboots = 5
new_model = True
mass_in_cls = False
nepochs = 50
sideband = True
# bkg_scaler = joblib.load(data_dir + 'bkg_scaler_quantile.pkl')
bkg_scaler = joblib.load(data_dir + "bkg_scaler.pkl")


# flow params
num_layers = 5
hidden_size = 32
flow_nbins = 24
RQS = True

cls_layers = [32, 64, 64, 16]


nEvents = None


data = data[:nEvents]
labels = labels[:nEvents]

# Filter out all signal
sig_mask = labels.reshape(-1) < 0.5
data = data[sig_mask]
labels = labels[sig_mask]


# account for any non-invertable transform (ie smearing)
data[:, 1:] = bkg_scaler.inverse_transform(bkg_scaler.transform(data[:, 1:]))
print(np.sum(np.isnan(data)))

# remove nans
for i in range(data.shape[1]):
    data[:, i] = clean_nans(data[:, i])

data_masses = data[:, :1]
data_feats = data[:, 1:]

mass_low_sr = convert_mgg(115.0)
mass_high_sr = convert_mgg(135.0)

# check modeling in 'validation region' of SB's right next to SR
mass_low_vr = convert_mgg(110.0)
mass_high_vr = convert_mgg(140.0)

SR_mask = ((data_masses > mass_low_sr) & (data_masses < mass_high_sr)).reshape(-1)
# SB_mask = ~SR_mask
SB_mask = (
    ((data_masses > mass_low_vr) & (data_masses < mass_low_sr))
    | ((data_masses > mass_high_sr) & (data_masses < mass_low_vr))
).reshape(-1)

SR_data_feats = data_feats[SR_mask]
SR_data_masses = data_masses[SR_mask]
SR_data_train = data[SR_mask] if mass_in_cls else SR_data_feats

data_feats = data_feats[SB_mask]
data_masses = data_masses[SB_mask]
data_train = data[SB_mask] if mass_in_cls else data_feats


Nensemble = 5
nfeatures = data_feats.shape[1]


for iboot in range(nboots):
    print("\n\n Boot %i" % iboot)

    plot_dir = plot_dir_all % iboot
    os.makedirs(plot_dir, exist_ok=True)

    samples = []
    SR_samples = []

    for iensemble in range(Nensemble):
        model_path = os.path.join(
            background_dir, "background_model_boot%i_%i.par" % (iboot, iensemble)
        )
        flowB = flow_model(
            nfeatures,
            num_layers=num_layers,
            hidden_size=hidden_size,
            num_bins=flow_nbins,
        )
        flowB.load_state_dict(torch.load(model_path, weights_only=True))
        flowB.eval()

        sample = (
            flowB.sample(
                num_samples=1, context=torch.tensor(data_masses, dtype=torch.float32)
            )[:, 0]
            .detach()
            .numpy()
        )
        sample = bkg_scaler.inverse_transform(sample)
        if mass_in_cls:
            sample = np.concatenate((data_masses, sample), axis=1)
        samples.append(sample)

        SR_sample = (
            flowB.sample(
                num_samples=1, context=torch.tensor(SR_data_masses, dtype=torch.float32)
            )[:, 0]
            .detach()
            .numpy()
        )
        SR_sample = bkg_scaler.inverse_transform(SR_sample)
        if mass_in_cls:
            SR_sample = np.concatenate((SR_data_masses, SR_sample), axis=1)
        SR_samples.append(SR_sample)

    samples = np.concatenate(samples)
    SR_samples = np.concatenate(SR_samples)

    nBins = 30
    for i in range(nfeatures):
        bkg_norm = data_feats.shape[0]
        thresh = np.percentile(data_feats[:, i], 99.8)
        feat_clean = np.clip(data_feats[:, i], None, thresh)
        samples_idx = i + 1 if mass_in_cls else i
        sample_clean = np.clip(samples[:, samples_idx], None, thresh)
        make_postfit_plot(
            feat_clean,
            template_samples=[sample_clean],
            template_norms=[bkg_norm],
            labels=["Bkg Model Template"],
            colors=["blue"],
            bins=nBins,
            axis_label="Feature %i" % i,
            ratio_range=(0.8, 1.2),
            fname=plot_dir + "SB_feat%i_modeling.png" % i,
        )

    # train a classifier to distinguish real vs synthetic in SB's right next to SR
    X = np.concatenate((samples, data_train))
    Y = np.concatenate(
        (np.zeros(samples.shape[0]), np.ones(data_train.shape[0]))
    ).reshape(-1, 1)
    print(Y.shape)
    weights = np.ones(Y.shape)
    weights[Y == 0] = 1.0 / Nensemble

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    X_SR = np.concatenate((SR_samples, SR_data_train))
    Y_SR = np.concatenate(
        (np.zeros(SR_samples.shape[0]), np.ones(SR_data_train.shape[0]))
    ).reshape(-1, 1)

    X_SR = scaler.transform(X_SR)

    test_frac = 0.2
    X_train_, X_test, Y_train_, Y_test, W_train, W_test = train_test_split(
        X, Y, weights, test_size=test_frac, random_state=42
    )
    X_train, X_val, Y_train, Y_val, W_train, W_val = train_test_split(
        X_train_, Y_train_, W_train, test_size=test_frac, random_state=42
    )
    batch_size = 256

    print("train on %i, test on %i" % (X_train.shape[0], X_test.shape[0]))
    print(X_train.shape)

    train_cls_dataset = xyDataset(X_train, Y_train, weights=W_train).to(device)
    train_cls_dataset_batched = DataLoader(
        train_cls_dataset, batch_size=batch_size, shuffle=True
    )

    val_cls_dataset = xyDataset(X_val, Y_val, weights=W_train).to(device)
    val_cls_dataset_batched = DataLoader(
        val_cls_dataset, batch_size=batch_size, shuffle=True
    )

    classifier = train_classifier(
        train_cls_dataset_batched,
        val_dataset=val_cls_dataset_batched,
        model_path=plot_dir + "bkg_cls.pth",
        weights=True,
        device=device,
        nepochs=nepochs,
        new_model=new_model,
        layers=cls_layers,
    )

    # also copy to model directory
    torch.save(classifier.state_dict(), background_dir + "bkg_boot%i_cls.pth" % iboot)
    joblib.dump(scaler, background_dir + "bkg_boot%i_cls_scaler.pkl" % iboot)

    classifier.cpu()
    preds_test = (
        classifier(torch.tensor(X_test, dtype=torch.float32)).detach().cpu().numpy()
    )
    fpr, tpr, _ = roc_curve(Y_test, preds_test)
    auc = roc_auc_score(Y_test, preds_test)
    print("test auc %.3f" % auc)

    plt.figure()
    plt.plot(fpr, tpr, label="bkg-bkg classifier (%.3f)" % auc)
    plt.plot(fpr, fpr, label="Random")

    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend(loc="upper right")
    plt.savefig(plot_dir + "roc.png")

    plt.figure()
    bkg_likelihood_ratio = preds_test[Y_test < 0.5] / (1.0 - preds_test[Y_test < 0.5])
    plt.hist(bkg_likelihood_ratio, bins=20)
    plt.xlabel("Likelihood Ratio : Truth / Model")
    plt.savefig(plot_dir + "SB_likelihood_ratio.png")

    # replot with weights
    preds_train = (
        classifier(torch.tensor(X_train, dtype=torch.float32)).detach().cpu().numpy()
    )

    # eval ratio on synthetic samples
    likelihood_ratio = preds_train[Y_train < 0.5] / (1.0 - preds_train[Y_train < 0.5])
    likelihood_ratio /= np.mean(likelihood_ratio)
    data_test = X_train[Y_train.reshape(-1) > 0.5]
    sample_test = X_train[Y_train.reshape(-1) < 0.5]
    sample_norm = data_test.shape[0]
    nBins = 30

    offset = 1 if mass_in_cls else 0
    for i in range(offset, nfeatures + offset):
        thresh = np.percentile(data_test[:, i], 99.8)
        feat_clean = np.clip(data_test[:, i], None, thresh)
        sample_clean = np.clip(sample_test[:, i], None, thresh)

        make_postfit_plot(
            feat_clean,
            template_samples=[sample_clean],
            template_norms=[sample_norm],
            template_weights=[likelihood_ratio],
            labels=["RW Bkg Model Template"],
            colors=["blue"],
            bins=nBins,
            axis_label="Feature %i" % (i - 1),
            ratio_range=(0.8, 1.2),
            fname=plot_dir + "SB_reweighted_feat%i_modeling.png" % (i - offset),
        )

    # Check modeling in the signal region,
    # with and without the reweighting from the classifier
    SR_preds = (
        classifier(torch.tensor(X_SR, dtype=torch.float32)).detach().cpu().numpy()
    )
    likelihood_ratio = SR_preds[Y_SR < 0.5] / (1.0 - SR_preds[Y_SR < 0.5])
    likelihood_ratio /= np.mean(likelihood_ratio)

    data_test = X_SR[Y_SR.reshape(-1) > 0.5]
    sample_test = X_SR[Y_SR.reshape(-1) < 0.5]
    sample_norm = data_test.shape[0]
    nBins = 30
    for i in range(offset, nfeatures + offset):
        thresh = np.percentile(data_test[:, i], 99.8)
        feat_clean = np.clip(data_test[:, i], None, thresh)
        sample_clean = np.clip(sample_test[:, i], None, thresh)

        make_postfit_plot(
            feat_clean,
            template_samples=[sample_clean],
            template_norms=[sample_norm],
            labels=["Bkg Model Template"],
            colors=["blue"],
            bins=nBins,
            axis_label="Feature %i" % (i - 1),
            ratio_range=(0.8, 1.2),
            fname=plot_dir + "SR_feat%i_modeling.png" % (i - offset),
        )

        make_postfit_plot(
            feat_clean,
            template_samples=[sample_clean],
            template_norms=[sample_norm],
            template_weights=[likelihood_ratio],
            labels=["RW Bkg Model Template"],
            colors=["blue"],
            bins=nBins,
            axis_label="Feature %i" % (i - 1),
            ratio_range=(0.8, 1.2),
            fname=plot_dir + "SR_reweighted_feat%i_modeling.png" % (i - offset),
        )
