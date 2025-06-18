from utils import *

working_dir = ""
data_dir = os.path.join(working_dir, "sim_data_4feats_v5/")

# background_dir = 'background_model_trans_2feats_v3/'
background_dir = "background_model_4feats_RQS_v5/"
signal_dir = "signal_model_4feats_RQS_v5/"
plot_dir = fit_dir = "plots/fits_april22_4feats_v5_n250/"
supervised_dir = "supervised_classifier_4feats_v5/"


# load classifiers

cls_full_model_path = supervised_dir + "model_full.par"
cls_no_mass_model_path = supervised_dir + "model_no_mass.par"

layers = [16, 32, 32, 8]
dropout = 0.2


# flow params
bkg_flow_num_layers = 4
bkg_flow_hidden_size = 32
bkg_flow_num_bins = 16
RQS = True

sig_flow_num_layers = 4
sig_flow_hidden_size = 32
sig_flow_num_bins = 16

Nbootstraps = 5
Nbootstraps_sim = 1
Nensemble = 5
device = "cpu"

evt_start = 0
evt_stop = None


os.makedirs(plot_dir, exist_ok=True)
data_holdout = np.load(data_dir + "data_holdout.npy")
labels_holdout = np.load(data_dir + "labels_holdout.npy")

sim_holdout = np.load(data_dir + "sim_holdout.npy")

bkg_scaler = joblib.load(data_dir + "bkg_scaler.pkl")
sig_scaler = joblib.load(data_dir + "sig_scaler.pkl")


data_feats = data_holdout[evt_start:evt_stop, 1:]
data_masses = data_holdout[evt_start:evt_stop, 0:1].reshape(-1)
labels_holdout = labels_holdout[evt_start:evt_stop]

nfeatures = data_feats.shape[1]

mgg_low = 90.0
mgg_high = 180.0
mass_low_sr = (115 - mgg_low) / (mgg_high - mgg_low)
mass_high_sr = (135 - mgg_low) / (mgg_high - mgg_low)
SR_mask = (data_masses > mass_low_sr) & (data_masses < mass_high_sr)


data_feats = data_feats[SR_mask]
labels_holdout = labels_holdout[SR_mask]
data_masses = data_masses[SR_mask]


bkg_inputs = bkg_scaler.transform(data_feats)
sig_inputs = sig_scaler.transform(data_feats)

print("initial Nans")
print(np.sum(np.isnan(bkg_inputs)), np.sum(np.isnan(sig_inputs)))
# cleanup outliers, different procedure for high vs low nans
mean_data_feats = np.mean(data_feats, axis=0)
cutoff = 5.0
for i in range(data_feats.shape[1]):
    above_avg = data_feats[:, i] > mean_data_feats[i]
    bkg_inputs[:, i] = clean_nans(bkg_inputs[:, i], above_avg)
    sig_inputs[:, i] = clean_nans(sig_inputs[:, i], above_avg)

bkg_inputs = np.clip(bkg_inputs, -cutoff, cutoff)
sig_inputs = np.clip(sig_inputs, -cutoff, cutoff)


bkg_inputs = torch.tensor(bkg_inputs, dtype=torch.float32)
sig_inputs = torch.tensor(sig_inputs, dtype=torch.float32)

bkg_flow_prob = np.zeros((Nbootstraps, len(data_feats)))
sig_flow_prob = np.zeros((Nbootstraps_sim, len(data_feats)))

for nboot in range(Nbootstraps):
    models = []
    for iensemble in range(Nensemble):
        model_path = os.path.join(
            background_dir, "background_model_boot%i_%i.par" % (nboot, iensemble)
        )
        flowB = flow_model(
            nfeatures,
            num_layers=bkg_flow_num_layers,
            hidden_size=bkg_flow_hidden_size,
            RQS=RQS,
            num_bins=bkg_flow_num_bins,
        )
        flowB.load_state_dict(torch.load(model_path, weights_only=True))
        flowB.to(device)
        flowB.eval()

        bkg_flow_prob[nboot] += (
            flowB.log_prob(
                inputs=bkg_inputs,
                context=torch.tensor(data_masses.reshape(-1, 1), dtype=torch.float32),
            )
            .cpu()
            .detach()
            .numpy()
        )
        models.append(flowB)

    bkg_flow_prob[nboot] /= Nensemble
    bkg_flow_prob[nboot] = np.exp(bkg_flow_prob[nboot])


for nboot in range(Nbootstraps_sim):
    models = []
    for iensemble in range(Nensemble):
        model_path = os.path.join(
            signal_dir, "signal_model_boot%i_%i.par" % (nboot, iensemble)
        )
        flowS = flow_model(
            nfeatures,
            num_layers=sig_flow_num_layers,
            hidden_size=sig_flow_hidden_size,
            RQS=RQS,
            num_bins=sig_flow_num_bins,
        )
        flowS.load_state_dict(torch.load(model_path, weights_only=True))
        flowS.to(device)
        flowS.eval()

        sig_flow_prob[nboot] += (
            flowS.log_prob(
                inputs=sig_inputs,
                context=torch.tensor(data_masses.reshape(-1, 1), dtype=torch.float32),
            )
            .cpu()
            .detach()
            .numpy()
        )
        models.append(flowS)

    sig_flow_prob[nboot] /= Nensemble
    sig_flow_prob[nboot] = np.exp(sig_flow_prob[nboot])


dcb_params = joblib.load(fit_dir + "dcb_fit.pkl")
bkg_params = joblib.load(fit_dir + "m_only_fit.pkl")["params"]
print(bkg_params)
fit_fn = ExpShape(np.min(data_masses), np.max(data_masses), order=3)
bkg_mass_prob = fit_fn.density(data_masses, bkg_params[1:])
sig_mass_prob = cb.pdf(data_masses, *dcb_params)

flow_learned_ratio_no_mass = sig_flow_prob[0] / np.mean(bkg_flow_prob, axis=0)
density_ratio = flow_learned_ratio_no_mass * sig_mass_prob / bkg_mass_prob

print(data_masses.reshape(-1, 1).shape)

print(np.mean(data_feats[labels_holdout.reshape(-1) < 0.5], axis=0))
print(np.std(data_feats[labels_holdout.reshape(-1) < 0.5], axis=0))
vals_full = np.concatenate((data_masses.reshape(-1, 1), data_feats), axis=1)


classifier_full = DNN(
    input_dim=vals_full.shape[1], layers=layers, dropout_probability=dropout
)
classifier_full.load_state_dict(torch.load(cls_full_model_path))
classifier_full.eval()

classifier_no_mass = DNN(
    input_dim=data_feats.shape[1], layers=layers, dropout_probability=dropout
)
classifier_no_mass.load_state_dict(torch.load(cls_no_mass_model_path))
classifier_no_mass.eval()

cls_scores_full = (
    classifier_full(torch.tensor(vals_full, dtype=torch.float32)).detach().cpu().numpy()
)
cls_scores_no_m = (
    classifier_no_mass(torch.tensor(data_feats, dtype=torch.float32))
    .detach()
    .cpu()
    .numpy()
)

scores = [cls_scores_full, cls_scores_no_m, density_ratio, flow_learned_ratio_no_mass]
labels = [
    "Classifier (full info)",
    "Classifier (no mass)",
    "Density ratio (full info)",
    "Density ratio (no mass)",
]

print(
    np.mean(cls_scores_no_m[labels_holdout.reshape(-1) < 0.5]),
    np.mean(cls_scores_no_m[labels_holdout.reshape(-1) > 0.5]),
)
print(
    np.mean(cls_scores_full[labels_holdout.reshape(-1) < 0.5]),
    np.mean(cls_scores_full[labels_holdout.reshape(-1) > 0.5]),
)


from sklearn.metrics import roc_curve

y_test = labels_holdout

sics = []
rocs = []
aucs = []

for i in range(len(scores)):
    print(scores[i].shape)
    eps = 1e-8
    fpr, tpr, _ = roc_curve(y_test, scores[i])
    auc = roc_auc_score(y_test, scores[i])
    sic = tpr / (np.sqrt(fpr) + eps)
    sic = np.clip(sic, None, 10)
    sics.append((fpr, sic))
    rocs.append((tpr, 1.0 / (fpr + eps)))
    aucs.append(auc)


plt.figure()
for i in range(len(aucs)):
    plt.plot(sics[i][0], sics[i][1], label=labels[i] + " auc=%.3f" % aucs[i])

plt.xlabel("False Positive Rate")
plt.ylabel("Significance Improvement")
plt.xscale("log")
plt.legend(loc="upper right")
plt.savefig(plot_dir + "classifier_density_sic_cmp.png")

plt.figure()
for i in range(len(aucs)):
    plt.plot(rocs[i][0], rocs[i][1], label=labels[i] + " auc=%.3f" % aucs[i])

plt.xlabel("True Positive Rate")
plt.ylabel("1./False Positive Rate")
plt.yscale("log")
plt.legend(loc="upper right")
plt.savefig(plot_dir + "classifier_density_roc_cmp.png")
