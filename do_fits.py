from utils import *
from iminuit.cost import poisson_chi2


def ensemble_sample(models, context, scaler=None, num_samples=10):
    samples = []
    for model in models:
        model.to("cpu")
        sample = model.sample(num_samples=num_samples, context=context).detach().numpy()
        sample = sample.reshape(-1, sample.shape[-1])
        samples.append(sample)

    samples = np.squeeze(np.concatenate(samples, axis=0))

    if scaler is not None:
        samples = scaler.inverse_transform(samples)

    return samples


def ensemble_sample_bis(models, context, scaler=None):
    samples = []
    model_labels = np.random.randint(0, len(models), size=len(context))
    for nmodel, model in enumerate(models):
        model.to("cpu")
        sample = (
            model.sample(num_samples=1, context=context[model_labels == nmodel])
            .detach()
            .numpy()
        )
        samples.append(sample)

    samples = np.squeeze(np.concatenate(samples, axis=0))

    if scaler is not None:
        samples = scaler.inverse_transform(samples)

    return samples


working_dir = ""
data_dir = os.path.join(working_dir, "datasets/4feats_smear2/")

data_iter = 1
plot_dir = "plots/fits_june17_jac/iter%i_nojac/" % data_iter
zero_align_profiles = True
SB_flow_mask = True

nSig = 200
sample_size = 45000

background_dir = "bkg_models/4feats_smear2/"
signal_dir = "sig_models/4feats_smear2"
cls_dir = "classifiers/4feats_smear0/"  #'classifiers/4feats_smear15_with_enlargment/'

feat_names = [
    r"$\Delta R_{\gamma\gamma}$",
    r"$\Delta R_{bb}$",
    r"H$_{\gamma\gamma}$ $p_T / M$",
    r"H$_{bb}$ $p_T$",
    "Total Mass",
    r"$\Delta R_{\gamma^1 b^1}$",
]

bkg_scaler = joblib.load(data_dir + "bkg_scaler.pkl")
sig_scaler = joblib.load(data_dir + "sig_scaler.pkl")

do_bkg_shape_syst = False
do_sig_shape_syst = False


feats_percentual_uncertainty = 1e-1 * np.ones(4)
variations = np.array([[0, 0, 0, 1], [0, 0, 0, -1]])
variations_names = ["H_bb_up_ten_percent_unc", "H_bb_down_ten_percent_unc"]
quadratic = True


# flow params
bkg_flow_num_layers = 5
bkg_flow_hidden_size = 32
bkg_flow_num_bins = 24
RQS = True

sig_flow_num_layers = 5
sig_flow_hidden_size = 32
sig_flow_num_bins = 24

# num bootstraps of bkg model
Nbootstraps = 5
# num bootstraps of sig model
Nbootstraps_sim = 1
# num models per ensemble
Nensemble = 5
device = "cpu"


evt_start = data_iter * sample_size
evt_stop = (data_iter + 1) * sample_size

# num bins for plots
nBins = 30
# for plotting histograms
samples_per_model = 1


os.makedirs(plot_dir, exist_ok=True)
data_holdout = np.load(data_dir + "data_holdout.npy")
labels_holdout = np.load(data_dir + "labels_holdout.npy")

# for fitting signal shape
sim = np.load(data_dir + "sim.npy")

nfeatures = data_holdout.shape[1] - 1

mass_low_sr = convert_mgg(115.0)
mass_high_sr = convert_mgg(135.0)

cut_low = convert_mgg(95.0)
cut_high = convert_mgg(180.0)

# Fit signal crystall ball shape
sig_masses = sim[:, 0:1]
sig_masses = sig_masses[:20000]
sig_masses_mask = (sig_masses > mass_low_sr + 0.01) & (sig_masses < mass_high_sr - 0.01)
sig_masses_fit = sig_masses[sig_masses_mask]
c = cost.UnbinnedNLL(sig_masses_fit, pdf_DCB)

plt.figure()
plt.hist(sig_masses_fit, bins=40)
plt.savefig(plot_dir + "sig_masses.pdf")


loc = np.mean(sig_masses_fit)
m_sig_dcb = Minuit(
    c,
    m_left=2.0,
    beta_left=2.0,
    scale_left=0.5,
    beta_right=2.0,
    m_right=2.0,
    scale_right=0.5,
    loc=loc,
)
m_sig_dcb.limits["scale_left"] = (0, None)
m_sig_dcb.limits["scale_right"] = (0, None)
m_sig_dcb.limits["beta_left"] = (1, None)
m_sig_dcb.limits["beta_right"] = (1, None)

m_sig_dcb.migrad()  # finds minimum
m_sig_dcb.hesse()  # accurately computes uncertainties

joblib.dump(m_sig_dcb.values[:], plot_dir + "dcb_fit.pkl")


(
    beta_left_fit,
    m_left_fit,
    scale_left_fit,
    beta_right_fit,
    m_right_fit,
    scale_right_fit,
    loc_fit,
) = m_sig_dcb.values

sig_pdf = lambda x: len(sig_masses_fit) * cb.pdf(
    x,
    beta_left_fit,
    m_left_fit,
    scale_left_fit,
    beta_right_fit,
    m_right_fit,
    scale_right_fit,
    loc_fit,
)

make_postfit_plot(
    sig_masses_fit,
    pdfs=[sig_pdf],
    labels=["Signal"],
    colors=["red"],
    bins=50,
    axis_label="Diphoton mass",
    fname=plot_dir + "dcb_fit.pdf",
)

data_feats = data_holdout[evt_start:evt_stop, 1:]
data_masses = data_holdout[evt_start:evt_stop, 0:1].reshape(-1)
labels_holdout = labels_holdout[evt_start:evt_stop]

template_bins = np.linspace(0.0, 1.0, 50)


data_feats = data_feats[(data_masses > cut_low) & (data_masses < cut_high)]
labels_holdout = labels_holdout[(data_masses > cut_low) & (data_masses < cut_high)]
data_masses = data_masses[(data_masses > cut_low) & (data_masses < cut_high)]


# Filter signal to chosen amount
sig_mask = get_signal_mask(labels_holdout, nSig, seed=42)
data_feats = data_feats[sig_mask]
data_masses = data_masses[sig_mask]
labels_holdout = labels_holdout[sig_mask]

SB_mask = (data_masses < mass_low_sr) | (data_masses >= mass_high_sr)
SR_mask = (data_masses >= mass_low_sr) & (data_masses < mass_high_sr)

sig_eff_SR_mask = np.mean(SR_mask[labels_holdout.reshape(-1) > 0.5])


bkg_flow_prob = np.zeros((Nbootstraps, len(data_feats)))
sig_flow_prob = np.zeros((Nbootstraps_sim, len(data_feats)))

if do_bkg_shape_syst:
    bkg_flow_flucts = np.zeros((Nbootstraps, len(variations), len(data_feats)))
    if not quadratic:
        coefs_bkg_syst = np.zeros(
            (Nbootstraps, int(0.5 * len(variations)), 6, len(data_feats))
        )
    else:
        coefs_bkg_syst = np.zeros(
            (Nbootstraps, int(0.5 * len(variations)), 2, len(data_feats))
        )

if do_sig_shape_syst:
    sig_flow_flucts = np.zeros((Nbootstraps_sim, len(variations), len(data_feats)))
    if not quadratic:
        coefs_sig_syst = np.zeros(
            (Nbootstraps_sim, int(0.5 * len(variations)), 6, len(data_feats))
        )
    else:
        coefs_sig_syst = np.zeros(
            (Nbootstraps_sim, int(0.5 * len(variations)), 2, len(data_feats))
        )

overall_bkg_probs = []

# separate preprocessing for signal and bkg flows
bkg_inputs = bkg_scaler.transform(data_feats)

# compute jacobian of preprocessing transformations
bkg_scaler_jacobian = np.ones((len(data_feats), nfeatures))
for nfeat in range(nfeatures):
    h1 = (
        bkg_scaler["logitscaler"].scale_[nfeat] * data_feats[:, nfeat]
        + bkg_scaler["logitscaler"].min_[nfeat]
    )
    h1 = np.where(
        h1 < 1 - bkg_scaler["logitscaler"].epsilon,
        h1,
        1 - bkg_scaler["logitscaler"].epsilon,
    )
    h1 = np.where(
        h1 > bkg_scaler["logitscaler"].epsilon, h1, bkg_scaler["logitscaler"].epsilon
    )

    bkg_scaler_jacobian[:, nfeat] = (
        (1 / bkg_scaler["standardscaler"].scale_[nfeat])
        * (1 / (h1 * (1 - h1)))
        * bkg_scaler["logitscaler"].scale_[nfeat]
    )

det_bkg_scaler_jacobian = np.abs(np.prod(bkg_scaler_jacobian, 1))
print("Bkg jacobian")
print(np.sum(np.isnan(det_bkg_scaler_jacobian)), np.sum(det_bkg_scaler_jacobian == 0))
print(det_bkg_scaler_jacobian[:10])

sig_inputs = sig_scaler.transform(data_feats)

sig_scaler_jacobian = np.ones((len(data_feats), nfeatures))
for nfeat in range(nfeatures):
    h1 = (
        sig_scaler["logitscaler"].scale_[nfeat] * data_feats[:, nfeat]
        + sig_scaler["logitscaler"].min_[nfeat]
    )
    h1 = np.where(
        h1 < 1 - sig_scaler["logitscaler"].epsilon,
        h1,
        1 - sig_scaler["logitscaler"].epsilon,
    )
    h1 = np.where(
        h1 > sig_scaler["logitscaler"].epsilon, h1, sig_scaler["logitscaler"].epsilon
    )

    sig_scaler_jacobian[:, nfeat] = (
        (1 / sig_scaler["standardscaler"].scale_[nfeat])
        * (1 / (h1 * (1 - h1)))
        * sig_scaler["logitscaler"].scale_[nfeat]
    )

det_sig_scaler_jacobian = np.abs(np.prod(sig_scaler_jacobian, 1))

print("Sig jacobian")
print(np.sum(np.isnan(det_sig_scaler_jacobian)), np.sum(det_sig_scaler_jacobian == 0))
print(det_sig_scaler_jacobian[:10])

data_feats_sig = np.copy(data_feats)
# sideband events will be outside domain of sig flow training, so use central value
data_feats_sig[:, 0][SB_mask] = (mass_low_sr + mass_high_sr) / 2.0


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

# bkg_inputs = torch.tensor(np.nan_to_num(bkg_inputs, max_value), dtype=torch.float32)
# sig_inputs = torch.tensor(np.nan_to_num(sig_inputs, max_value), dtype=torch.float32)

bkg_samples = []

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
    bkg_flow_prob[nboot] = np.exp(bkg_flow_prob[nboot]) * det_bkg_scaler_jacobian
    bkg_samples.append(
        ensemble_sample(
            models,
            context=torch.tensor(
                data_masses[SR_mask].reshape(-1, 1), dtype=torch.float32
            ),
            scaler=bkg_scaler,
            num_samples=samples_per_model,
        )
    )

    if do_bkg_shape_syst:
        for nvariation in range(len(variations)):
            models = []
            for iensemble in range(Nensemble):
                model_path = os.path.join(
                    background_dir,
                    "background_model_boot%i_%i_%s.par"
                    % (nboot, iensemble, variations_names[nvariation]),
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

                bkg_flow_flucts[nboot, nvariation] += (
                    flowB.log_prob(
                        inputs=bkg_inputs,
                        context=torch.tensor(
                            data_masses.reshape(-1, 1), dtype=torch.float32
                        ),
                    )
                    .cpu()
                    .detach()
                    .numpy()
                )
                models.append(flowB)

            bkg_flow_flucts[nboot, nvariation] /= Nensemble
            bkg_flow_flucts[nboot, nvariation] = (
                np.exp(bkg_flow_flucts[nboot, nvariation]) * det_bkg_scaler_jacobian
            )

            if nvariation % 2:
                print(nvariation)
                if not quadratic:
                    coefs_bkg_syst[nboot, int((nvariation - 1) / 2)] = (
                        piecewise_exp_coefficient(
                            bkg_flow_flucts[nboot, nvariation - 1]
                            / bkg_flow_prob[nboot],
                            bkg_flow_flucts[nboot, nvariation] / bkg_flow_prob[nboot],
                        )
                    )
                else:
                    coefs_bkg_syst[nboot, int((nvariation - 1) / 2)] = (
                        piecewise_quadratic_coefficient(
                            bkg_flow_flucts[nboot, nvariation - 1]
                            / bkg_flow_prob[nboot],
                            bkg_flow_flucts[nboot, nvariation] / bkg_flow_prob[nboot],
                        )
                    )


sig_samples = []
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
    sig_flow_prob[nboot] = np.exp(sig_flow_prob[nboot]) * det_sig_scaler_jacobian
    sig_samples.append(
        ensemble_sample(
            models,
            context=torch.tensor(
                data_masses[SR_mask].reshape(-1, 1), dtype=torch.float32
            ),
            scaler=sig_scaler,
            num_samples=samples_per_model,
        )
    )

    if do_sig_shape_syst:
        for nvariation in range(len(variations)):
            models = []
            for iensemble in range(Nensemble):
                model_path = os.path.join(
                    signal_dir,
                    "signal_model_boot%i_%i_%s.par"
                    % (nboot, iensemble, variations_names[nvariation]),
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

                sig_flow_flucts[nboot, nvariation] += (
                    flowS.log_prob(
                        inputs=sig_inputs,
                        context=torch.tensor(
                            data_masses.reshape(-1, 1), dtype=torch.float32
                        ),
                    )
                    .cpu()
                    .detach()
                    .numpy()
                )
                models.append(flowS)

            sig_flow_flucts[nboot, nvariation] /= Nensemble
            sig_flow_flucts[nboot, nvariation] = (
                np.exp(sig_flow_flucts[nboot, nvariation]) * det_sig_scaler_jacobian
            )

            if nvariation % 2:
                print(nvariation)
                if not quadratic:
                    coefs_sig_syst[nboot, int((nvariation - 1) / 2)] = (
                        piecewise_exp_coefficient(
                            sig_flow_flucts[nboot, nvariation - 1]
                            / sig_flow_prob[nboot],
                            sig_flow_flucts[nboot, nvariation] / sig_flow_prob[nboot],
                        )
                    )
                else:
                    coefs_sig_syst[nboot, int((nvariation - 1) / 2)] = (
                        piecewise_quadratic_coefficient(
                            sig_flow_flucts[nboot, nvariation - 1]
                            / sig_flow_prob[nboot],
                            sig_flow_flucts[nboot, nvariation] / sig_flow_prob[nboot],
                        )
                    )

# bkg_flow_prob_all = copy.deepcopy(bkg_flow_prob)
# sig_flow_prob_all = copy.deepcopy(sig_flow_prob)
likelihood_ratios = [sig_flow_prob[0] / bkg_flow_prob[i] for i in range(Nbootstraps)]
s_over_b_ratios = [
    sig_flow_prob[0] / (sig_flow_prob[0] + bkg_flow_prob[i]) for i in range(Nbootstraps)
]

# ignore feature densities outside SR, where there is no relevant info
if SB_flow_mask:
    for i in range(len(bkg_flow_prob)):
        bkg_flow_prob[i][SB_mask] = 1.0 / (2 * cutoff) ** 4
        if do_bkg_shape_syst:
            bkg_flow_flucts[i, :, SB_mask] = 1.0
            coefs_bkg_syst[i, :, :, SB_mask] = 0.0

    for i in range(len(sig_flow_prob)):
        sig_flow_prob[i][SB_mask] = 1.0 / (2 * cutoff) ** 4
        if do_sig_shape_syst:
            sig_flow_flucts[i, :, SB_mask] = 1.0
            coefs_sig_syst[i, :, :, SB_mask] = 0.0


print(
    np.min(bkg_flow_prob),
    np.min(sig_flow_prob),
    np.max(bkg_flow_prob),
    np.max(sig_flow_prob),
)

is_sig = labels_holdout[SR_mask] > 0.5
for i in range(Nbootstraps):
    LR = likelihood_ratios[i][SR_mask]
    plt.figure()
    bins = np.linspace(0, 10, 100)
    plt.hist(
        [LR[~is_sig], LR[is_sig]],
        bins=bins,
        color=["blue", "red"],
        label=["Bkg", "Signal"],
        histtype="step",
        density=True,
    )
    plt.xlabel("Likelihood Ratio")
    plt.savefig(plot_dir + "model%i_likelihood_ratios.pdf" % i)


vals_full = np.concatenate((data_masses.reshape(-1, 1), data_feats), axis=1)

# load classifiers
cls_full_model_path = cls_dir + "model_full.par"
cls_no_mass_model_path = cls_dir + "model_no_mass.par"

cls_layers = [16, 32, 32, 8]
dropout = 0.2

classifier_full = DNN(
    input_dim=vals_full.shape[1], layers=cls_layers, dropout_probability=dropout
)
classifier_full.load_state_dict(torch.load(cls_full_model_path))
classifier_full.eval()

classifier_no_mass = DNN(
    input_dim=data_feats.shape[1], layers=cls_layers, dropout_probability=dropout
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

cls_scores_sig = [
    classifier_no_mass(torch.tensor(samples, dtype=torch.float32))
    .detach()
    .cpu()
    .numpy()
    .reshape(-1)
    for samples in sig_samples
]
cls_scores_bkg = [
    classifier_no_mass(torch.tensor(samples, dtype=torch.float32))
    .detach()
    .cpu()
    .numpy()
    .reshape(-1)
    for samples in bkg_samples
]


cls_cut_plot = 0.9
data_cls_mask = (cls_scores_no_m > cls_cut_plot).reshape(-1)
sig_cls_mask = [scores > cls_cut_plot for scores in cls_scores_sig]
bkg_cls_mask = [scores > cls_cut_plot for scores in cls_scores_bkg]


# cut for mass-indep classifier
cls_percentile = 95
cut_val = np.percentile(cls_scores_no_m[SR_mask], cls_percentile)
cls_cut = (cls_scores_no_m > cut_val).reshape(-1)
cls_cut_sig_eff = np.mean(cls_cut[labels_holdout.reshape(-1) > 0.5])


# Do template fit of binned DNN score
temps = joblib.load(cls_dir + "templates.pkl")
sig_temp, bkg_temp = temps["sig"], temps["bkg"]


DNN_bins = temps["bins"]
data_temp, _ = np.histogram(cls_scores_full[SR_mask], bins=DNN_bins)


def temp_cost_func(bkg_norm, sig_norm):
    exp = bkg_norm * bkg_temp + sig_norm * sig_temp
    return poisson_chi2(data_temp, exp)


# plt.figure()
# plt.stairs(data_temp, bins, label = 'data')
# plt.stairs(sig_temp*true_s, bins, label = 'sig')
# plt.stairs(bkg_temp*true_b, bins, label = 'bkg')
# plt.savefig(plot_dir + "DNN_templates.pdf")


fit_fn = ExpShape(np.min(data_masses), np.max(data_masses), order=3, num_integral=True)
p0_guess = len(data_masses)
p1_guess = 1.2
p2_guess = 0.1
p0_range = (0, None)
p1_range = (-100, 100)
p2_range = (-100, 100)

if do_bkg_shape_syst or do_sig_shape_syst:
    alpha_guess = 0
    alpha_range = (-5, 5)

print("Fitting %i events" % (len(data_masses)))
print("Fitting %i events" % (np.sum(labels_holdout)))


# define different likelihood functions for various fits


def pdf_comb_only_m(x, s, p0, p1, p2):
    bkg_shape_norm = fit_fn.norm([p0, p1, p2])
    norm = s + bkg_shape_norm
    bkg_mass_prob = fit_fn.density(x, [p0, p1, p2])
    sig_mass_prob = cb.pdf(
        x,
        beta_left_fit,
        m_left_fit,
        scale_left_fit,
        beta_right_fit,
        m_right_fit,
        scale_right_fit,
        loc_fit,
    )

    L = bkg_mass_prob + s * sig_mass_prob
    return norm, L


def pdf_comb_one_bkg_choose(index):
    def internal_function(x, s, p0, p1, p2):
        bkg_shape_norm = fit_fn.norm([p0, p1, p2])
        norm = s + bkg_shape_norm
        bkg_mass_prob = fit_fn.density(x, [p0, p1, p2])
        sig_mass_prob = cb.pdf(
            x,
            beta_left_fit,
            m_left_fit,
            scale_left_fit,
            beta_right_fit,
            m_right_fit,
            scale_right_fit,
            loc_fit,
        )

        L = (
            bkg_mass_prob * (bkg_flow_prob[index])
            + s * sig_mass_prob * sig_flow_prob[0]
        )
        return norm, L

    return internal_function


def pdf_comb_one_bkg_one_sig_choose(index_bkg, index_sig):
    def internal_function(x, s, p0, p1, p2):
        bkg_shape_norm = fit_fn.norm([p0, p1, p2])
        norm = s + bkg_shape_norm
        bkg_mass_prob = fit_fn.density(x, [p0, p1, p2])
        sig_mass_prob = cb.pdf(
            x,
            beta_left_fit,
            m_left_fit,
            scale_left_fit,
            beta_right_fit,
            m_right_fit,
            scale_right_fit,
            loc_fit,
        )

        L = (
            bkg_mass_prob * (bkg_flow_prob[index_bkg])
            + s * sig_mass_prob * sig_flow_prob[index_sig]
        )
        return norm, L

    return internal_function


def pdf_comb_one_bkg_one_sig_choose_sig_shape_syst(index_bkg, index_sig):
    def internal_function(x, s, p0, p1, p2, alpha):
        bkg_shape_norm = fit_fn.norm([p0, p1, p2])
        bkg_mass_prob = fit_fn.density(x, [p0, p1, p2])
        sig_mass_prob = cb.pdf(
            x,
            beta_left_fit,
            m_left_fit,
            scale_left_fit,
            beta_right_fit,
            m_right_fit,
            scale_right_fit,
            loc_fit,
        )
        if not quadratic:
            alpha_fluctuations = piecewise_exp(
                alpha,
                sig_flow_flucts[index_sig, 0, :] / sig_flow_prob[index_sig, :],
                sig_flow_flucts[index_sig, 1, :] / sig_flow_prob[index_sig, :],
                coefs_sig_syst[index_sig, 0],
            )
        else:
            alpha_fluctuations = piecewise_quadratic(
                alpha,
                sig_flow_flucts[index_sig, 0, :] / sig_flow_prob[index_sig, :],
                sig_flow_flucts[index_sig, 1, :] / sig_flow_prob[index_sig, :],
                coefs_sig_syst[index_sig, 0],
            )

        L = (
            bkg_mass_prob * (bkg_flow_prob[index_bkg])
            + s * sig_mass_prob * sig_flow_prob[index_sig] * alpha_fluctuations
        )
        return s + bkg_shape_norm, L

    return internal_function


def pdf_comb_one_bkg_one_sig_choose_bkg_shape_syst(index_bkg, index_sig):
    def internal_function(x, s, p0, p1, p2, alpha):
        bkg_shape_norm = fit_fn.norm([p0, p1, p2])
        bkg_mass_prob = fit_fn.density(x, [p0, p1, p2])
        sig_mass_prob = cb.pdf(
            x,
            beta_left_fit,
            m_left_fit,
            scale_left_fit,
            beta_right_fit,
            m_right_fit,
            scale_right_fit,
            loc_fit,
        )
        if not quadratic:
            alpha_fluctuations = piecewise_exp(
                alpha,
                bkg_flow_flucts[index_bkg, 0, :] / bkg_flow_prob[index_bkg, :],
                bkg_flow_flucts[index_bkg, 1, :] / bkg_flow_prob[index_bkg, :],
                coefs_bkg_syst[index_bkg, 0],
            )
        else:
            alpha_fluctuations = piecewise_quadratic(
                alpha,
                bkg_flow_flucts[index_bkg, 0, :] / bkg_flow_prob[index_bkg, :],
                bkg_flow_flucts[index_bkg, 1, :] / bkg_flow_prob[index_bkg, :],
                coefs_bkg_syst[index_bkg, 0],
            )

        L = (
            bkg_mass_prob * (bkg_flow_prob[index_bkg]) * alpha_fluctuations
            + s * sig_mass_prob * sig_flow_prob[index_sig]
        )
        return s + bkg_shape_norm, L

    return internal_function


def pdf_comb_one_bkg_one_sig_choose_both_shape_syst(index_bkg, index_sig):
    def internal_function(x, s, p0, p1, p2, alpha):
        bkg_shape_norm = fit_fn.norm([p0, p1, p2])
        bkg_mass_prob = fit_fn.density(x, [p0, p1, p2])
        sig_mass_prob = cb.pdf(
            x,
            beta_left_fit,
            m_left_fit,
            scale_left_fit,
            beta_right_fit,
            m_right_fit,
            scale_right_fit,
            loc_fit,
        )
        if not quadratic:
            alpha_fluctuations_bkg = piecewise_exp(
                alpha,
                bkg_flow_flucts[index_bkg, 0, :] / bkg_flow_prob[index_bkg, :],
                bkg_flow_flucts[index_bkg, 1, :] / bkg_flow_prob[index_bkg, :],
                coefs_bkg_syst[index_bkg, 0],
            )
            alpha_fluctuations_sig = piecewise_exp(
                alpha,
                sig_flow_flucts[index_sig, 0, :] / sig_flow_prob[index_sig, :],
                sig_flow_flucts[index_sig, 1, :] / sig_flow_prob[index_sig, :],
                coefs_sig_syst[index_sig, 0],
            )
        else:
            alpha_fluctuations_bkg = piecewise_quadratic(
                alpha,
                bkg_flow_flucts[index_bkg, 0, :] / bkg_flow_prob[index_bkg, :],
                bkg_flow_flucts[index_bkg, 1, :] / bkg_flow_prob[index_bkg, :],
                coefs_bkg_syst[index_bkg, 0],
            )
            alpha_fluctuations_sig = piecewise_quadratic(
                alpha,
                sig_flow_flucts[index_sig, 0, :] / sig_flow_prob[index_sig, :],
                sig_flow_flucts[index_sig, 1, :] / sig_flow_prob[index_sig, :],
                coefs_sig_syst[index_sig, 0],
            )

        L = (
            bkg_mass_prob * (bkg_flow_prob[index_bkg]) * alpha_fluctuations_bkg
            + s * sig_mass_prob * sig_flow_prob[index_sig] * alpha_fluctuations_sig
        )
        return s + bkg_shape_norm, L

    return internal_function


# plt.figure()
# plt.hist(data_masses, bins = 40)
# plt.savefig("masses.pdf")
# plt.figure()

# Fit only mass for a check
c_mass = cost.ExtendedUnbinnedNLL(data_masses, pdf_comb_only_m)

m_sig_bkg_mass = Minuit(c_mass, s=0, p0=p0_guess, p1=p1_guess, p2=p2_guess)
m_sig_bkg_mass.limits["p0"] = p0_range
m_sig_bkg_mass.limits["p1"] = p1_range
m_sig_bkg_mass.limits["p2"] = p2_range
m_sig_bkg_mass.fixed["s"] = False


m_sig_bkg_mass.migrad()  # finds minimum
m_sig_bkg_mass.hesse()  # accurately computes uncertainties
# plt.figure()
# m_plot = m_sig_bkg_mass.visualize()
# plt.savefig(plot_dir + "mass_fit.pdf")


mass_only_s = m_sig_bkg_mass.values[0]
mass_only_s_err = m_sig_bkg_mass.errors[0]
best_p0 = m_sig_bkg_mass.values[1]
best_p1 = m_sig_bkg_mass.values[2]
best_p2 = m_sig_bkg_mass.values[3]
bkg_params = m_sig_bkg_mass.values[1:]


print(
    "\n\n Mass only fit: signal strength is %.2f +/- %.2f "
    % (mass_only_s, mass_only_s_err)
)
print(m_sig_bkg_mass.values)

p0_guess = best_p0
p1_guess = best_p1
p2_guess = best_p2


def draw_mass_postfit(masses, fit_params, fname=""):
    mgg_high = 180
    mgg_low = 90
    bkg_params = fit_params[1:]
    sig_pdf = (
        lambda x: fit_params[0]
        * cb.pdf(
            convert_mgg(x),
            beta_left_fit,
            m_left_fit,
            scale_left_fit,
            beta_right_fit,
            m_right_fit,
            scale_right_fit,
            loc_fit,
        )
        / (mgg_high - mgg_low)
    )
    bkg_pdf = lambda x: fit_fn.density(convert_mgg(x), bkg_params) / (
        mgg_high - mgg_low
    )

    make_postfit_plot(
        reverse_mgg(masses),
        pdfs=[bkg_pdf, sig_pdf],
        labels=["Bkg", "Sig"],
        colors=["blue", "red"],
        bins=nBins,
        bottom_panel="subtract",
        axis_label="Diphoton mass",
        fname=fname,
    )


draw_mass_postfit(
    data_masses, m_sig_bkg_mass.values, fname=plot_dir + "mass_only_fit.pdf"
)


# sig only masses

if nSig > 0:
    sig_pdf = lambda x: mass_only_s * cb.pdf(
        x,
        beta_left_fit,
        m_left_fit,
        scale_left_fit,
        beta_right_fit,
        m_right_fit,
        scale_right_fit,
        loc_fit,
    )
    make_postfit_plot(
        data_masses[labels_holdout > 0.5],
        pdfs=[sig_pdf],
        labels=["Signal"],
        colors=["red"],
        bins=50,
        axis_label="Diphoton mass (signal events)",
        fname=plot_dir + "signal_mass_fit.pdf",
    )


# #### Now let's profile over the background

# 3 cases:
# - Only mass
# - One background model for $x$
# - Profiling


def analyser(
    func, masses=None, verbose=True, grid=None, s_guess=0, syst=False, plt_name=""
):

    norm = len(masses)

    if syst:
        c_comb = cost.ExtendedUnbinnedNLL(masses, func) + cost.NormalConstraint(
            ["alpha"], [0], [1]
        )
        m = Minuit(
            c_comb, s=s_guess, p0=norm, p1=p1_guess, p2=p2_guess, alpha=alpha_guess
        )
        m.limits["alpha"] = alpha_range
    else:
        c_comb = cost.ExtendedUnbinnedNLL(masses, func)
        m = Minuit(c_comb, s=s_guess, p0=norm, p1=p1_guess, p2=p2_guess)

    m.limits["p0"] = p0_range
    m.limits["p1"] = p1_range
    m.limits["p2"] = p2_range
    m.fixed["s"] = False
    # m.fixed['alpha']=True

    m.migrad()  # finds minimum
    m.hesse()

    if grid is None:
        num_grid = 60
        grid = np.linspace(
            m.values["s"] - 3 * m.errors["s"],
            m.values["s"] + 3 * m.errors["s"],
            num_grid,
        )
        s_values, s_profile_likelihood, s_convergence = m.mnprofile("s", grid=grid)
    else:
        s_values, s_profile_likelihood, s_convergence = m.mnprofile("s", grid=grid)

    eps = 1e-8
    best_s = m.values["s"]
    best_s_err = m.errors["s"] + eps

    L_s_free = func(masses, *m.values)[1]
    NLL_s_free = -np.sum(np.log(L_s_free))
    # print(NLL_s_free)

    if verbose:
        print(
            "best fit signal strength is %.2f +/- %.2f. Naive signif %.2f"
            % (best_s, best_s_err, best_s / best_s_err)
        )
        print("NLL %.2f" % NLL_s_free)

    if syst:
        c_comb = cost.ExtendedUnbinnedNLL(masses, func) + cost.NormalConstraint(
            ["alpha"], [0], [1]
        )
        m_fixed_s = Minuit(
            c_comb, s=0, p0=norm, p1=p1_guess, p2=p2_guess, alpha=alpha_guess
        )
        m_fixed_s.limits["alpha"] = alpha_range
    else:
        c_comb = cost.ExtendedUnbinnedNLL(masses, func)
        m_fixed_s = Minuit(c_comb, s=0, p0=norm, p1=p1_guess, p2=p2_guess)

    m_fixed_s.limits["p0"] = p0_range
    m_fixed_s.limits["p1"] = p1_range
    m_fixed_s.limits["p2"] = p2_range
    m_fixed_s.fixed["s"] = True
    # m_fixed_s.fixed['alpha']=True

    if len(plt_name) > 0:
        print("drawing fit %s" % plt_name)
        draw_mass_postfit(masses, m.values, fname=plt_name)

    m_fixed_s.migrad()  # finds minimum
    m_fixed_s.hesse()  # accurately computes uncertainties

    L_s_fixed = func(masses, *m_fixed_s.values)[1]
    NLL_s_fixed = -np.sum(np.log(L_s_fixed))

    delta_NLL = 2.0 * (NLL_s_fixed - NLL_s_free)
    # avoid errors
    delta_NLL = max(0.0, delta_NLL)

    if verbose:
        print(
            "Delta NLL is %.1f. Approx significance is %.2f"
            % (delta_NLL, delta_NLL**0.5)
        )
        print(m.values)

    out = {
        "s_values": s_values,
        "s_profile": s_profile_likelihood,
        "best_s": best_s,
        "best_s_err": best_s_err,
        "NLL_s_free": NLL_s_free,
        "NLL_s_fixed": NLL_s_fixed,
        "delta_NLL": delta_NLL,
        "params": m.values[:],
    }

    return out


results_only_m = analyser(pdf_comb_only_m, masses=data_masses)
joblib.dump(results_only_m, plot_dir + "m_only_fit.pkl")


results_cls_m = analyser(
    pdf_comb_only_m, masses=data_masses[cls_cut], plt_name=plot_dir + "cls_mass_fit.pdf"
)
joblib.dump(results_cls_m, plot_dir + "cls_m_fit.pkl")


print("\n\n DNN template fit")
temp_cost_func.errordef = Minuit.LEAST_SQUARES
temp_cost_func.ndata = np.sum(data_temp)
m_temp = Minuit(temp_cost_func, bkg_norm=np.sum(data_temp), sig_norm=0)
m_temp.migrad()
m_temp.hesse()


DNN_best_s = m_temp.values["sig_norm"]
num_grid = 60
grid = np.linspace(
    m_temp.values["sig_norm"] - 3 * m_temp.errors["sig_norm"],
    m_temp.values["sig_norm"] + 3 * m_temp.errors["sig_norm"],
    num_grid,
)

DNN_s_values, DNN_s_profile, DNN_s_convergence = m_temp.mnprofile("sig_norm")
# rescale based on eff of SR mask
DNN_s_values /= sig_eff_SR_mask

print(m_temp.values)

eps = 1e-8
best_sig = m_temp.values["sig_norm"]
best_bkg = m_temp.values["bkg_norm"]
best_sig_err = m_temp.errors["sig_norm"] + eps
print(
    "best fit signal strength is %.2f +/- %.2f. Naive signif %.2f"
    % (best_sig, best_sig_err, best_sig / best_sig_err)
)

m_temp_frozen = Minuit(temp_cost_func, bkg_norm=np.sum(data_temp), sig_norm=0)
m_temp_frozen.fixed["sig_norm"] = True
m_temp_frozen.migrad()
m_temp_frozen.hesse()

NLL_DNN = temp_cost_func(best_bkg, best_sig)
NLL_DNN_frozen = temp_cost_func(m_temp_frozen.values[0], 0)
DLL_DNN = NLL_DNN_frozen - NLL_DNN
print("DNN signifi %.2f" % DLL_DNN**0.5)


results_DNN = {
    "s_values": DNN_s_values,
    "s_profile": DNN_s_profile,
    "best_s": DNN_best_s,
    "best_s_err": m_temp.errors["sig_norm"],
    "NLL_s_free": NLL_DNN,
    "NLL_s_fixed": NLL_DNN_frozen,
    "delta_NLL": DLL_DNN,
    "params": m_temp.values[:],
}

joblib.dump(results_DNN, plot_dir + "DNN_fit.pkl")

make_postfit_plot(
    cls_scores_full[SR_mask],
    templates=[bkg_temp * best_bkg, sig_temp * best_sig],
    do_norm=False,
    labels=["Bkg", "Signal"],
    colors=["blue", "red"],
    bins=DNN_bins,
    axis_label="DNN Score",
    bottom_panel="subtract",
    fname=plot_dir + "DNN_fit.pdf",
)


print("\n First model")
results_one_bkg = analyser(
    pdf_comb_one_bkg_one_sig_choose(0, 0),
    masses=data_masses,
    s_guess=results_only_m["best_s"],
    plt_name=plot_dir + "model0_sig0_mass_fit.pdf",
)

if do_sig_shape_syst and (not do_bkg_shape_syst):
    print("\n First model + signal systematics")
    results_one_bkg_with_syst = analyser(
        pdf_comb_one_bkg_one_sig_choose_sig_shape_syst(0, 0),
        masses=data_masses,
        s_guess=results_only_m["best_s"],
        syst=True,
        plt_name=plot_dir + "model0_sig0_syst_mass_fit.png",
    )
    joblib.dump(results_one_bkg_with_syst, plot_dir + "model0_sig0_sig_syst_fit.pkl")

if (not do_sig_shape_syst) and do_bkg_shape_syst:
    print("\n First model + bkg systematics")
    results_one_bkg_with_syst = analyser(
        pdf_comb_one_bkg_one_sig_choose_bkg_shape_syst(0, 0),
        masses=data_masses,
        s_guess=results_only_m["best_s"],
        syst=True,
        plt_name=plot_dir + "model0_sig0_syst_mass_fit.png",
    )
    joblib.dump(results_one_bkg_with_syst, plot_dir + "model0_sig0_bkg_syst_fit.pkl")

if do_sig_shape_syst and do_bkg_shape_syst:
    print("\n First model + sig and bkg systematics")
    results_one_bkg_with_syst = analyser(
        pdf_comb_one_bkg_one_sig_choose_both_shape_syst(0, 0),
        masses=data_masses,
        s_guess=results_only_m["best_s"],
        syst=True,
        plt_name=plot_dir + "model0_sig0_syst_mass_fit.png",
    )
    joblib.dump(results_one_bkg_with_syst, plot_dir + "model0_sig0_both_syst_fit.pkl")


# 30 points, +/- 4 sigma
num_grid = 60
mu_grid = np.linspace(
    results_one_bkg["best_s"] - 3 * results_one_bkg["best_s_err"],
    results_one_bkg["best_s"] + 3 * results_one_bkg["best_s_err"],
    num_grid,
)

if do_sig_shape_syst or do_bkg_shape_syst:
    mu_grid_with_syst = np.linspace(
        results_one_bkg_with_syst["best_s"]
        - 3 * results_one_bkg_with_syst["best_s_err"],
        results_one_bkg_with_syst["best_s"]
        + 3 * results_one_bkg_with_syst["best_s_err"],
        num_grid,
    )

# fit all the bootstraps
print("\n Bootstraps")
explicit_bootstraps = [
    [
        analyser(
            pdf_comb_one_bkg_one_sig_choose(index_bkg, index_sig),
            masses=data_masses,
            s_guess=results_only_m["best_s"],
            grid=mu_grid,
        )
        for index_sig in range(Nbootstraps_sim)
    ]
    for index_bkg in range(Nbootstraps)
]


for i in range(Nbootstraps):
    for j in range(Nbootstraps_sim):
        joblib.dump(
            explicit_bootstraps[i][j], plot_dir + "model%i_sig%i_fit.pkl" % (i, j)
        )


if do_sig_shape_syst and (not do_bkg_shape_syst):
    explicit_bootstraps_with_syst = [
        [
            analyser(
                pdf_comb_one_bkg_one_sig_choose_sig_shape_syst(index_bkg, index_sig),
                masses=data_masses,
                syst=True,
                s_guess=results_only_m["best_s"],
                grid=mu_grid_with_syst,
            )
            for index_sig in range(Nbootstraps_sim)
        ]
        for index_bkg in range(Nbootstraps)
    ]
if (not do_sig_shape_syst) and do_bkg_shape_syst:
    explicit_bootstraps_with_syst = [
        [
            analyser(
                pdf_comb_one_bkg_one_sig_choose_bkg_shape_syst(index_bkg, index_sig),
                masses=data_masses,
                syst=True,
                s_guess=results_only_m["best_s"],
                grid=mu_grid_with_syst,
            )
            for index_sig in range(Nbootstraps_sim)
        ]
        for index_bkg in range(Nbootstraps)
    ]
if do_sig_shape_syst and do_bkg_shape_syst:
    explicit_bootstraps_with_syst = [
        [
            analyser(
                pdf_comb_one_bkg_one_sig_choose_both_shape_syst(index_bkg, index_sig),
                masses=data_masses,
                syst=True,
                s_guess=results_only_m["best_s"],
                grid=mu_grid_with_syst,
            )
            for index_sig in range(Nbootstraps_sim)
        ]
        for index_bkg in range(Nbootstraps)
    ]

# remove nans
for i in range(data_feats.shape[1]):
    data_feats[:, i] = clean_nans(data_feats[:, i])

for i in range(nfeatures):
    sig_only = data_feats[:, i][labels_holdout > 0.5]
    if len(sig_only) > 0:
        for j in range(Nbootstraps_sim):
            make_postfit_plot(
                sig_only,
                template_samples=[sig_samples[j][:, i]],
                template_norms=[sig_only.shape[0]],
                labels=["Signal"],
                colors=["red"],
                bins=nBins,
                axis_label=feat_names[i] + " (signal only)",
                ratio_range=(0, 2),
                fname=plot_dir + "sig%i_feat%i.pdf" % (j, i),
            )

    for j in range(Nbootstraps):
        for j2 in range(Nbootstraps_sim):
            bkg_norm = fit_fn.norm(
                explicit_bootstraps[j][j2]["params"][1:],
                start=mass_low_sr,
                stop=mass_high_sr,
            )
            sig_norm = explicit_bootstraps[j][j2]["best_s"]  # assume all in SR
            feat_min = np.percentile(data_feats[SR_mask, i], 0.2)
            feat_max = np.percentile(data_feats[SR_mask, i], 99.8)
            binning = np.linspace(feat_min, feat_max, nBins)

            make_postfit_plot(
                data_feats[SR_mask, i],
                template_samples=[bkg_samples[j][:, i], sig_samples[j2][:, i]],
                template_norms=[bkg_norm, sig_norm],
                bottom_panel="subtract",
                draw_chi2=False,
                labels=["Bkg", "Signal"],
                colors=["blue", "red"],
                bins=binning,
                axis_label=feat_names[i],
                ratio_range=(0, 2),
                fname=plot_dir + "model%i_sig%i_feat%i_fit.pdf" % (j, j2, i),
            )

            # plot in signal enhanced region
            sig_cut = sig_cls_mask[j2]
            bkg_cut = bkg_cls_mask[j]
            sig_eff = np.mean(sig_cut)
            bkg_eff = np.mean(bkg_cut)
            make_postfit_plot(
                data_feats[SR_mask & data_cls_mask, i],
                template_samples=[
                    bkg_samples[j][bkg_cut, i],
                    sig_samples[j2][sig_cut, i],
                ],
                template_norms=[bkg_norm * bkg_eff, sig_norm * sig_eff],
                bottom_panel="subtract",
                draw_chi2=False,
                labels=["Bkg", "Signal"],
                colors=["blue", "red"],
                bins=binning,
                axis_label=feat_names[i],
                ratio_range=(0, 2),
                fname=plot_dir + "model%i_sig%i_feat%i_enriched_fit.pdf" % (j, j2, i),
            )

            if do_sig_shape_syst or do_bkg_shape_syst:
                bkg_norm = fit_fn.norm(
                    explicit_bootstraps_with_syst[j][j2]["params"][1:],
                    start=mass_low_sr,
                    stop=mass_high_sr,
                )
                sig_norm = explicit_bootstraps_with_syst[j][j2][
                    "best_s"
                ]  # assume all in SR
                make_postfit_plot(
                    data_feats[SR_mask, i],
                    template_samples=[bkg_samples[j][:, i], sig_samples[j2][:, i]],
                    template_norms=[bkg_norm, sig_norm],
                    bottom_panel="subtract",
                    labels=["Bkg", "Signal"],
                    colors=["blue", "red"],
                    bins=binning,
                    axis_label=feat_names[i],
                    ratio_range=(0, 2),
                    fname=plot_dir
                    + "model%i_sig%i_feat%i_fit_with_syst.pdf" % (j, j2, i),
                )


NLL_explicit_profile = np.zeros(len(mu_grid))
significance_profile = 0

all_profiles = np.array(
    [
        [
            explicit_bootstraps[i_bkg][i_sig]["s_profile"]
            for i_sig in range(Nbootstraps_sim)
        ]
        for i_bkg in range(Nbootstraps)
    ]
)

if do_sig_shape_syst or do_bkg_shape_syst:
    NLL_explicit_profile_with_syst = np.zeros(len(mu_grid_with_syst))
    significance_profile_with_syst = 0

    all_profiles_with_syst = np.array(
        [
            [
                explicit_bootstraps_with_syst[i_bkg][i_sig]["s_profile"]
                for i_sig in range(Nbootstraps_sim)
            ]
            for i_bkg in range(Nbootstraps)
        ]
    )


# sometimes fits fail, giving nan likleihood, replace with dummy value
dummy = 99999.0
for i_bkg in range(Nbootstraps):
    for i_sig in range(Nbootstraps_sim):
        all_profiles[i_bkg][i_sig][np.isnan(all_profiles[i_bkg][i_sig])] = dummy
        if do_sig_shape_syst or do_bkg_shape_syst:
            all_profiles_with_syst[i_bkg][i_sig][
                np.isnan(all_profiles_with_syst[i_bkg][i_sig])
            ] = dummy

# Rescale bootstraps to have same likelihood min
if zero_align_profiles:
    for i_bkg in range(Nbootstraps):
        for i_sig in range(Nbootstraps_sim):
            all_profiles[i_bkg][i_sig] -= np.min(all_profiles[i_bkg][i_sig])
            if do_sig_shape_syst or do_bkg_shape_syst:
                all_profiles_with_syst[i_bkg][i_sig] -= np.min(
                    all_profiles_with_syst[i_bkg][i_sig]
                )


NLL_explicit_profile = np.min(all_profiles, (0, 1))


all_likelihoods_free = np.array(
    [
        [
            explicit_bootstraps[i_bkg][i_sig]["NLL_s_free"]
            for i_sig in range(Nbootstraps_sim)
        ]
        for i_bkg in range(Nbootstraps)
    ]
)
all_likelihoods_fixed = np.array(
    [
        [
            explicit_bootstraps[i_bkg][i_sig]["NLL_s_fixed"]
            for i_sig in range(Nbootstraps_sim)
        ]
        for i_bkg in range(Nbootstraps)
    ]
)

print("Likelihood scales")
print(all_likelihoods_free - np.min(all_likelihoods_free))
print(all_likelihoods_fixed - np.min(all_likelihoods_fixed))

deltaNLL_explicit = 2.0 * (np.min(all_likelihoods_fixed) - np.min(all_likelihoods_free))
# deltaNLL_explicit= np.min(all_likelihoods_fixed - all_likelihoods_free)
print(all_likelihoods_fixed)
print(all_likelihoods_free)
print(deltaNLL_explicit)
significance_profile = deltaNLL_explicit**0.5

explicit_profile_curve = 2 * (NLL_explicit_profile - np.min(NLL_explicit_profile))

# ymax = np.amax(explicit_profile_curve[ explicit_profile_curve < 30])
ymax = 20.0

if do_sig_shape_syst or do_bkg_shape_syst:

    NLL_explicit_profile_with_syst = np.min(all_profiles_with_syst, (0, 1))

    all_likelihoods_free_with_syst = np.array(
        [
            [
                explicit_bootstraps_with_syst[i_bkg][i_sig]["NLL_s_free"]
                for i_sig in range(Nbootstraps_sim)
            ]
            for i_bkg in range(Nbootstraps)
        ]
    )
    all_likelihoods_fixed_with_syst = np.array(
        [
            [
                explicit_bootstraps_with_syst[i_bkg][i_sig]["NLL_s_fixed"]
                for i_sig in range(Nbootstraps_sim)
            ]
            for i_bkg in range(Nbootstraps)
        ]
    )

    print("Likelihood scales with syst")
    print(all_likelihoods_free_with_syst - np.min(all_likelihoods_free_with_syst))
    print(all_likelihoods_fixed_with_syst - np.min(all_likelihoods_fixed_with_syst))

    deltaNLL_explicit_with_syst = 2.0 * (
        np.min(all_likelihoods_fixed_with_syst) - np.min(all_likelihoods_free_with_syst)
    )
    # deltaNLL_explicit= np.min(all_likelihoods_fixed - all_likelihoods_free)
    print(all_likelihoods_fixed_with_syst)
    print(all_likelihoods_free_with_syst)
    print(deltaNLL_explicit_with_syst)
    significance_profile_with_syst = deltaNLL_explicit_with_syst**0.5

    explicit_profile_curve_with_syst = 2 * (
        NLL_explicit_profile_with_syst - np.min(NLL_explicit_profile_with_syst)
    )

    # ymax_with_syst = np.amax(explicit_profile_curve_with_syst[ explicit_profile_curve_with_syst < 30])
    y_max_with_syst = 20.0

true_nsig = np.sum(labels_holdout)

sigma_only_m = one_sig_CI(
    results_only_m["best_s"] / true_nsig,
    results_only_m["s_values"] / true_nsig,
    results_only_m["s_profile"],
)

# divide by cut eff for apples to apples comparison
rescaled_cls_s_values = results_cls_m["s_values"] / (cls_cut_sig_eff * true_nsig)
sigma_cls_mass = one_sig_CI(
    results_cls_m["best_s"] / (cls_cut_sig_eff * true_nsig),
    rescaled_cls_s_values,
    results_cls_m["s_profile"],
)

sigma_DNN = one_sig_CI(DNN_best_s / true_nsig, DNN_s_values / true_nsig, DNN_s_profile)

eps = 1e-4
# Take mean of the different models as the best fit
best_s_explicit_profile = np.mean(mu_grid[NLL_explicit_profile < eps]) / true_nsig
sigma_explicit_profile = one_sig_CI(
    best_s_explicit_profile, mu_grid / true_nsig, NLL_explicit_profile
)

print("Model best fits", mu_grid[NLL_explicit_profile < eps] / true_nsig)

if do_sig_shape_syst or do_bkg_shape_syst:
    best_s_explicit_profile_with_syst = (
        np.mean(mu_grid_with_syst[NLL_explicit_profile_with_syst < eps]) / true_nsig
    )
    sigma_explicit_profile_with_syst = one_sig_CI(
        best_s_explicit_profile_with_syst,
        mu_grid_with_syst / true_nsig,
        NLL_explicit_profile_with_syst,
    )

print("sigma_only_m", sigma_only_m, results_only_m["best_s_err"])

fig = plt.figure(figsize=(2.5 * 4, 2.5 * 3))
plt.plot(
    results_only_m["s_values"] / true_nsig,
    2 * (results_only_m["s_profile"] - np.min(results_only_m["s_profile"])),
    color="black",
    label=r"Mass fit : $%.2f \pm %.2f $"
    % (results_only_m["best_s"] / true_nsig, (sigma_only_m[0] + sigma_only_m[1]) / 2.0),
    linestyle="dashed",
)


plt.plot(
    rescaled_cls_s_values,
    2 * (results_cls_m["s_profile"] - np.min(results_cls_m["s_profile"])),
    color="blue",
    label=r"Classifier cut + mass fit : $%.2f \pm %.2f$"
    % (
        results_cls_m["best_s"] / (cls_cut_sig_eff * true_nsig),
        (sigma_cls_mass[0] + sigma_cls_mass[1]) / 2.0,
    ),
    linestyle="dashed",
)

plt.plot(
    DNN_s_values / true_nsig,
    2 * (DNN_s_profile - np.min(DNN_s_profile)),
    color="green",
    label="Binned DNN score fit : $%.2f \pm %.2f$"
    % (DNN_best_s / true_nsig, (sigma_DNN[0] + sigma_DNN[1]) / 2.0),
    linestyle="dashed",
)


# overall envelope
plt.plot(
    mu_grid / true_nsig,
    explicit_profile_curve,
    color="black",
    label="HI-SIGMA : $%.2f \pm %.2f$"
    % (
        best_s_explicit_profile,
        (sigma_explicit_profile[0] + sigma_explicit_profile[1]) / 2.0,
    ),
    linestyle="solid",
)

# plot individual models
for i_bkg in range(Nbootstraps):
    for i_sig in range(Nbootstraps_sim):
        model_sigma = one_sig_CI(
            explicit_bootstraps[i_bkg][i_sig]["best_s"] / true_nsig,
            explicit_bootstraps[i_bkg][i_sig]["s_values"] / true_nsig,
            explicit_bootstraps[i_bkg][i_sig]["s_profile"],
        )
        plt.plot(
            explicit_bootstraps[i_bkg][i_sig]["s_values"] / true_nsig,
            2 * (all_profiles[i_bkg][i_sig] - np.min(NLL_explicit_profile)),
            label="Model %i : $%.2f \pm %.2f$"
            % (
                i_bkg,
                explicit_bootstraps[i_bkg][i_sig]["best_s"] / true_nsig,
                (model_sigma[0] + model_sigma[1]) / 2.0,
            ),
            linestyle="dotted",
        )

if do_sig_shape_syst or do_bkg_shape_syst:
    plt.plot(
        mu_grid_with_syst / true_nsig,
        explicit_profile_curve_with_syst,
        color="black",
        label="HI-SIGMA with syst, $%.2f \pm %.2f$"
        % (
            best_s_explicit_profile_with_syst,
            (sigma_explicit_profile_with_syst[0] + sigma_explicit_profile_with_syst[1])
            / 2.0,
        ),
        linestyle="dotted",
    )

ax = plt.gca()
ax.tick_params(axis="both", which="major", labelsize=18)
plt.ylim([0, ymax])
plt.xlabel("Signal Strength", fontsize=22)
plt.ylabel(r"- 2 $\Delta$ ln($\mathcal{L}$)", fontsize=22)
plt.legend(loc="upper center")
plt.savefig(plot_dir + "likelihood_cmp.pdf")
