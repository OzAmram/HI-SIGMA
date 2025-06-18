from utils import *
#Make summary plots of postfit results

iters = [0, 1, 2, 3, 4]
f_DNN = ["plots/fits_may7_4feats_n200/iter%i_smear0/DNN_fit.pkl" % i for i in iters]
f_mass_only = [ "plots/fits_may7_4feats_n200/iter%i_smear0/m_only_fit.pkl" % i for i in iters ]
f_cls_mass = [ "plots/fits_may7_4feats_n200/iter%i_smear0/cls_m_fit.pkl" % i for i in iters ]
# first model (for no sys comparison)
f_model = [ "plots/fits_may20_4feats_n200/iter%i_smear2/model0_sig0_fit.pkl" % i for i in iters ]
f_model_with_syst = [ "plots/fits_june4_4feats_n200/iter%i_smear15/model1_sig0_fit_with_syst.pkl" % i for i in iters ]

# including all models for profiling
nbootstraps = 5
iters = [0, 1, 2, 3, 4]
f_models = [
    [
        "plots/fits_may20_4feats_n200/iter%i_smear2/model%i_sig0_fit.pkl" % (i, imodel)
        for i in iters
    ]
    for imodel in range(nbootstraps)
]
f_models_with_syst = [
    [
        "plots/fits_june4_4feats_n200/iter%i_smear15/model%i_sig0_fit_with_syst.pkl"
        % (i, imodel)
        for i in iters
    ]
    for imodel in range(nbootstraps)
]

fit_res = [f_mass_only, f_cls_mass, f_DNN, f_model]
fit_res_with_syst = [f_model, f_model_with_syst]
multiple_fit_res_with_syst = [f_models, f_models_with_syst]
colors = ["black", "blue", "green", "red"]
linestyles = ["dashed", "dashed", "dashed", "dashed"]
labels = ["Mass fit", "DNN cut + mass fit", "Binned DNN score fit", "HI-SIGMA"]

true_nsig = 200

plot_dir = "plots/fits_may20_4feats_n200/"

do_shape_syst = False

if True:
    plt.figure()
    for i in range(len(fit_res)):
        unc = 0.0
        s_vals = None
        DLL_curve = None
        # average over the different iterations
        for j in iters:
            results = joblib.load(fit_res[i][j])

            # align all likelihoods for visualization
            corr_factor = true_nsig / results["best_s"]
            sigma_up, sigma_down = one_sig_CI(
                results["best_s"] / true_nsig,
                results["s_values"] / true_nsig,
                results["s_profile"],
            )
            unc += corr_factor * (sigma_up + sigma_down) / 2.0

            DLL_curve_ = 2 * (results["s_profile"] - np.min(results["s_profile"]))
            if j == 0:
                DLL_curve = np.array(DLL_curve_)
                s_vals = np.array(corr_factor * results["s_values"] / true_nsig)
            else:
                DLL_curve += np.array(DLL_curve_)
                s_vals += np.array(corr_factor * results["s_values"] / true_nsig)

        unc /= len(iters)
        DLL_curve /= len(iters)
        s_vals /= len(iters)
        label = labels[i] + r" : $\pm %.2f$" % unc
        print(fit_res[i][0], unc)
        plt.plot(
            s_vals, DLL_curve, color=colors[i], label=label, linestyle=linestyles[i]
        )

    ax = plt.gca()
    ax.tick_params(axis="both", which="major", labelsize=18)
    ymax = 12.0
    plt.ylim([0, ymax])
    plt.xlabel("Signal Strength", fontsize=22)
    plt.ylabel(r"- 2 $\Delta$ ln($\mathcal{L}$)", fontsize=22)
    plt.axhline(1, color="grey", linestyle="dotted")
    plt.axhline(4, color="grey", linestyle="dotted")
    # plt.axhline(9)
    leg = plt.legend(loc="upper center", fontsize=16)
    leg.get_frame().set_alpha(1)  # Set the alpha value to 0 for full transparency
    plt.savefig(plot_dir + "summary_likelihood_cmp.pdf", bbox_inches="tight")


overall_pulls = []
model_pulls = []

for j in range(len(iters)):
    plt.figure()
    all_profiles = []
    for i in range(len(f_models)):
        unc = 0.0
        s_vals = None
        DLL_curve = None
        # average over the different iterations
        results = joblib.load(f_models[i][j])

        all_profiles.append(results["s_profile"])
        s_grid = results["s_values"]

        # align all likelihoods for visualization
        corr_factor = 1.0
        sigma_up, sigma_down = one_sig_CI(
            results["best_s"] / true_nsig,
            results["s_values"] / true_nsig,
            results["s_profile"],
        )
        unc = (sigma_up + sigma_down) / 2.0

        DLL_curve = 2 * (results["s_profile"] - np.min(results["s_profile"]))

        s_vals = np.array(results["s_values"] / true_nsig)

        label = "Model %i" % i + r" : $%.2f \pm %.2f$" % (
            results["best_s"] / true_nsig,
            unc,
        )
        print(label)
        plt.plot(s_vals, DLL_curve, label=label, linestyle="dashed")

        model_pulls.append((results["best_s"] / true_nsig - 1.0) / unc)

    # profile curve
    likelihood_scales = np.min(all_profiles, axis=1) - np.min(all_profiles)
    print("likelihood_scales", likelihood_scales)

    all_profiles = [
        np.array(profile) - np.min(profile) for profile in all_profiles
    ]  # zero align
    explicit_profile = np.min(all_profiles, axis=0)
    eps = 1e-6
    best_s_explicit_profile = np.mean(s_grid[explicit_profile < eps]) / true_nsig
    sigma_explicit_profile = one_sig_CI(
        best_s_explicit_profile, s_grid / true_nsig, explicit_profile
    )
    explicit_profile_curve = 2 * (explicit_profile - np.min(explicit_profile))
    profile_unc = (sigma_explicit_profile[0] + sigma_explicit_profile[1]) / 2.0

    plt.plot(
        s_grid / true_nsig,
        explicit_profile_curve,
        color="black",
        label="Envelope : $%.2f \pm %.2f$" % (best_s_explicit_profile, profile_unc),
        linestyle="solid",
    )

    overall_pulls.append((best_s_explicit_profile - 1.0) / profile_unc)

    ax = plt.gca()
    ax.tick_params(axis="both", which="major", labelsize=18)
    ymax = 12.0
    plt.ylim([0, ymax])
    plt.xlabel("Signal Strength", fontsize=22)
    plt.ylabel(r"- 2 $\Delta$ ln($\mathcal{L}$)", fontsize=22)
    plt.axhline(1, color="grey", linestyle="dotted")
    plt.axhline(4, color="grey", linestyle="dotted")
    # plt.axhline(9)
    leg = plt.legend(loc="upper center", fontsize=14)
    leg.get_frame().set_alpha(1)  # Set the alpha value to 0 for full transparency
    plt.savefig(plot_dir + "summary_profiling_iter%i.pdf" % j, bbox_inches="tight")


print("model pulls")
print(model_pulls)
print(
    np.mean(model_pulls),
    np.std(model_pulls),
    np.mean(np.abs(model_pulls) < 1.0),
    np.mean(np.array(model_pulls) > 0.0),
)

print("Overall pulls")
print(overall_pulls)

if do_shape_syst:
    plt.figure()
    for i in range(len(fit_res_with_syst)):
        unc = 0.0
        s_vals = None
        DLL_curve = None
        # average over the different iterations
        for j in iters:
            all_profiles = []
            for nboot in range(nbootstraps):

                results = joblib.load(multiple_fit_res_with_syst[i][nboot][j])

                all_profiles.append(results["s_profile"])
                s_grid = results["s_values"]

            # profile curve
            likelihood_scales = np.min(all_profiles, axis=1) - np.min(all_profiles)
            print("likelihood_scales", likelihood_scales)

            all_profiles = [
                np.array(profile) - np.min(profile) for profile in all_profiles
            ]  # zero align
            explicit_profile = np.min(all_profiles, axis=0)
            eps = 1e-6
            best_s_explicit_profile = (
                np.mean(s_grid[explicit_profile < eps]) / true_nsig
            )
            sigma_explicit_profile = one_sig_CI(
                best_s_explicit_profile, s_grid / true_nsig, explicit_profile
            )
            explicit_profile_curve = 2 * (explicit_profile - np.min(explicit_profile))

            corr_factor = 1 / best_s_explicit_profile

            profile_unc = (sigma_explicit_profile[0] + sigma_explicit_profile[1]) / 2.0
            unc += corr_factor * profile_unc

            if j == 0:
                DLL_curve = np.array(explicit_profile_curve)
                s_vals = np.array(corr_factor * s_grid / true_nsig)
            else:
                DLL_curve += np.array(explicit_profile_curve)
                s_vals += np.array(corr_factor * s_grid / true_nsig)

        unc /= len(iters)
        DLL_curve /= len(iters)
        s_vals /= len(iters)
        label = labels[i] + r" : $\pm %.2f$" % unc
        print(fit_res_with_syst[i][0], unc)
        plt.plot(
            s_vals, DLL_curve, color=colors[i], label=label, linestyle=linestyles[i]
        )

    ax = plt.gca()
    ax.tick_params(axis="both", which="major", labelsize=18)
    ymax = 12.0
    plt.ylim([0, ymax])
    plt.xlabel("Signal Strength", fontsize=22)
    plt.ylabel(r"- 2 $\Delta$ ln($\mathcal{L}$)", fontsize=22)
    plt.axhline(1, color="grey", linestyle="dotted")
    plt.axhline(4, color="grey", linestyle="dotted")
    # plt.axhline(9)
    leg = plt.legend(loc="upper center", fontsize=16)
    leg.get_frame().set_alpha(1)  # Set the alpha value to 0 for full transparency
    plt.savefig(
        plot_dir + "summary_likelihood_cmp_averaged_over_bootstraps.pdf",
        bbox_inches="tight",
    )
