from utils import *


def check_modeling(model, data, plot_dir="plots/", scaler=None):
    model.to("cpu")
    # first index is mass
    sample = (
        model.sample(
            num_samples=1, context=torch.tensor(data[:, 0:1], dtype=torch.float32)
        )[:, 0]
        .detach()
        .numpy()
    )

    for nfeature in range(sample.shape[1]):
        plt.figure()
        a, b, c = plt.hist(
            data[:, 1 + nfeature],
            histtype="step",
            density=False,
            bins=30,
            color="black",
            label="Unlabelled data",
        )
        plt.hist(
            sample[:, nfeature],
            histtype="step",
            density=False,
            bins=b,
            color="magenta",
            label="Flow samples",
        )
        plt.xlabel("$x$")
        plt.ylabel("Events")
        plt.savefig(plot_dir + "feat%i_scaled_modeling.png" % (nfeature))
        plt.close()

    if scaler is not None:
        sample_unscaled = scaler.inverse_transform(sample)
        data_unscaled = scaler.inverse_transform(data[:, 1:])

        for nfeature in range(sample.shape[1]):
            plt.figure()
            a, b, c = plt.hist(
                data_unscaled[:, nfeature],
                histtype="step",
                density=False,
                bins=30,
                color="black",
                label="Unlabelled data",
            )
            plt.hist(
                sample_unscaled[:, nfeature],
                histtype="step",
                density=False,
                bins=b,
                color="magenta",
                label="Flow samples",
            )
            plt.xlabel("$x$")
            plt.ylabel("Events")
            plt.savefig(plot_dir + "feat%i_modeling.png" % (nfeature))
            plt.close()


def check_modeling_with_syst(model, data, plot_dir="plots/", scaler=None):
    model.to("cpu")
    # first index is mass
    sample = (
        model.sample(
            num_samples=1,
            context=torch.tensor(
                np.hstack([data[:, 0:1], np.zeros(len(data)).reshape(-1, 1)]),
                dtype=torch.float32,
            ),
        )[:, 0]
        .detach()
        .numpy()
    )

    for nfeature in range(sample.shape[1]):
        plt.figure()
        a, b, c = plt.hist(
            data[:, 1 + nfeature],
            histtype="step",
            density=False,
            bins=30,
            color="black",
            label="Unlabelled data",
        )
        plt.hist(
            sample[:, nfeature],
            histtype="step",
            density=False,
            bins=b,
            color="magenta",
            label="Flow samples",
        )
        plt.xlabel("$x$")
        plt.ylabel("Events")
        plt.savefig(plot_dir + "feat%i_scaled_modeling.png" % (nfeature))
        plt.close()

    if scaler is not None:
        sample_unscaled = scaler.inverse_transform(sample)
        data_unscaled = scaler.inverse_transform(data[:, 1:])

        for nfeature in range(sample.shape[1]):
            plt.figure()
            a, b, c = plt.hist(
                data_unscaled[:, nfeature],
                histtype="step",
                density=False,
                bins=30,
                color="black",
                label="Unlabelled data",
            )
            plt.hist(
                sample_unscaled[:, nfeature],
                histtype="step",
                density=False,
                bins=b,
                color="magenta",
                label="Flow samples",
            )
            plt.xlabel("$x$")
            plt.ylabel("Events")
            plt.savefig(plot_dir + "feat%i_modeling.png" % (nfeature))
            plt.close()


def train_flow(
    data=None,
    val=None,
    new_model=False,
    model_path="test.h5",
    plot_dir="",
    nepochs=200,
    early_stop=10,
    num_layers=3,
    hidden_size=32,
    num_bins=8,
):
    early_stopper = EarlyStopper(patience=early_stop)

    batch_size = 256
    learning_rate = 0.001

    nfeatures = data.shape[1] - 1
    flowB = flow_model(
        nfeatures,
        num_layers=num_layers,
        hidden_size=hidden_size,
        RQS=RQS,
        num_bins=num_bins,
    )

    if not new_model:
        try:
            flowB.load_state_dict(torch.load(model_path, weights_only=True))
            print("Loading model from %s" % model_path)
        except:
            print("New model")

    optimizer = torch.optim.Adam(flowB.parameters(), lr=learning_rate)

    ### because I'm doing a 1d, I need to add redundant information

    train_dataset = xyDataset(data[:, 1:], data[:, 0:1]).to(device)
    train_dataset_batched = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True
    )

    val_dataset = xyDataset(val[:, 1:], val[:, 0:1]).to(device)
    val_dataset_batched = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    flowB.to(device)
    min_val = 9999.0
    for nepoch in range(1, nepochs + 1):
        total_loss = 0
        flowB.train()
        for batch, (batched_X, batched_m) in enumerate(train_dataset_batched):
            # batched_X = batched_X.to(device)
            # batched_m = batched_m.to(device)
            optimizer.zero_grad()
            loss = -flowB.log_prob(inputs=batched_X, context=batched_m).mean()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        val_loss = 0
        flowB.eval()
        for batch, (batched_X, batched_m) in enumerate(val_dataset_batched):
            loss = -flowB.log_prob(inputs=batched_X, context=batched_m).mean()
            val_loss += loss.item()

        total_loss /= len(train_dataset_batched)
        val_loss /= len(val_dataset_batched)

        # Save if lower val loss
        if val_loss < min_val:
            torch.save(flowB.state_dict(), model_path)
            min_val = val_loss
        if nepoch == 1 or nepoch % 10 == 0:
            print("Epoch %i train %.4f val %.4f" % (nepoch, total_loss, val_loss))

        if early_stopper.early_stop(val_loss):
            print("Early stopping!")
            print("Epoch %i train %.4f val %.4f" % (nepoch, total_loss, val_loss))
            break

    # Load the best model
    flowB.load_state_dict(torch.load(model_path, weights_only=True))
    return flowB


def train_flow_with_syst(
    data=None,
    val=None,
    variation_index=None,
    feats_percentual_uncertainty=None,
    new_model=False,
    model_path="test.h5",
    plot_dir="",
    nepochs=200,
    early_stop=10,
    num_layers=3,
    hidden_size=32,
    num_bins=8,
):

    early_stopper = EarlyStopper(patience=early_stop)

    batch_size = 256
    learning_rate = 0.0001

    nfeatures = data.shape[1] - 1
    flowB = flow_model_with_syst(
        nfeatures,
        num_layers=num_layers,
        hidden_size=hidden_size,
        RQS=RQS,
        num_bins=num_bins,
    )

    if not new_model:
        try:
            flowB.load_state_dict(torch.load(model_path, weights_only=True))
            print("Loading model from %s" % model_path)
        except:
            print("New model")

    optimizer = torch.optim.Adam(flowB.parameters(), lr=learning_rate)

    ### because I'm doing a 1d, I need to add redundant information

    alpha = np.random.normal(loc=0.0, scale=1.0, size=data.shape[0])

    print("Testing adding alpha to mass")
    print(np.hstack([data[:, 0:1], alpha.reshape(-1, 1)]).shape)
    print(np.mean(alpha), np.std(alpha))

    data[:, 1:] = scaler.inverse_transform(data[:, 1:])
    data[:, 1:] = scaler.transform(
        data[:, 1:]
        + np.einsum(
            "nk,n,k,k->nk",
            np.abs(data[:, 1:]),
            alpha,
            variation_index,
            feats_percentual_uncertainty,
        )
    )

    train_dataset = xyDataset(
        data[:, 1:], np.hstack([data[:, 0:1], (1 / 5) * alpha.reshape(-1, 1)])
    ).to(device)
    train_dataset_batched = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True
    )

    alpha_val = np.random.normal(loc=0.0, scale=1.0, size=val.shape[0])

    val[:, 1:] = scaler.inverse_transform(val[:, 1:])
    val[:, 1:] = scaler.transform(
        val[:, 1:]
        + np.einsum(
            "nk,n,k,k->nk",
            np.abs(val[:, 1:]),
            alpha_val,
            variation_index,
            feats_percentual_uncertainty,
        )
    )

    val_dataset = xyDataset(
        val[:, 1:], np.hstack([val[:, 0:1], (1 / 5) * alpha_val.reshape(-1, 1)])
    ).to(device)
    val_dataset_batched = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    flowB.to(device)
    min_val = 9999.0
    for nepoch in range(1, nepochs + 1):

        total_loss = 0
        flowB.train()
        for batch, (batched_X, batched_m) in enumerate(train_dataset_batched):
            # batched_X = batched_X.to(device)
            # batched_m = batched_m.to(device)
            optimizer.zero_grad()
            loss = -flowB.log_prob(inputs=batched_X, context=batched_m).mean()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        val_loss = 0
        flowB.eval()
        for batch, (batched_X, batched_m) in enumerate(val_dataset_batched):
            loss = -flowB.log_prob(inputs=batched_X, context=batched_m).mean()
            val_loss += loss.item()

        total_loss /= len(train_dataset_batched)
        val_loss /= len(val_dataset_batched)

        # Save if lower val loss
        if val_loss < min_val:
            torch.save(flowB.state_dict(), model_path)
            min_val = val_loss
        if nepoch == 1 or nepoch % 10 == 0:
            print("Epoch %i train %.4f val %.4f" % (nepoch, total_loss, val_loss))

        if early_stopper.early_stop(val_loss):
            print("Early stopping!")
            print("Epoch %i train %.4f val %.4f" % (nepoch, total_loss, val_loss))
            break

    # Load the best model
    flowB.load_state_dict(torch.load(model_path, weights_only=True))
    return flowB


working_dir = ""
data_dir = os.path.join(working_dir, "datasets/4feats_smear15/")

Nbootstraps = 5
Nensemble = 5
nepochs = 100
new_model = True
do_sig = False
SR_train = False
model_dir = os.path.join(working_dir, "bkg_models/4feats_smear15/")
# model_dir = os.path.join(working_dir,'sig_models/4feats_smear15/')
print("Writing models to %s" % model_dir)

cutoff = 5

# flow params
num_layers = 5
hidden_size = 32
flow_nbins = 24
RQS = True

test_frac = 0.15
val_frac = 0.15

do_shape_syst = False
feats_percentual_uncertainty = np.array([0.01, 0.01, 0.01, 0.01])
variations = np.array([[0, 0, 0, 1], [0, 0, 0, -1]])
variations_names = ["H_bb_up_percent_unc", "H_bb_down_percent_unc"]


os.makedirs(model_dir, exist_ok=True)
device = "cuda"


if not do_sig:

    data = np.load(data_dir + "data.npy")
    labels = np.load(data_dir + "labels.npy")

    # remove signal
    data = data[labels.reshape(-1) < 0.5]

    label = "background"

    print("Training bkg flow on background ")

    # train in sidebands only
    mass_low_sr = convert_mgg(115.0)
    mass_high_sr = convert_mgg(135.0)
    SB_mask = (data[:, 0] < mass_low_sr) | (data[:, 0] > mass_high_sr)

    mask = ~SB_mask if SR_train else SB_mask
    print(" Mask eff", np.mean(mask))
    data = data[mask]
    scaler = joblib.load(data_dir + "bkg_scaler.pkl")

else:
    label = "signal"
    data = np.load(data_dir + "sim.npy")
    scaler = joblib.load(data_dir + "sig_scaler.pkl")
    print("Training signal flow on sim")


print("Training on %i events" % data.shape[0])

mean_data_feats = np.mean(data, axis=0)
above_avg = [data[:, i] > mean_data_feats[i] for i in range(data.shape[1])]
data[:, 1:] = scaler.transform(data[:, 1:])

print("initial Nans")
print(np.sum(np.isnan(data)))
# cleanup outliers, different procedure for high vs low nans
for i in range(data.shape[1]):
    data[:, i] = clean_nans(data[:, i], above_avg[i])

# cut out extreme outliers which could harm the training
data[:, 1:] = np.clip(data[:, 1:], -cutoff, cutoff)

# print(np.min(data, axis=0), np.max(data, axis=0))

if do_shape_syst:
    fig, axes = plt.subplots(2, 2, figsize=(5 * 4, 5 * 3))

    # plt.show()
    colors = ["black", "blue", "red"]

    _, b1 = np.histogram(data[:, 1], bins=40)
    _, b2 = np.histogram(data[:, 2], bins=40)
    _, b3 = np.histogram(data[:, 3], bins=40)
    _, b4 = np.histogram(data[:, 4], bins=40)

    # for nvariation, (variation, variation_name,color) in enumerate(zip(np.array([[0,0]]+list(variations)),["Nominal"]+variations_names,colors)):
    for nvariation, (variation, variation_name, color) in enumerate(
        zip(
            np.array([[0, 0, 0, 0]] + list(variations)),
            ["Nominal"] + variations_names,
            colors,
        )
    ):

        data_syst = data.copy()
        data_syst[:, 1:] = scaler.inverse_transform(data[:, 1:])
        data_syst[:, 1:] = scaler.transform(
            data_syst[:, 1:]
            + np.einsum(
                "nk,k,k->nk",
                np.abs(data_syst[:, 1:]),
                variation,
                feats_percentual_uncertainty,
            )
        )
        # print(nvariation,variation,variation_name,data_syst[:10,1:])

        # SR_mask = [a and b for a,b in zip(data[:,0]>=mask_limit[0],data[:,0]<mask_limit[1])]
        # for nfeature in range(nfeatures):
        a, b, c = axes[0, 0].hist(
            data_syst[:, 1],
            histtype="step",
            density=True,
            bins=b1,
            label=variation_name,
            color=color,
        )
        aN, b = np.histogram(data_syst[:, 1], bins=b)
        axes[0, 0].errorbar(
            0.5 * (b[1:] + b[:-1]),
            aN / (len(data) * (b[1] - b[0])),
            yerr=np.sqrt(aN) / (len(data) * (b[1] - b[0])),
            fmt="o",
            color=color,
        )

        a, b, c = axes[0, 1].hist(
            data_syst[:, 2],
            histtype="step",
            density=True,
            bins=b2,
            label=variation_name,
            color=color,
        )
        aN, b = np.histogram(data_syst[:, 2], bins=b)
        axes[0, 1].errorbar(
            0.5 * (b[1:] + b[:-1]),
            aN / (len(data) * (b[1] - b[0])),
            yerr=np.sqrt(aN) / (len(data) * (b[1] - b[0])),
            fmt="o",
            color=color,
        )

        a, b, c = axes[1, 0].hist(
            data_syst[:, 3],
            histtype="step",
            density=True,
            bins=b3,
            label=variation_name,
            color=color,
        )
        aN, b = np.histogram(data_syst[:, 3], bins=b)
        axes[1, 0].errorbar(
            0.5 * (b[1:] + b[:-1]),
            aN / (len(data) * (b[1] - b[0])),
            yerr=np.sqrt(aN) / (len(data) * (b[1] - b[0])),
            fmt="o",
            color=color,
        )

        a, b, c = axes[1, 1].hist(
            data_syst[:, 4],
            histtype="step",
            density=True,
            bins=b4,
            label=variation_name,
            color=color,
        )
        aN, b = np.histogram(data_syst[:, 4], bins=b)
        axes[1, 1].errorbar(
            0.5 * (b[1:] + b[:-1]),
            aN / (len(data) * (b[1] - b[0])),
            yerr=np.sqrt(aN) / (len(data) * (b[1] - b[0])),
            fmt="o",
            color=color,
        )

    # axes[0,0[.set_xlabel('$x$')
    # plt.ylabel('Events')
    # plt.show()
    axes[0, 0].legend(loc="best")
    plt.savefig(model_dir + "validation_variations.png")


# holdout a test set from all models
data_train, data_test = train_test_split(data, test_size=test_frac, random_state=42)


data_test_mass = data_test[:, 0:1]
data_test_feats = data_test[:, 1:]

indexes = [i for i in range(len(data))]
np.random.seed(42)
indexes_boostrapped = [
    np.random.choice(indexes, size=len(data)) for n in range(Nbootstraps)
]

models = []

for nboot in range(Nbootstraps):
    # for nboot in range(3,Nbootstraps):
    print("Boot " + str(nboot))

    # random split train / val for each boot strap
    train_boot, val_boot = train_test_split(
        data_train, test_size=val_frac, random_state=nboot
    )

    ensemble_test_loss = 0.0

    for nensemble in range(Nensemble):
        print("Model %i-%i" % (nboot, nensemble))
        model_path = os.path.join(
            model_dir, "%s_model_boot%i_%i.par" % (label, nboot, nensemble)
        )

        plot_dir = model_dir + "model%i_ensemble%i_feats/" % (nboot, nensemble)
        os.makedirs(plot_dir, exist_ok=True)

        model = train_flow(
            data=train_boot,
            val=val_boot,
            new_model=new_model,
            model_path=model_path,
            plot_dir=plot_dir,
            nepochs=nepochs,
            num_layers=num_layers,
            hidden_size=hidden_size,
            num_bins=flow_nbins,
        )

        model.eval()
        check_modeling(model, val_boot, plot_dir, scaler=scaler)
        test_loss = (
            -model.log_prob(
                inputs=torch.tensor(data_test_feats, dtype=torch.float32),
                context=torch.tensor(data_test_mass, dtype=torch.float32),
            )
            .detach()
            .cpu()
            .numpy()
        )

        print("Test loss", np.mean(test_loss))
        ensemble_test_loss += test_loss

    ensemble_test_loss /= Nensemble
    print("Ensemble %i test loss %.4f" % (nboot, np.mean(ensemble_test_loss)))

    if do_shape_syst:

        for nvariation, (variation, variation_name) in enumerate(
            zip(variations, variations_names)
        ):

            data_syst = data.copy()
            data_syst[:, 1:] = scaler.inverse_transform(data[:, 1:])
            data_syst[:, 1:] = scaler.transform(
                data_syst[:, 1:]
                + np.einsum(
                    "nk,k,k->nk",
                    np.abs(data_syst[:, 1:]),
                    variation,
                    feats_percentual_uncertainty,
                )
            )
            train_boot_syst = train_boot.copy()
            train_boot_syst[:, 1:] = scaler.inverse_transform(train_boot[:, 1:])
            train_boot_syst[:, 1:] = scaler.transform(
                train_boot_syst[:, 1:]
                + np.einsum(
                    "nk,k,k->nk",
                    np.abs(train_boot[:, 1:]),
                    variation,
                    feats_percentual_uncertainty,
                )
            )

            val_boot_syst = val_boot.copy()
            val_boot_syst[:, 1:] = scaler.inverse_transform(val_boot[:, 1:])
            val_boot_syst[:, 1:] = scaler.transform(
                val_boot_syst[:, 1:]
                + np.einsum(
                    "nk,k,k->nk",
                    np.abs(val_boot[:, 1:]),
                    variation,
                    feats_percentual_uncertainty,
                )
            )

            data_test_feats_syst = data_test_feats.copy()
            data_test_feats_syst = scaler.inverse_transform(data_test_feats)
            data_test_feats_syst = scaler.transform(
                data_test_feats_syst
                + np.einsum(
                    "nk,k,k->nk",
                    np.abs(data_test_feats),
                    variation,
                    feats_percentual_uncertainty,
                )
            )

            ensemble_test_loss = 0.0

            for nensemble in range(Nensemble):
                print("Model %i-%i" % (nboot, nensemble))
                model_path = os.path.join(
                    model_dir,
                    "%s_model_boot%i_%i_%s.par"
                    % (label, nboot, nensemble, variation_name),
                )

                plot_dir = model_dir + "model%i_ensemble%i_%s_feats/" % (
                    nboot,
                    nensemble,
                    variation_name,
                )
                os.makedirs(plot_dir, exist_ok=True)

                model = train_flow(
                    data=train_boot_syst,
                    val=val_boot_syst,
                    new_model=new_model,
                    model_path=model_path,
                    plot_dir=plot_dir,
                    nepochs=nepochs,
                    num_layers=num_layers,
                    hidden_size=hidden_size,
                    num_bins=flow_nbins,
                )

                model.eval()
                check_modeling(model, val_boot_syst, plot_dir, scaler=scaler)
                test_loss = (
                    -model.log_prob(
                        inputs=torch.tensor(data_test_feats_syst, dtype=torch.float32),
                        context=torch.tensor(data_test_mass, dtype=torch.float32),
                    )
                    .detach()
                    .cpu()
                    .numpy()
                )

                print("Test loss", np.mean(test_loss))
                ensemble_test_loss += test_loss
