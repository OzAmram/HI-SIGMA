from utils import *

# half the events to train the classifier, half to make the templates


def train_classifier(
    train_dataset,
    val_dataset=None,
    weights=False,
    model_path="",
    device="cpu",
    nepochs=100,
    new_model=True,
    layers=[16, 32, 32, 8],
):

    input_dim = next(iter(train_dataset))[0].shape[1]
    cls_lr = 1e-4
    dropout = 0.2
    loss_fn = torch.nn.BCELoss(reduction="none")
    early_stop = 15

    model = DNN(input_dim=input_dim, layers=layers, dropout_probability=dropout)
    if not new_model and os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, weights_only=True))

    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cls_lr)

    early_stopper = EarlyStopper(patience=early_stop)
    batched_W = None

    min_val = 99999.0
    for nepoch in range(nepochs):
        total_loss = 0.0
        model.train()
        for ibatch, batch in enumerate(train_dataset):
            if weights:
                batched_X, batched_y, batched_W = batch
            else:
                batched_X, batched_y = batch

            optimizer.zero_grad()
            preds = model(batched_X)

            loss = loss_fn(preds, batched_y)
            if batched_W is not None:
                loss *= batched_W
            loss = loss.mean()

            loss.backward()
            optimizer.step()
            total_loss += loss.item() / len(train_dataset)

        model.eval()
        val_loss = 0.0
        for ibatch, batch in enumerate(val_dataset):
            if weights:
                batched_X, batched_y, batched_W = batch
            else:
                batched_X, batched_y = batch

            preds = model(batched_X)

            loss = loss_fn(preds, batched_y)
            if batched_W is not None:
                loss *= batched_W
            loss = loss.mean()

            val_loss += loss.item() / len(val_dataset)

        if nepoch == 1 or nepoch % 10 == 0:
            print(nepoch, total_loss, val_loss)

        if val_loss < min_val:
            min_val = val_loss
            torch.save(model.state_dict(), model_path)

        if early_stopper.early_stop(val_loss):
            print("Early stopping!")
            print("Epoch %i train %.4f val %.4f" % (nepoch, total_loss, val_loss))
            break
    # restore best model
    model.load_state_dict(torch.load(model_path, weights_only=True))
    model.eval()

    return model


if __name__ == "__main__":
    working_dir = ""
    # data_dir = os.path.join(working_dir,'sim_example_data/')
    data_dir = os.path.join(working_dir, "sim_data_4feats_smear0/")
    # data_dir = os.path.join(working_dir,'sim_data_7feats/')
    data = np.load(data_dir + "data.npy")
    labels = np.load(data_dir + "labels.npy")
    scaler = joblib.load(data_dir + "bkg_scaler.pkl")

    sim = np.load(data_dir + "sim.npy")

    # bkg only
    bkg = data[labels < 0.5]

    do_distortion = False
    new_model = True
    # cls_dir = "supervised_classifier_2feats/"
    cls_dir = "supervised_classifier_4feats_smear0/"
    # cls_dir = "supervised_classifier_7feats/"
    nepochs = 100
    plot_dir = cls_dir + "plots/"
    os.makedirs(plot_dir, exist_ok=True)
    DNN_nbins = 50
    DNN_bins = np.linspace(0, 1, DNN_nbins + 1)

    # account for smearing
    data[:, 1:] = scaler.inverse_transform(scaler.transform(data[:, 1:]))

    device = "cuda"

    test_frac = 0.2
    val_frac = 0.2

    # SR mask
    mass_low_sr = convert_mgg(115.0)
    mass_high_sr = convert_mgg(135.0)

    sig_SR_mask = (sim[:, 0] > mass_low_sr) & (sim[:, 0] < mass_high_sr)
    bkg_SR_mask = (bkg[:, 0] > mass_low_sr) & (bkg[:, 0] < mass_high_sr)

    sig = sim[sig_SR_mask]
    bkg = bkg[bkg_SR_mask]

    print(np.mean(bkg, axis=0))

    for i in range(sig.shape[1]):
        plt.figure()

        n, bins, _ = plt.hist(
            sig[:, i], color="red", histtype="step", bins=50, label="sig", density=True
        )
        plt.hist(
            bkg[:, i],
            color="blue",
            histtype="step",
            bins=bins,
            label="bkg (SR)",
            density=True,
        )
        plt.xlabel("Feat %i" % i)
        plt.legend()
        plt.savefig(plot_dir + "feat%i" % i + ".png")

    sig_data, sig_data_test = train_test_split(
        sig, test_size=test_frac, random_state=42
    )
    sig_data_train, sig_data_val = train_test_split(
        sig_data, test_size=val_frac, random_state=42
    )

    bkg_data, bkg_data_test = train_test_split(
        bkg, test_size=test_frac, random_state=42
    )

    # Distort bkg training data to mock mismodeled MC

    if do_distortion:
        # slightly distort training data to mock imperfect MC modeling of bkg
        bkg_data_distort = np.copy(bkg_data)
        # dR gam gam, dR bb
        bkg_data_distort[:, 1] += 0.2 * np.random.randn(bkg_data.shape[0])
        bkg_data_distort[:, 2] += 0.2 * np.random.randn(bkg_data.shape[0])
        # Hgg pt/m, Hbb pt
        bkg_data_distort[:, 3] += 0.05 * (
            1.0 + 0.2 * np.random.randn(bkg_data.shape[0])
        )
        bkg_data_distort[:, 4] += 7.0 * (1.0 + 0.2 * np.random.randn(bkg_data.shape[0]))

        for i in range(1, bkg_data.shape[1]):
            plt.figure()
            n, bins, _ = plt.hist(
                bkg_data[:, i],
                bins=30,
                histtype="step",
                linestyle="solid",
                label="Truth",
                color="black",
            )
            n, bins, _ = plt.hist(
                bkg_data_distort[:, i],
                bins=bins,
                histtype="step",
                linestyle="solid",
                label="Mock MC Distortion",
                color="red",
            )
            plt.xlabel("Feature %i" % i)
            plt.legend(loc="best")
            plt.savefig(plot_dir + "feat%i_distortion.png" % i)

        bkg_data = bkg_data_distort

    bkg_data_train, bkg_data_val = train_test_split(
        bkg_data, test_size=val_frac, random_state=42
    )
    print("train mean", np.mean(bkg_data_train, axis=0))
    print("train std", np.std(bkg_data_train, axis=0))
    print("test mean", np.mean(bkg_data_test, axis=0))
    print("test std", np.std(bkg_data_test, axis=0))

    labels_train = np.concatenate(
        (
            torch.ones(sig_data_train.shape[0], 1),
            torch.zeros(bkg_data_train.shape[0], 1),
        )
    )
    data_train = np.concatenate((sig_data_train, bkg_data_train))
    # equalize weights in training
    weights_train = np.ones((data_train.shape[0], 1))
    norm = np.sum(bkg_data_train.shape[0]) / np.sum(sig_data_train.shape[0])
    weights_train[labels_train.reshape(-1) == 1] = norm
    weights_train /= np.mean(weights_train)

    print(
        "Training on %i bkg and %i signal"
        % (bkg_data_train.shape[0], sig_data_train.shape[0])
    )

    labels_val = np.concatenate(
        (torch.ones(sig_data_val.shape[0], 1), torch.zeros(bkg_data_val.shape[0], 1))
    )
    data_val = np.concatenate((sig_data_val, bkg_data_val))
    # equalize weights
    weights_val = np.ones((data_val.shape[0], 1))
    norm = np.sum(bkg_data_val.shape[0]) / np.sum(sig_data_val.shape[0])
    weights_val[labels_val.reshape(-1) == 1] = norm
    weights_val /= np.mean(weights_val)

    labels_test = np.concatenate(
        (torch.ones(sig_data_test.shape[0], 1), torch.zeros(bkg_data_test.shape[0], 1))
    )
    data_test = torch.tensor(
        np.concatenate((sig_data_test, bkg_data_test)), dtype=torch.float32
    ).to(device)

    # DNN includes batchnorm so no preprocessing

    data_train_no_mass = data_train[:, 1:]
    data_val_no_mass = data_val[:, 1:]

    print(np.mean(data_train_no_mass[labels_train.reshape(-1) < 0.5], axis=0))
    print(np.std(data_train_no_mass[labels_train.reshape(-1) < 0.5], axis=0))

    batch_size = 256
    train_cls_dataset_no_mass = xyDataset(
        data_train_no_mass, labels_train, weights_train
    ).to(device)
    train_cls_dataset_no_mass_batched = DataLoader(
        train_cls_dataset_no_mass, batch_size=batch_size, shuffle=True
    )

    val_cls_dataset_no_mass = xyDataset(data_val_no_mass, labels_val, weights_val).to(
        device
    )
    val_cls_dataset_no_mass_batched = DataLoader(
        val_cls_dataset_no_mass, batch_size=batch_size, shuffle=False
    )

    cls_model_path = cls_dir + "model_no_mass.par"

    print("Training classifier on just features")
    classifier_no_mass = train_classifier(
        train_cls_dataset_no_mass_batched,
        val_dataset=val_cls_dataset_no_mass_batched,
        nepochs=nepochs,
        model_path=cls_model_path,
        weights=True,
        device=device,
        new_model=new_model,
    )

    cls_full_model_path = cls_dir + "model_full.par"

    train_cls_dataset = xyDataset(data_train, labels_train, weights_train).to(device)
    train_cls_dataset_batched = DataLoader(
        train_cls_dataset, batch_size=batch_size, shuffle=True
    )

    val_cls_dataset = xyDataset(data_val, labels_val, weights_val).to(device)
    val_cls_dataset_batched = DataLoader(
        val_cls_dataset, batch_size=batch_size, shuffle=True
    )

    print("Training classifier on full info (features and mass)")
    classifier_full = train_classifier(
        train_cls_dataset_batched,
        val_dataset=val_cls_dataset_batched,
        nepochs=nepochs,
        model_path=cls_full_model_path,
        weights=True,
        device=device,
        new_model=new_model,
    )

    preds_no_m_test = classifier_no_mass(data_test[:, 1:]).detach().cpu().numpy()
    preds_test = classifier_full(data_test).detach().cpu().numpy()

    print(torch.mean(data_test[labels_test.reshape(-1) < 0.5], axis=0))
    print(
        np.mean(preds_test[labels_test.reshape(-1) < 0.5]),
        np.mean(preds_test[labels_test.reshape(-1) > 0.5]),
    )

    eps = 1e-8
    fpr1, tpr1, _ = roc_curve(labels_test, preds_test)
    auc1 = roc_auc_score(labels_test, preds_test)
    sic1 = tpr1 / (np.sqrt(fpr1) + eps)
    sic1 = np.clip(sic1, None, 10)

    fpr2, tpr2, _ = roc_curve(labels_test, preds_no_m_test)
    auc2 = roc_auc_score(labels_test, preds_no_m_test)
    sic2 = tpr2 / (np.sqrt(fpr2) + eps)
    sic2 = np.clip(sic2, None, 10)

    print("classifier (full info), auc=%.3f" % auc1)
    print("classifier (no mass), auc=%.3f" % auc2)

    plt.figure()
    plt.plot(fpr1, sic1, label="classifier (full info), auc=%.3f" % auc1)
    plt.plot(fpr2, sic2, label="classifier (no mass), auc=%.3f" % auc2)

    plt.xlabel("False Positive Rate")
    plt.ylabel("Significance Improvement")
    plt.xscale("log")
    plt.legend(loc="upper right")
    plt.savefig(plot_dir + "classifier_sic.png")

    plt.figure()
    plt.plot(tpr1, 1.0 / (fpr1 + eps), label="classifier (full info), auc=%.3f" % auc1)
    plt.plot(tpr2, 1.0 / (fpr2 + eps), label="classifier (no mass), auc=%.3f" % auc2)

    plt.xlabel("True Positive Rate")
    plt.ylabel(" 1 / False Positive Rate")
    plt.yscale("log")

    plt.legend(loc="upper right")
    plt.savefig(plot_dir + "classifier_roc.png")

    sig_mask = labels_test.reshape(-1) > 0.5
    # make templates
    plt.figure()
    print(DNN_bins)
    # sig_vals,_,_ = plt.hist(preds_test[sig_mask],bins=bins,  color='red',  label = 'Sig', histtype='step', density = True)
    # bkg_vals,_,_ = plt.hist(preds_test[~sig_mask], bins=bins, color='blue', label = 'Bkg', histtype='step', density = True)

    sig_vals, _ = np.histogram(preds_test[sig_mask], bins=DNN_bins)
    bkg_vals, _ = np.histogram(preds_test[~sig_mask], bins=DNN_bins)

    sig_vals = sig_vals / np.sum(sig_vals)
    bkg_vals = bkg_vals / np.sum(bkg_vals)

    plt.stairs(sig_vals, DNN_bins, color="red", label="sig")
    plt.stairs(bkg_vals, DNN_bins, color="blue", label="bkg")

    plt.savefig(plot_dir + "templates.png")

    print(len(sig_vals))
    print(bkg_vals)

    out = {
        "sig": sig_vals,
        "bkg": bkg_vals,
        "bins": DNN_bins,
    }
    joblib.dump(out, cls_dir + "templates.pkl")
