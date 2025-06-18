#!/usr/bin/env python

import os, sys, copy

os.environ["OMP_NUM_THREADS"] = "20"

import numpy as np
import h5py
import joblib
import scipy.stats as st
import scipy.optimize as scipy_optimizer
from scipy.stats import expon
from scipy.special import erf

from sklearn.preprocessing import MinMaxScaler, StandardScaler

import matplotlib.pyplot as plt
from matplotlib import rc
from matplotlib import gridspec

# font = {'size'   : 25}
# rc('font', **font)
# plt.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}'
# rc('text', usetex=True)
# plt.rcParams['font.family']='Computer Modern'

from scipy.spatial.distance import cdist
from sklearn.base import BaseEstimator
from sklearn.preprocessing import StandardScaler, MinMaxScaler, QuantileTransformer
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, roc_auc_score
from scipy.interpolate import CubicSpline


import torch

torch.set_num_threads(5)
from torch import nn
from tqdm import tqdm
from torch.nn.modules import Module
from torch.utils.data import DataLoader
import torch.distributions as D

from torch import optim

from numba_stats import norm, bernstein, poisson, truncexpon
from numba_stats import crystalball_ex as cb
from numbers import Real
from scipy.special import logit, expit


import iminuit
from iminuit import cost, Minuit
from scipy import integrate

import nflows
from nflows.flows.base import Flow
from nflows.distributions.normal import StandardNormal, ConditionalDiagonalNormal
from nflows.distributions.uniform import MG1Uniform, BoxUniform
from nflows.transforms.base import CompositeTransform
from nflows.transforms.autoregressive import (
    MaskedAffineAutoregressiveTransform,
    MaskedPiecewiseRationalQuadraticAutoregressiveTransform,
)
from nflows.transforms.permutations import ReversePermutation
from nflows.transforms.normalization import BatchNorm
from nflows.nn.nets import ResidualNet


class SmearScaler(BaseEstimator):

    def __init__(self, scale=0.01, copy=True):

        self.copy = copy
        self.noise_scales = None
        self.stds = None
        self.scale = scale

    def fit(self, X, y=None):
        self.stds = np.std(X, axis=0)
        self.noise_scales = self.scale * self.stds
        return self

    def transform(self, X, y=None, copy=None):
        rnd = np.random.randn(*X.shape)
        z = X + self.noise_scales * rnd
        return z

    def inverse_transform(self, X, y=None):
        return X

    def fit_transform(self, X, y=None):
        return self.fit(X).transform(X)


class LogitScaler(MinMaxScaler):
    """Preprocessing scaler that performs a logit transformation on top
    of the sklean MinMaxScaler. It scales to a range [0+epsilon, 1-epsilon]
    before applying the logit. Setting a small finitie epsilon avoids
    features being mapped to exactly 0 and 1 before the logit is applied.
    If the logit does encounter values beyond (0, 1), it outputs nan for
    these values.
    """

    _parameter_constraints: dict = {
        "epsilon": [Real],
        "copy": ["boolean"],
        "clip": ["boolean"],
    }

    def __init__(self, epsilon=0, copy=True, clip=False):
        self.epsilon = epsilon
        self.copy = copy
        self.clip = clip
        super().__init__(feature_range=(0 + epsilon, 1 - epsilon), copy=copy, clip=clip)

    def fit(self, X, y=None):
        super().fit(X, y)
        return self

    def transform(self, X):
        z = logit(super().transform(X))
        z[np.isinf(z)] = np.nan
        return z

    def fit_transform(self, X, y=None):
        return self.fit(X).transform(X)

    def inverse_transform(self, X):
        return super().inverse_transform(expit(X))

    def jacobian_determinant(self, X):
        z = super().transform(X)
        return np.prod(z * (1 - z), axis=1, keepdims=True) / np.prod(self.scale_)

    def log_jacobian_determinant(self, X):
        z = super().transform(X)
        return np.sum(np.log(z * (1 - z)), axis=1, keepdims=True) - np.sum(
            np.log(self.scale_)
        )

    def inverse_jacobian_determinant(self, X):
        z = expit(X)
        return np.prod(z * (1 - z), axis=1, keepdims=True) * np.prod(self.scale_)

    def inverse_log_jacobian_determinant(self, X):
        z = expit(X)
        return np.sum(np.log(z * (1 - z)), axis=1, keepdims=True) + np.sum(
            np.log(self.scale_)
        )


class xyDataset(torch.utils.data.Dataset):
    """
    Joins the x and y into a dataset, so that it can be used by the pythorch syntax.
    """

    def __init__(self, x, y, weights=None):
        self.x = torch.tensor(x).to(torch.float)
        self.y = torch.tensor(y).to(torch.float)

        if weights is not None:
            self.weights = torch.tensor(weights).to(torch.float)
        else:
            self.weights = None

    def __len__(self):
        return len(self.x)

    def to(self, device=""):
        self.x = self.x.to(device=device)
        self.y = self.y.to(device=device)
        if self.weights is not None:
            self.weights = self.weights.to(device=device)
        return self

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        if self.weights is None:
            sample = [self.x[idx], self.y[idx]]
        else:
            sample = [self.x[idx], self.y[idx], self.weights[idx]]

        return sample


class DNN(torch.nn.Module):
    """NN for vanilla classifier."""

    def __init__(self, input_dim, layers=[16, 32, 16], dropout_probability=0.0):
        super(DNN, self).__init__()

        self.dpo = dropout_probability

        self.batchnorm = torch.nn.BatchNorm1d(input_dim)
        self.inputlayer = torch.nn.Linear(input_dim, layers[0])
        self.outputlayer = torch.nn.Linear(layers[-1], 1)

        all_layers = [
            self.batchnorm,
            self.inputlayer,
            torch.nn.SiLU(),
            torch.nn.Dropout(self.dpo),
        ]
        for i in range(len(layers) - 1):
            all_layers.append(torch.nn.Linear(layers[i], layers[i + 1]))
            all_layers.append(torch.nn.LeakyReLU())
            all_layers.append(torch.nn.Dropout(self.dpo))

        all_layers.append(self.outputlayer)
        all_layers.append(torch.nn.Sigmoid())
        self.layers = torch.nn.Sequential(*all_layers)

    def forward(self, x):
        """Forward pass through the DNN"""
        x = self.layers(x)
        return x


def flow_model(
    nfeatures, num_layers=3, hidden_size=32, RQS=True, num_blocks=2, num_bins=8
):

    # hidden_size = 2*nfeatures
    base_dist = ConditionalDiagonalNormal(
        shape=[nfeatures], context_encoder=nn.Linear(1, 2 * nfeatures)
    )

    if RQS:
        flow_fn = lambda: MaskedPiecewiseRationalQuadraticAutoregressiveTransform(
            features=nfeatures,
            hidden_features=hidden_size,
            context_features=1,
            num_bins=num_bins,
            tails="linear",
            tail_bound=5.0,
            num_blocks=num_blocks,
            use_residual_blocks=True,
            activation=nn.ReLU(),
        )
    else:
        flow_fn = lambda: MaskedAffineAutoregressiveTransform(
            features=nfeatures, hidden_features=hidden_size, context_features=1
        )

    transforms = []
    for _ in range(num_layers):
        transforms.append(flow_fn())
        transforms.append(ReversePermutation(features=nfeatures))

        transforms.append(BatchNorm(nfeatures))

    transform = CompositeTransform(transforms)

    flowB = Flow(transform, base_dist)
    return flowB


def flow_model_with_syst(
    nfeatures, num_layers=3, hidden_size=32, RQS=True, num_blocks=2, num_bins=8
):

    # hidden_size = 2*nfeatures
    base_dist = ConditionalDiagonalNormal(
        shape=[nfeatures], context_encoder=nn.Linear(2, 2 * nfeatures)
    )

    if RQS:
        flow_fn = lambda: MaskedPiecewiseRationalQuadraticAutoregressiveTransform(
            features=nfeatures,
            hidden_features=hidden_size,
            context_features=2,
            num_bins=num_bins,
            tails="linear",
            tail_bound=5.0,
            num_blocks=num_blocks,
            use_residual_blocks=True,
            activation=nn.ReLU(),
        )
    else:
        flow_fn = lambda: MaskedAffineAutoregressiveTransform(
            features=nfeatures, hidden_features=hidden_size, context_features=2
        )

    transforms = []
    for _ in range(num_layers):
        transforms.append(flow_fn())
        transforms.append(ReversePermutation(features=nfeatures))

        transforms.append(BatchNorm(nfeatures))

    transform = CompositeTransform(transforms)

    flowB = Flow(transform, base_dist)
    return flowB


class EarlyStopper:
    def __init__(self, patience=1, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = float("inf")

    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False


# create a mask that removes signal events to enforce a given fraction
# randomly selects which signal events to remove
def get_signal_mask(labels, num_sig, seed=42):

    num_events = labels.shape[0]
    cur_sig = np.sum(labels > 0)
    num_drop = int(cur_sig - num_sig)
    keep_idxs = np.array([True] * num_events)
    if num_drop < 0:
        return keep_idxs
    all_idxs = np.arange(num_events, dtype=np.int32)
    sig_idxs = all_idxs[labels.reshape(-1) > 0]

    np.random.seed(seed)
    drop_sigs = np.random.choice(sig_idxs, num_drop, replace=False)
    keep_idxs[drop_sigs] = False
    return keep_idxs


def clean_nans(inputs, above_avg=None):
    # cleanup nans from the data,
    #'high' nans to max, low nans to min
    if above_avg is None:
        above_avg = inputs > 0
    nans = np.isnan(inputs)
    inputs_max = np.amax(inputs[~nans])
    inputs_min = np.amin(inputs[~nans])
    inputs[nans & above_avg] = inputs_max
    inputs[nans & ~above_avg] = inputs_min
    return inputs


# Fitting related stuff


def pdf_DCB(x, beta_left, m_left, scale_left, beta_right, m_right, scale_right, loc):
    return cb.pdf(
        x, beta_left, m_left, scale_left, beta_right, m_right, scale_right, loc
    )


class ExpShape:
    def __init__(self, start, stop, order=2, num_integral=False):
        super().__init__()
        # polynomial order
        self.order = order

        # fit range
        self.start = start
        self.stop = stop
        self.num_integral = num_integral

    def norm(self, pars, start=None, stop=None):
        if self.order == 2:
            norm = (pars[0] / pars[1]) * (1 - np.exp(-pars[1]))
        else:
            if start is None:
                start = self.start
            if stop is None:
                stop = self.stop
            if self.num_integral:
                norm = integrate.quad(self.density, start, stop, pars)[0]
            else:
                # Use analytic formula
                # https://www.wolframalpha.com/input?i2d=true&i=integral+from+0+to+1+of+p0+*+exp%5C%2840%29-x*p1+-Power%5Bx%2C2%5D*p2%5C%2841%29
                def indef_integral(x):
                    out = (
                        pars[0]
                        * ((np.pi / pars[2]) ** 0.5 / 2)
                        * np.exp(pars[1] ** 2 / (4 * pars[2]))
                        * erf((pars[1] + 2.0 * pars[2] * x) / (2 * pars[2] ** 0.5))
                    )
                    return out

                norm = indef_integral(stop) - indef_integral(start)
        return norm

    def density(self, x, pars):
        if self.order == 2:
            base = pars[0] * np.exp(-pars[1] * x)
        if self.order == 3:
            base = pars[0] * np.exp(-pars[1] * x - pars[2] * x * x)
        return base

    def pdf(self, x, pars):
        norm = self.norm(pars)
        base = self.base(x, pars)
        return base / norm


class BernShape:
    def __init__(self, start, stop, order=2):
        super().__init__()
        # polynomial order
        self.order = order

        # fit range
        self.start = start
        self.stop = stop

    def norm(self, pars):
        norm = bernstein.integral([self.stop], pars, self.start, self.stop)[0]
        return norm

    def density(self, x, pars):
        out = bernstein.density(x, pars, self.start, self.stop)
        return out


# piecewise functions for systematics

# We have
# f(syst) = uplim^syst if syst > 1
# f(syst) = 1 + sum_{i=1}^{6} a_i syst^i if -1 <= syst <= 1
# f(syst) = lowlim^-syst if syst < -1


def piecewise_exp_coefficient(uplim, lowlim):
    ### uplim and lowlim are $g_{+}$ and $g_{-}$
    A = np.zeros((6, 6))
    A[0] = [(1) ** i for i in range(1, 7)]
    A[1] = [(-1) ** i for i in range(1, 7)]
    A[2] = [i * (1) ** (i - 1) for i in range(1, 7)]
    A[3] = [i * (-1) ** (i - 1) for i in range(1, 7)]
    A[4, 1:] = [i * (i - 1) * (1) ** (i - 2) for i in range(2, 7)]
    A[5, 1:] = [i * (i - 1) * (-1) ** (i - 2) for i in range(2, 7)]

    b = np.array(
        [
            uplim - 1.0,
            lowlim - 1.0,
            uplim * np.log(uplim),
            -lowlim * np.log(lowlim),
            uplim * np.log(uplim) ** 2,
            lowlim * np.log(lowlim) ** 2,
        ]
    )

    a = np.linalg.solve(A, b)

    return a


def piecewise_exp(syst, uplim, lowlim, a):
    if syst >= 1:
        return uplim**syst
    elif np.abs(syst) < 1:
        return np.ones(a.shape[1]) + np.einsum(
            "ik,i->k", a, np.array([(syst) ** (power + 1) for power in range(6)])
        )
    else:
        return lowlim ** (-syst)


def piecewise_quadratic_coefficient(uplim, lowlim):
    ### uplim and lowlim are $g_{+}$ and $g_{-}$

    return np.array([0.5 * (uplim - lowlim), 0.5 * (uplim + lowlim - 2)])


def piecewise_quadratic(syst, uplim, lowlim, a):
    return 1 + a[0] * (syst) + a[1] * (syst) ** 2


def make_postfit_plot(
    data=None,
    weights=None,
    pdfs=None,
    flow_probs=None,
    templates=None,
    template_samples=None,
    template_norms=None,
    template_weights=None,
    labels=None,
    colors=None,
    axis_label=None,
    title=None,
    bins=10,
    normalize=False,
    h_range=None,
    y_label="Events/bin",
    fname="",
    ratio_range=-1,
    bottom_panel="ratio",
    logy=False,
    max_rw=5,
    draw_chi2=True,
    do_norm=True,
    leg_loc="upper right",
):
    h_type = "step"
    alpha = 1.0
    fontsize = 26
    label_size = 38
    leg_size = 26

    lw = 3
    # matplotlib.rcParams['text.usetex'] =  True

    fig_size = (12, 9)
    fig = plt.figure(figsize=fig_size)
    gs = gridspec.GridSpec(2, 1, height_ratios=[3, 1])
    ax0 = plt.subplot(gs[0])

    if h_range is None and type(bins) == int:
        low = np.amin(data)
        high = np.amax(data)
    elif h_range is not None:
        low, high = h_range
    else:
        low, high = bins[0], bins[-1]

    if weights is None:
        weights = np.ones_like(data)

    data_binned, bins = np.histogram(
        data, bins=bins, range=(low, high), weights=weights
    )
    data_uncs = np.sqrt(data_binned)

    bin_centers = (bins[:-1] + bins[1:]) / 2.0

    num_bins = len(bins) - 1

    bin_widths = np.diff(bins)
    norm = 1.0

    # normalize to density by dividing by bin width
    if do_norm:
        norm = 1.0 * bin_widths[0]  # counts * bin width
    else:
        norm = 1.0
    # data_binned /= norm
    # data_uncs /=norm

    nsteps = 500
    tot_fit = np.zeros(nsteps)
    tot_fit_binned = np.zeros(num_bins)
    bin_sizes = (high - low) / nsteps
    x_vals = np.linspace(low, high, nsteps)

    bkg_fit_binned = None

    # don't draw pdf below this
    pdf_thresh = 1e-2

    # analytic pdfs
    if pdfs is not None:
        all_pdf_vals = []
        for i in range(len(pdfs)):
            pdf_vals = pdfs[i](x_vals)
            all_pdf_vals.append(pdf_vals)

            mask = pdf_vals > pdf_thresh

            plt.plot(
                x_vals[mask],
                pdf_vals[mask] * norm,
                color=colors[i],
                label=labels[i],
                linewidth=lw,
            )

            tot_fit += pdf_vals * norm

            bin_integrals = np.array(
                [
                    integrate.quad(pdfs[i], bins[j], bins[j + 1])[0]
                    for j in range(num_bins)
                ]
            )

            tot_fit_binned += bin_integrals

            if i == 0:
                bkg_fit_binned = bin_integrals

        tot_color = "gray"
        # plt.step(bins, tot_fit, color=tot_color, label= 'Total Fit', where='post' )
        plt.plot(x_vals, tot_fit, color=tot_color, label="Total Fit", linewidth=lw)

    if template_samples is not None:
        templates = []
        # samples to make a binned template
        for i in range(len(template_samples)):
            if template_weights is None:
                t_weights = np.ones(template_samples[i].shape[0])
            else:
                t_weights = template_weights[i]
            t_weights = (template_norms[i] / template_samples[i].shape[0]) * t_weights
            templ_vals, _ = np.histogram(
                template_samples[i], weights=t_weights, bins=bins
            )
            templates.append(templ_vals)

    if templates is not None:
        bkg_fit_binned = templates[0]
        prev = np.zeros_like(templates[0])
        # stack stair plots on top of each other
        for i in range(len(templates)):
            tot_fit_binned += templates[i]
            plt.stairs(
                tot_fit_binned,
                bins,
                baseline=prev,
                color=colors[i],
                fill=True,
                label=labels[i],
            )
            prev = copy.copy(tot_fit_binned)

    # not used
    if flow_probs is not None:
        for i in range(len(flow_probs)):
            # compute avg prob for events in that bin
            probs = np.zeros(num_bins)
            for j in range(num_bins):
                mask = (data > bins[j]) & (data < bins[j + 1])
                bin_val = np.mean(flow_probs[i][mask])
                probs[j] = bin_val

            rescale = np.sum(data_binned) / np.sum(probs)
            probs *= rescale
            plt.stairs(probs, bins, color=colors[i], label=labels[i])
            tot_fit_binned += probs

    # draw data on top
    data_color = "black"
    data_drawn, _, _ = ax0.errorbar(
        bin_centers,
        data_binned,
        yerr=data_uncs,
        label="data",
        fmt="o",
        markersize=10,
        linewidth=3,
        markerfacecolor=data_color,
        ecolor=data_color,
        markeredgecolor=data_color,
    )

    plt.xlim([low, high])
    if logy:
        plt.yscale("log")

    eps = 1e-6
    mask = data_uncs > eps
    chi2 = np.sum(((data_binned[mask] - tot_fit_binned[mask]) / (data_uncs[mask])) ** 2)

    ax1 = plt.subplot(gs[1])

    if bottom_panel == "ratio":
        ratio = np.clip(data_binned, 1e-8, None) / np.clip(tot_fit_binned, 1e-8, None)
        ratio_uncs = np.clip(data_uncs, 1e-8, None) / np.clip(
            tot_fit_binned, 1e-8, None
        )
        ax1.errorbar(
            bin_centers[mask],
            ratio[mask],
            yerr=ratio_uncs[mask],
            color=data_color,
            fmt="o",
            markersize=10,
            linewidth=3,
            markerfacecolor=data_color,
            ecolor=data_color,
            markeredgecolor=data_color,
        )

        if type(ratio_range) == list or type(ratio_range) == tuple:
            plt.ylim(ratio_range[0], ratio_range[1])
        else:
            if ratio_range > 0:
                plt.ylim([1 - ratio_range, 1 + ratio_range])

        ax1.set_ylabel("Ratio", fontsize=label_size, loc="center")

    elif bottom_panel == "subtract":
        diff = data_binned - bkg_fit_binned
        ax1.errorbar(
            bin_centers[mask],
            diff[mask],
            yerr=data_uncs[mask],
            color=data_color,
            fmt="o",
            markersize=10,
            linewidth=3,
            markerfacecolor=data_color,
            ecolor=data_color,
            markeredgecolor=data_color,
        )
        ax1.set_ylabel("Data - Bkg", fontsize=label_size, loc="center")

        # draw signal in bottom panel, assume it is last entry
        if pdfs is not None:
            mask = all_pdf_vals[-1] > pdf_thresh
            plt.plot(
                x_vals[mask],
                all_pdf_vals[-1][mask] * norm,
                color=colors[-1],
                linewidth=lw,
            )

        if templates is not None:
            plt.stairs(templates[-1], bins, color=colors[-1], fill=False)

    # leg = ax0.legend(leg_list, labels, loc=leg_loc, fontsize = leg_size)
    leg = ax0.legend(loc=leg_loc, fontsize=leg_size)
    leg.set_title(title)

    ax0.set_ylabel(y_label, fontsize=label_size)

    ax0.tick_params(
        which="major", axis="y", bottom=False, top=False, labelsize=label_size * 0.7
    )
    ax1.tick_params(
        which="major", axis="x", bottom=False, top=False, labelsize=label_size * 0.7
    )
    ax1.tick_params(
        which="major", axis="y", bottom=False, top=False, labelsize=label_size * 0.7
    )

    xlabel_size = label_size
    ax1.set_xlabel(axis_label, fontsize=xlabel_size)
    plt.subplots_adjust(top=1.0)

    ax0.tick_params(which="minor", axis="x", bottom=False, top=False)
    ax0.tick_params(which="major", axis="x", bottom=False, top=False)
    plt.setp(ax0.get_xticklabels(), visible=False)

    plt.xlim([low, high])

    plt.grid(axis="y")
    plt.sca(ax0)
    y_max = ax0.get_ylim()[1] * 1.7
    plt.ylim([None, y_max])

    # plt.title(title, fontsize=fontsize)
    if draw_chi2:
        plt.sca(ax0)
        plt.gcf().canvas.draw()
        if leg_loc == "best":
            y_val = ax0.get_ylim()[1] * 0.7 - ax0.get_ylim()[0]
            x_val = ax0.get_xlim()[1] * 0.1 - ax0.get_xlim()[0]
        elif leg_loc == "upper left":
            y_val = 0.7 * (ax0.get_ylim()[1] - ax0.get_ylim()[0]) + ax0.get_ylim()[0]
            x_val = 0.53 * (ax0.get_xlim()[1] - ax0.get_xlim()[0]) + ax0.get_xlim()[0]
        elif leg_loc == "upper right":
            y_val = 0.82 * (ax0.get_ylim()[1] - ax0.get_ylim()[0]) + ax0.get_ylim()[0]
            x_val = 0.2 * (ax0.get_xlim()[1] - ax0.get_xlim()[0]) + ax0.get_xlim()[0]

        ndof = len(bins) - 1
        print("Chi2/ndof = %.3f / %i" % (chi2, ndof))
        txt = r"$\chi^2$ / ndof = %.0f / %i" % (chi2, ndof)
        plt.text(
            x_val,
            y_val,
            txt,
            color="gray",
            horizontalalignment="left",
            fontweight="bold",
            fontsize=fontsize,
        )
        y_val -= y_val * 0.06

    if fname != "":
        plt.savefig(fname, bbox_inches="tight")
        # fname_pdf = fname.replace(".png", ".pdf")
        # plt.savefig(fname_pdf, bbox_inches = "tight")
        print("saving fig %s" % fname)

    plt.close()

    return


def convert_mgg(m, low=90, high=180):
    return (m - low) / (high - low)


def reverse_mgg(x, low=90, high=180):
    return x * (high - low) + low


def one_sig_CI(best_fit, s_values, s_profile):
    # range of 2*dNLL = 1 crossings
    dLL = 2.0 * (s_profile - np.min(s_profile))
    eps = 0.1
    dLL_high = dLL[(s_values > best_fit) & (dLL > eps)]
    dLL_low = dLL[(s_values < best_fit) & (dLL > eps)]

    try:
        upper_spline = CubicSpline(
            dLL_high, s_values[(s_values > best_fit) & (dLL > eps)]
        )
        upper = upper_spline(1.0)
        upper = upper - best_fit
    except:
        print("Issue with upper spline! Aborting finding proper 1 sigma CI")
        print(dLL_high)
        upper = 1.0
    try:
        # reverse order so monotonically increasing
        lower_spline = CubicSpline(
            dLL_low[::-1], s_values[(s_values < best_fit) & (dLL > eps)][::-1]
        )
        lower = lower_spline(1.0)
        lower = best_fit - lower
    except:
        print("Issue with lower spline! Aborting finding proper 1 sigma CI")
        print(dLL_low)
        lower = 1.0
    return lower, upper
