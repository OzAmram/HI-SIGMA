# HI-SIGMA 

Code base for 'Data-Driven High Dimensional Statistical Inference with Generative Models' [arXiv:2506.06438](https://arxiv.org/abs/2506.06438).
The (h5) datasets used can be found on [Zenodo](https://doi.org/10.5281/zenodo.15587841). 

Code to apply basic preselections and process the Delphes data into the h5 format can be found in the `delphes/` subdirectory. 


The basic scripts to reproduce the results of the paper (starting from the provided h5 datasets) are:

**create_dataset.py** : Performs a train/test split to the signal and
background datasets. Applies chosen data smearing and saves preprocessors

**check_feats.py** : Makes plots of the different signal vs background feature distributions.

**train_classifier.py** : Trains a supervised classifier.
It trains a version both with and without the resonant feature included.
It also constructs templates of the signal and background shapes, to be used for
fits.

**train_flows.py** : Trains the normalizing flows used as density estimators.
Options consist on whether to train on signal or background datasets, and whether to
include systematic variations. 

**bkg_check.py**: Performs some checks of the quality of the learned background estimate.

**do_fits.py** : Performs statistical inference with the trained flow models,
    and comparison methods (including classifier).

**draw_postfit.py**: Makes summary plots of the results of the different fits. 

**compare_classifier.py**: Makes a ROC curve based on the S(x)/B(x) from the density estimate vs
the supervised classifier to compare. 


For all scripts there are hardcoded directory names and whatnot at the top of
the file that need to be changed (sorry).
