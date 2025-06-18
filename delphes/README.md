# Scripts to process delphes files

A basic preselection is applied and then the events are saved into an h5 file
format.

The h5 file has 3 fields

*'bjets'* : (X, 8). The 4-vectors (pt, eta, phi, mass) of the two bjets  in the
event. The first one is the higher pt jet

*'photons'* : (X, 6). The 3-vectors (pt, eta, phi) of the two photons  in the
event. The first one is the higher pt photon


*'feats'* : (X, 11). Useful pre-computed features of the di Higgs system
The features are : (tot\_mass Hbb\_mass Hgg\_mass, Hbb\_pt, Hgg\_pt, dR\_11, dR\_12, dR\_21, dR\_22, dR\_bb, dR\_gg
tot\_mass is the mass of the total bb gam gam system
Hbb and Hgg are the H to bb and H to gam gam candidates
dR\_ij  are the deltaR's between the ith bjet and j-th photon

basic\_plots makes some plots of all these variables from the saved h5 file
