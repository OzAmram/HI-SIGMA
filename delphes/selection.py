#!/usr/bin/env python
import sys, os 
import numpy as np
import ROOT
import h5py

ROOT.gSystem.Load("libDelphes")

try:
    ROOT.gInterpreter.Declare('#include "classes/DelphesClasses.h"')
    ROOT.gInterpreter.Declare('#include "external/ExRootAnalysis/ExRootTreeReader.h"')
except:
    pass



def append_h5(f, name, data):
    prev_size = f[name].shape[0]
    f[name].resize(( prev_size + data.shape[0]), axis=0)
    f[name][prev_size:] = data


class Outputer:
    def __init__(self, outputFileName="out.h5", batch_size = 5000):

        self.batch_size = batch_size
        self.output_name = outputFileName
        self.first_write = False
        self.idx = 0
        self.nBatch = 0
        self.nfeats = 11
        self.reset()

    def reset(self):
        self.idx = 0
        self.bjets = np.zeros((self.batch_size, 8), dtype=np.float32)
        self.photons = np.zeros((self.batch_size, 6), dtype=np.float32)
        self.feats = np.zeros((self.batch_size, self.nfeats), dtype=np.float32)

    
    def fill_event(self, bjet1, bjet2, photon1, photon2):
        
        self.bjets[self.idx] = [bjet1.pt(), bjet1.eta(), bjet1.phi(), bjet1.mass(),
                                bjet2.pt(), bjet2.eta(), bjet2.phi(), bjet2.mass()]

        self.photons[self.idx] = [photon1.pt(), photon1.eta(), photon1.phi(),
                                  photon2.pt(), photon2.eta(), photon2.phi()]
                    
        
        dR_11 = ROOT.Math.VectorUtil.DeltaR(bjet1, photon1)
        dR_12 = ROOT.Math.VectorUtil.DeltaR(bjet1, photon2)
        dR_21 = ROOT.Math.VectorUtil.DeltaR(bjet2, photon1)
        dR_22 = ROOT.Math.VectorUtil.DeltaR(bjet2, photon2)

        dR_bb = ROOT.Math.VectorUtil.DeltaR(bjet1, bjet2)
        dR_gg = ROOT.Math.VectorUtil.DeltaR(photon1, photon2)

        Hgg = (photon1 + photon2)
        Hbb = (bjet1 + bjet2)
        tot = Hgg + Hbb

        self.feats[self.idx] = [tot.mass(), Hbb.mass(), Hgg.mass(), Hbb.pt(), Hgg.pt(), dR_11, dR_12, dR_21, dR_22, dR_bb, dR_gg]


        self.idx +=1
        if(self.idx % self.batch_size == 0): self.write_out()


    def write_out(self):
        self.idx = 0
        print("Writing out batch %i \n" % self.nBatch)
        self.nBatch += 1

        if(not self.first_write):
            self.first_write = True
            print("First write, creating dataset with name %s \n" % self.output_name)
            with h5py.File(self.output_name, "w") as f:
                f.create_dataset("bjets", data=self.bjets, chunks = True, maxshape=(None, 8))
                f.create_dataset("photons", data=self.photons, chunks = True, maxshape=(None, 6))
                f.create_dataset("feats", data=self.feats, chunks = True, maxshape=(None, self.nfeats))

        else:
            with h5py.File(self.output_name, "a") as f:
                append_h5(f,'bjets',self.bjets)
                append_h5(f,'photons',self.photons)
                append_h5(f,'feats',self.feats)
        self.reset()

    def final_write_out(self ):
        if(self.idx < self.batch_size):
            print("Last batch only filled %i events, shortening arrays \n" % self.idx)
            self.bjets = self.bjets[:self.idx]
            self.photons = self.photons[:self.idx]
            self.feats = self.feats[:self.idx]

        self.write_out()


if len(sys.argv) < 3:
    print(" Usage: Example1.py input_file outfile")
    sys.exit(1)

infile = sys.argv[1]
outfile = sys.argv[2]

if(".root" in infile): 
    flist = [infile]
else:
    flist = open(infile).read().splitlines()


# Create chain of root trees
chain = ROOT.TChain("Delphes")
for file in flist:
    chain.Add(file)

# Create object of class ExRootTreeReader
treeReader = ROOT.ExRootTreeReader(chain)
numberOfEntries = treeReader.GetEntries()
print("Running over %i events" % numberOfEntries)

output = Outputer(outfile)

H_bb_mass = []
H_gg_mass = []

branchJet = treeReader.UseBranch("Jet")
branchPhoton = treeReader.UseBranch("Photon")

pt_cut = 30.
pt_cut_sublead = 20.
eta_cut = 2.4

dR_cut = True

# Loop over all events
count = 0
for entry in range(0, numberOfEntries):
    #if(entry > 10): exit(1)
    # Load selected branches with data from specified event
    treeReader.ReadEntry(entry)

    bjet1 = bjet2 = phot1 = phot2 = None


    for jet in branchJet:
        #print("jet", jet.PT, jet.Eta, jet.BTag)
        if(jet.PT > pt_cut and abs(jet.Eta) < eta_cut and jet.BTag):
        #if(jet.PT > pt_cut and abs(jet.Eta) < eta_cut):
            if(bjet1 is None): bjet1 = jet
            elif(bjet2 is None): bjet2 = jet


    for phot in branchPhoton:
        #print("phot", phot.PT, phot.Eta)
        if(phot.PT > pt_cut_sublead and abs(phot.Eta) < eta_cut):
            if(phot.PT > pt_cut and phot1 is None): phot1 = phot
            elif(phot2 is None): phot2 = phot


    if((bjet1 is None) or (bjet2 is None) or (phot1 is None) or (phot2 is None)):
        continue

    count +=1
    bjet1_vec = ROOT.Math.PtEtaPhiMVector(bjet1.PT, bjet1.Eta, bjet1.Phi, bjet1.Mass)
    bjet2_vec = ROOT.Math.PtEtaPhiMVector(bjet2.PT, bjet2.Eta, bjet2.Phi, bjet2.Mass)

    phot1_vec = ROOT.Math.PtEtaPhiMVector(phot1.PT, phot1.Eta, phot1.Phi, 0.)
    phot2_vec = ROOT.Math.PtEtaPhiMVector(phot2.PT, phot2.Eta, phot2.Phi, 0.)
    #H_bb = bjet1_vec + bjet2_vec
    #H_gg = phot1_vec + phot2_vec
    #print("Higgs cands \n", H_bb.mass(), H_gg.mass())

    if(dR_cut):
        dR_11 = ROOT.Math.VectorUtil.DeltaR(bjet1_vec, phot1_vec)
        dR_12 = ROOT.Math.VectorUtil.DeltaR(bjet1_vec, phot2_vec)
        dR_21 = ROOT.Math.VectorUtil.DeltaR(bjet2_vec, phot1_vec)
        dR_22 = ROOT.Math.VectorUtil.DeltaR(bjet2_vec, phot2_vec)

        if( (dR_11 < 1.0) or (dR_12 < 1.0) or (dR_21 < 1.0) or (dR_22 < 1.0)): continue

    output.fill_event(bjet1_vec, bjet2_vec, phot1_vec, phot2_vec)


print("%i / %i events passed (%.3f)"% (count, numberOfEntries, count /numberOfEntries))
output.final_write_out()


