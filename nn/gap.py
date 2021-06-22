# gap.py
# (c) 2013 Mikael Vejdemo-Johansson  
# BSD License
#
# SciPy function to compute the gap statistic for evaluating k-means clustering.
# Gap statistic defined in
# Tibshirani, Walther, Hastie:
#  Estimating the number of clusters in a data set via the gap statistic
#  J. R. Statist. Soc. B (2001) 63, Part 2, pp 411-423
# Available here: https://gist.github.com/michiexile/5635273

# Maria Korosteleva: Re-written for PyTorch allowing GPU-only operations
# PyTorch & GPU friendliness 
import torch 
from kmeans_pytorch import kmeans as km_torch

from sklearn.cluster import KMeans  # https://gist.github.com/michiexile/5635273#gistcomment-1437301
from sklearn.exceptions import ConvergenceWarning
import warnings  # to catch the clustering warning

import numpy as np  # for consistent isclose check

import scipy
import scipy.cluster.vq
import scipy.spatial.distance
dst = scipy.spatial.distance.euclidean


# If requested number of clusters = 1
def single_cluster_torch(data):
    cluster_center = data.mean(dim=0).unsqueeze(0)
    labels = torch.zeros(data.shape[0], dtype=torch.int, device=data.device)

    return labels, cluster_center


def gap(data, data_torch=None, refs=None, nrefs=20, ks=range(1, 11)):
    """
        Compute the Gap statistic for an nxm dataset in data.
        Either give a precomputed set of reference distributions in refs as an (n,m,k) scipy array,
        or state the number k of reference distributions in nrefs for automatic generation with a
        uniformed distribution within the bounding box of data.
        Give the list of k-values for which you want to compute the statistic in ks.
    """
    shape = data.shape
    if refs==None:
        tops = data.max(axis=0)
        bots = data.min(axis=0)
        dists = scipy.matrix(scipy.diag(tops - bots))
        rands = scipy.random.random_sample(size=(shape[0],shape[1],nrefs))
        for i in range(nrefs):
            rands[:,:,i] = rands[:,:,i] * dists + bots
    else:
        rands = refs

    gaps = np.zeros((len(ks),))
    labels = []
    for (i,k) in enumerate(ks):    
        with warnings.catch_warnings():
            # https://stackoverflow.com/questions/48100939/how-to-detect-a-scikit-learn-warning-programmatically
            warnings.filterwarnings('error', category=ConvergenceWarning)
            try:
                kmeanModel = KMeans(n_clusters=k).fit(data)
                labels.append(kmeanModel.labels_)
                (kmc,kml) = kmeanModel.cluster_centers_, kmeanModel.labels_
                disp = sum([dst(data[m,:],kmc[kml[m],:]) for m in range(shape[0])])

                if np.isclose(disp, 0.):
                    gaps[i] = None
                    continue

                refdisps = scipy.zeros((rands.shape[2],))
                for j in range(rands.shape[2]):
                    kmeanModel = KMeans(n_clusters=k).fit(rands[:,:,j])
                    (kmc,kml) = kmeanModel.cluster_centers_, kmeanModel.labels_
                    refdisps[j] = sum([dst(rands[m,:,j],kmc[kml[m],:]) for m in range(shape[0])])

            except ConvergenceWarning as w:
                gaps[i] = None  
                break  # next iterations will also have convergence warning

        if np.isclose(disp, 0.) or np.allclose(refdisps, 0.):  # degenerate cases
            gaps[i] = None
        else:
            # flipped mean & log https://gist.github.com/michiexile/5635273#gistcomment-2324237
            gaps[i] = scipy.mean(scipy.log(refdisps)) - scipy.log(disp)
    return gaps, labels[-1] if len(labels) else None


def gap_torch(data_torch, refs=None, nrefs=20, ks=range(1, 11)):
    """
        Compute the Gap statistic for an nxm dataset in data.
        Either give a precomputed set of reference distributions in refs as an (n,m,k) scipy array,
        or state the number k of reference distributions in nrefs for automatic generation with a
        uniformed distribution within the bounding box of data.
        Give the list of k-values for which you want to compute the statistic in ks.
    """
    shape_t = data_torch.shape
    if refs is None:
        tops_t, _ = data_torch.max(dim=0)
        bots_t, _ = data_torch.min(dim=0)
        dists_t = tops_t - bots_t 

        # uniform distribution
        rands_t = torch.rand((nrefs, shape_t[0], shape_t[1]), device=data_torch.device)
        rands_t = rands_t * dists_t + bots_t

    else:
        rands = refs

    gaps_t = torch.zeros(len(ks))
    zero_t = torch.zeros(1, device=data_torch.device)
    labels = []
    for (i, k) in enumerate(ks):    
        with warnings.catch_warnings():
            # https://stackoverflow.com/questions/48100939/how-to-detect-a-scikit-learn-warning-programmatically
            warnings.filterwarnings('error', category=ConvergenceWarning)
            try:
                # pytorch
                if k == 1:
                    labels_t, cluster_centers_t = single_cluster_torch(data_torch)
                else:
                    labels_t, cluster_centers_t = km_torch(X=data_torch, num_clusters=k, device=data_torch.device, tqdm_flag=False)
                    labels_t, cluster_centers_t = labels_t.to(data_torch.device), cluster_centers_t.to(data_torch.device)
                
                labels.append(labels_t)
                disp_t = sum([torch.dist(data_torch[m], cluster_centers_t[labels_t[m]]) for m in range(shape_t[0])])

                if torch.isclose(disp_t, zero_t):
                    gaps_t[i] = None
                    continue

                refdisps_t = torch.zeros(nrefs, device=data_torch.device)
                for j in range(nrefs):
                    if k == 1:
                        labels_t, cluster_centers_t = single_cluster_torch(rands_t[j])
                    else:
                        labels_t, cluster_centers_t = km_torch(X=rands_t[j], num_clusters=k, device=rands_t.device, tqdm_flag=False)
                        labels_t, cluster_centers_t = labels_t.to(rands_t.device), cluster_centers_t.to(rands_t.device)

                    refdisps_t[j] = sum([torch.dist(rands_t[j, m], cluster_centers_t[labels_t[m]]) for m in range(shape_t[0])])

                    # TODO any sort of warning in degenerate cases here?

            except ConvergenceWarning as w:
                gaps[i] = None  
                break  # next iterations will also have convergence warning

        # pytorch
        if torch.isclose(disp_t, zero_t) or torch.allclose(refdisps_t, zero_t):  # degenerate cases
            gaps_t[i] = None
        else:
            # flipped mean & log https://gist.github.com/michiexile/5635273#gistcomment-2324237
            gaps_t[i] = torch.mean(torch.log(refdisps_t)) - torch.log(disp_t)

    return gaps_t, labels[-1] if len(labels) else None
